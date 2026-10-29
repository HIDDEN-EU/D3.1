import json
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import time

import pytorch_lightning as pl
import logging
from pathlib import Path
import pandas as pd
from os import path as osp

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from tqdm import tqdm

import logging

from src.utils.util_fn import read_jpg, read_pcd
logging.getLogger(__file__)

class DAIRV2X_DATASET(Dataset):
    """
    Dataset for detection in DAIRV2XC,currently for veh-side cam+lid detection
    Interface with DAIRV2XDataModule
    """
    def __init__(self, 
                 split:list,
                 desc:str,
                 data_path_veh_cam:str,
                 data_path_veh_lid:str,
                 path_to_data_info:str,
                 path_to_gt_labels:str,
                 path_to_veh_calib:str,
                 test_k_elem:int,
                 test_k:bool,
                 class_names
                 ):
        """
        data_path_veh_camera = DAIRV2X-C vehicle-side images 
        data_path_veh_lidar = DAIRV2X-C vehicle-side lidar
        path_to_data_info:str,=data_info.json, frame-label correspondsance
        path_to_gt_labels:str,  = label information (bbox) 
        path_to_veh_calib:str = calibration for vehicle frame (intrinsic, extrinsics)
        
        Read dataset for vehicle only detection
        """
        super().__init__()
        #------  metadata,gt,calib
        self.path_to_data_info = Path(path_to_data_info) #data anotations/metadata
        self.path_to_gt_labels = Path(path_to_gt_labels)
        self.path_to_veh_calib = path_to_veh_calib

        #--- inputs
        self.data_path_veh_cam = Path(data_path_veh_cam) #data paths 
        self.data_path_veh_lid = Path(data_path_veh_lid)

        #conf
        self.test_k_elem=test_k_elem
        self.test_k=test_k
      
        #classes
        self.class_names=tuple(class_names)
        self.class_to_id = {
            name: i for i, name in enumerate(self.class_names)
        }

        ##calib(vehicle)
        self.camera_intrinsics,\
        self.camera2lidar,\
        self.camera2ego,\
        self.lidar2ego,\
        self.img_aug_matrix,\
        self.lidar_aug_matrix=self.read_calib_files_once(frame_idx="000000")


        self.print_stats=True

        #class statistics
        self.statistics_classes={"Car":0,
                                 "Truck":0,
                                  "Van":0,
                                 "Bus":0,
                                 "Pedestrian":0,
                                 "Cyclist":0,
                                 "Tricyclist":0,
                                 "Motorcyclist":0,
                                 "Barrowlist":0,
                                 "Trafficcone":0}

        #-----database
        self.database = self.get_veh_cam_lid_frame_id(split , desc)  #construct database list from data_info of DAIRV2X 

        if self.print_stats:
            print("Printing statistics for classes found in DAIRV2X dataset ,{}".format(self.statistics_classes))

    def __len__(self):
        return len(self.database)
    
    def read_calib_files_once(self,frame_idx):
        """
        Read calibration for one timestamp/frame. Assuming 
        that these calibration files for the vehicle frame (intrinsics and extrinsics with relation to lidar etc remain constant which is true)

        Returns
        -------
        camera_intrinsics : [1, 4, 4]
            Camera intrinsic matrix. First dimension is N=1 camera.

        camera2lidar : [1, 4, 4]
            Transform from vehicle camera frame -> vehicle LiDAR frame.

        lidar2camera : [1, 4, 4]
            Transform from vehicle LiDAR frame -> vehicle camera frame.

        cam_D : [num_distortion_coeffs]
            Distortion coefficients.
        """
        

        with open(osp.join(self.path_to_veh_calib,"camera_intrinsic",frame_idx + ".json") , "r") as f:
            calib = json.load(f) 
        camera_intinsics = calib["cam_K"]

        K = torch.tensor(
            camera_intinsics,
            dtype=torch.float32,
        ).reshape(3, 3)

        # BEVFusion uses homogeneous 4x4 calibration matrices
        K4 = torch.eye(4, dtype=torch.float32)
        K4[:3, :3] = K

        # N = 1 camera
        camera_intrinsics = K4.unsqueeze(0)
        # [1,4,4]

        with open(osp.join(self.path_to_veh_calib , "lidar_to_camera", frame_idx + ".json") , "r") as f:
            calib_ext=json.load(f)

        R_lidar_to_camera = torch.tensor(
            calib_ext["rotation"],
            dtype=torch.float32,
        ).reshape(3, 3)

        t_lidar_to_camera = torch.tensor(
            calib_ext["translation"],
            dtype=torch.float32,
        ).reshape(3)
        lidar2camera = torch.eye(
            4,
            dtype=torch.float32,
        )

        lidar2camera[:3, :3] = R_lidar_to_camera
        lidar2camera[:3, 3] = t_lidar_to_camera

        # camera2lidar=[calib["rotation"] , calib["translation"]]
        camera2lidar = torch.linalg.inv(
            lidar2camera
        )

        # Add N=1 camera dimension
        lidar2camera = lidar2camera.unsqueeze(0)
        camera2lidar = camera2lidar.unsqueeze(0)

        camera2ego=camera2lidar.clone()
        lidar2ego=torch.eye(4)
        img_aug_matrix = torch.eye(
                4,
                dtype=torch.float32,
            )
        lidar_aug_matrix = torch.eye(
                4,
                dtype=torch.float32,
        )

        return (camera_intrinsics,
                camera2lidar, 
                camera2ego, 
                lidar2ego , 
                img_aug_matrix,
                lidar_aug_matrix)
    
    def __getitem__(self, idx):
        """
        fetcher

        Reads also the img, pcd from disk one at a time 
        """
        start = time.perf_counter()
        sample = self.database[idx]
        img = sample["img"]
        points= sample["points"]
        frame_idx = sample["frame_idx"]

        gt = sample["ground_truth_veh_cam"]  #multiple 3dbboxes at current time instance

        # ------------------------
        # Camera                |
        # ------------------------
        img = read_jpg(img)
        # numpy: [H, W, 3], uint8, BGR
        img = torch.from_numpy(img).permute(2,0,1).float()  # [3, H, W]

        # img=img.unsqueeze(0) #add camera node dimension N=1, required by lss

        #some info  
        logging.info("Outputting image shape from dataloader, after reading, and permuting 2,0,1 {}".format(img.shape))

        # ------------------------
        # LiDAR                 |
        # ------------------------
        points = read_pcd(points)

        logging.info("Outputting some points shape information after reading in dataset fetcher {}".format(points.shape))

        # points_batch = [torch.from_numpy(x).float() for x in points]
        # [N, 4]

        # (camera_intrinsics,
        #  camera2lidar, 
        #  camera2ego, 
        #  lidar2ego , 
        #  img_aug_matrix, 
        #  lidar_aug_matrix)= self.read_calib_path(frame_idx)
        camera_intrinsics=self.camera_intrinsics
        camera2lidar=self.camera2lidar
        camera2ego=self.camera2ego
        lidar2ego=self.lidar2ego
        img_aug_matrix=self.img_aug_matrix
        lidar_aug_matrix=self.lidar_aug_matrix

        elapsed = time.perf_counter() - start
        logging.info(f"Elapsed time in data fetch {elapsed}")
        print(f"Elapsed time in data fetch {elapsed}")

        return (img, 
                points , 
                gt , 
                camera_intrinsics,
                camera2lidar,
                camera2ego,
                lidar2ego,
                img_aug_matrix,
                lidar_aug_matrix
                )


    def read_calib_path(self , frame_idx:str):
        """
        Read calibration for one timestamp/frame.

        Returns
        -------
        camera_intrinsics : [1, 4, 4]
            Camera intrinsic matrix. First dimension is N=1 camera.

        camera2lidar : [1, 4, 4]
            Transform from vehicle camera frame -> vehicle LiDAR frame.

        lidar2camera : [1, 4, 4]
            Transform from vehicle LiDAR frame -> vehicle camera frame.

        cam_D : [num_distortion_coeffs]
            Distortion coefficients.
        """
        

        with open(osp.join(self.path_to_veh_calib,"camera_intrinsic",frame_idx + ".json") , "r") as f:
            calib = json.load(f) 
        camera_intinsics = calib["cam_K"]

        K = torch.tensor(
            camera_intinsics,
            dtype=torch.float32,
        ).reshape(3, 3)

        # BEVFusion uses homogeneous 4x4 calibration matrices
        K4 = torch.eye(4, dtype=torch.float32)
        K4[:3, :3] = K

        # N = 1 camera
        camera_intrinsics = K4.unsqueeze(0)
        # [1,4,4]

        with open(osp.join(self.path_to_veh_calib , "lidar_to_camera", frame_idx + ".json") , "r") as f:
            calib_ext=json.load(f)

        R_lidar_to_camera = torch.tensor(
            calib_ext["rotation"],
            dtype=torch.float32,
        ).reshape(3, 3)

        t_lidar_to_camera = torch.tensor(
            calib_ext["translation"],
            dtype=torch.float32,
        ).reshape(3)
        lidar2camera = torch.eye(
            4,
            dtype=torch.float32,
        )

        lidar2camera[:3, :3] = R_lidar_to_camera
        lidar2camera[:3, 3] = t_lidar_to_camera

        # camera2lidar=[calib["rotation"] , calib["translation"]]
        camera2lidar = torch.linalg.inv(
            lidar2camera
        )

        # Add N=1 camera dimension
        lidar2camera = lidar2camera.unsqueeze(0)
        camera2lidar = camera2lidar.unsqueeze(0)

        camera2ego=camera2lidar.clone()
        lidar2ego=torch.eye(4)
        img_aug_matrix = torch.eye(
                4,
                dtype=torch.float32,
            )
        lidar_aug_matrix = torch.eye(
                4,
                dtype=torch.float32,
        )

        return (camera_intrinsics,
                camera2lidar, 
               camera2ego, 
               lidar2ego , 
               img_aug_matrix,
               lidar_aug_matrix)


    def parse_camera_label(self,label_camera):
        """
        parse object detection ground truth for vehicle camera, at one timestamp
        Read Labels file that contains all labels seen by vehilce at current timestamp 
        accorind to data_info.json and append it to output 

        returns 
        ground_truth_data: list of 2d_bboxes at current timestamp
        """
        with open(osp.join(self.path_to_gt_labels , label_camera), "r") as f:
            ground_truth_file=json.load(f)
        ground_truth_data=[]
        for obj_dict in ground_truth_file:
            obj_id = self.class_to_id[obj_dict["type"]]  #conv class str to int
            ground_truth_data.append({"type":obj_id,
                                        "height":obj_dict["3d_location"]["z"],
                                        "bbox":(obj_dict["3d_location"]["x"] , 
                                                obj_dict["3d_location"]["y"]),
                                        "3d_dimensions":obj_dict["3d_dimensions"],
                                        "rotation":obj_dict["rotation"],
                                      })  #centerpoint needs 3d dimensions
            #check validity of obj, size can cause infinite training problems/wrong annotations
            size=obj_dict["3d_dimensions"]
            dims=torch.tensor([size["w"] , size["l"], size["h"]],dtype=torch.float32)
            if not (torch.isfinite(dims).all() and (dims>0).all()):
                raise ValueError("Found invalid sized object of type{} and bbox{} and size {}, at label file {}".format(obj_id,(obj_dict["3d_location"]["x"] , 
                                                obj_dict["3d_location"]["y"]),size, osp.join(self.path_to_gt_labels,label_camera) ))
             

            #find actual statistics
            self.statistics_classes[obj_dict["type"]] = self.statistics_classes[obj_dict["type"]]+1

        return ground_truth_data
    
    def get_veh_cam_lid_frame_id(self, split:list , split_desc:str):
        """ 
        generator for getting sequential dairv2x cam/lid corresponding inputs from data_info.json
        
        constructor for the detection database. Data_info reports sequentialy all frames and
        associated labels, pcd files at each teimestamp. Cross check with labels files to find labels
        of that frame.

        inputs
            split: list of frames to be used by current split
        
        Returns
        database: dict of inputs,ground truth imgs, pcd files , bboxes respectively
        """
        with open(self.path_to_data_info, "r") as f:
            data = json.load(f)
        database = [] 
        read_k_elements=self.test_k_elem
        c=0
        for elem in tqdm(data , desc=split_desc):
            c=c+1 
            frame_idx = elem["image_path"].split("/")[-1].replace(".jpg", "")

            if frame_idx not in split:
                continue   #keep only frame idx in current split

            label_camera = elem["label_camera_std_path"]        
            label_lidar = elem["label_lidar_std_path"]  
            img_path = self.data_path_veh_cam / elem["image_path"]                  #paths
            lidar_path = self.data_path_veh_lid / elem["pointcloud_path"]

            #ground truth veh cam returns bbox 3D properties, rotation
            try:
                ground_truth_veh_cam = self.parse_camera_label(label_camera)  # #ground truth labels(3dbbox) (list of bboxes)
            except ValueError as e:
                print("Skipping object/invalid size")
                print(e)
                continue

            
            # ------------------------
            # GT
            # ------------------------


            database.append({                               #training sample 
                             "img":img_path , 
                             "points":lidar_path , 
                             "ground_truth_veh_cam":ground_truth_veh_cam,  #2d bbox
                             "frame_idx": frame_idx
                                                                         })  
            
            if c>=read_k_elements and self.test_k :break  #for initial prototype

            
        logging.info("Loaded {} data for current {} split".format(len(database),split_desc))

        return database


class DAIRV2XDataModule(pl.LightningDataModule):
    """
    datamodule for dairv2x detection 
    """
    def __init__(
        self,
        data_path_veh_cam:str,
        data_path_veh_lid:str,
        path_to_data_info:str,
        path_to_gt_labels:str,
        path_to_data_splits:str,
        path_to_veh_calib:str,
        test_k_elem:int,
        test_k:bool,
        class_names,
        batch_size=128,
        num_workers=4,
        pin_memory=True,
    ):
        super().__init__()
        self.class_names =class_names

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory


        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        #requirements
        self.data_path_veh_cam=data_path_veh_cam    #input paths
        self.data_path_veh_lid=data_path_veh_lid
        self.path_to_data_info=path_to_data_info    #data (paths,labels) info
        self.path_to_gt_labels=path_to_gt_labels    #gt info
        self.path_to_data_splits=path_to_data_splits #train-val-test splits info
        self.path_to_veh_calib= path_to_veh_calib
        #splitting
        self.train_splits,self.val_splits,self.test_splits = self.parse_data_splits(self.path_to_data_splits) #get splits


        #conf
        self.test_k_elem=test_k_elem
        self.test_k=test_k

    def parse_data_splits(self,path_to_data_splits):
        """
        return frame idxs lists as splits 
        """
        with open(path_to_data_splits, "r") as f:
            splits = json.load(f)["vehicle_split"]

        train_splits = [x for x in splits["train"]]
        val_splits = [x for x in splits["val"]]
        test_splits = [x for x in splits["val"]]

        assert len(test_splits)>0
        # print(len(test_splits))
        # sys.exit(0)

        return train_splits,val_splits,test_splits

    def setup(self, stage=None):
        """
        setups current training stage. 
        fit:train and val
        test: test
        """
        if stage == "fit" or stage is None:
            self.train_dataset = DAIRV2X_DATASET(
                split=self.train_splits, 
                data_path_veh_cam=self.data_path_veh_cam,
                data_path_veh_lid=self.data_path_veh_lid,
                path_to_data_info=self.path_to_data_info,
                path_to_gt_labels=self.path_to_gt_labels,
                path_to_veh_calib=self.path_to_veh_calib,
                desc="train",
                test_k_elem=self.test_k_elem,
                test_k=self.test_k,
                class_names=self.class_names
            )

            self.val_dataset = DAIRV2X_DATASET(
                split=self.val_splits, 
                data_path_veh_cam=self.data_path_veh_cam,
                data_path_veh_lid=self.data_path_veh_lid,
                path_to_data_info=self.path_to_data_info,
                path_to_gt_labels=self.path_to_gt_labels,
                path_to_veh_calib=self.path_to_veh_calib,
                desc="val",
                test_k_elem=self.test_k_elem,
                test_k=self.test_k,
                class_names=self.class_names
            )

        #testing
        if stage == "test":
            self.test_dataset = DAIRV2X_DATASET(
                split=self.test_splits, 
                data_path_veh_cam=self.data_path_veh_cam,
                data_path_veh_lid=self.data_path_veh_lid,
                path_to_data_info=self.path_to_data_info,
                path_to_gt_labels=self.path_to_gt_labels,
                path_to_veh_calib=self.path_to_veh_calib,
                desc="test",
                test_k_elem=self.test_k_elem,
                test_k=self.test_k,
                class_names=self.class_names
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers ,
            collate_fn=self.bevfusion_collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers,
            collate_fn=self.bevfusion_collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers,
            collate_fn=self.bevfusion_collate_fn,
        )
    
    def bevfusion_collate_fn(self, batch):
            """
            simple dict based collator
            
            TODO handle pcd? 
            """
            images = torch.stack(
                [sample[0] for sample in batch]
            )

            points = torch.stack([
                sample[1]
                for sample in batch
            ])

            gt_boxes = [  #sample is a list of detections
                sample[2]
                for sample in batch
            ]

            camera_intrinsics=[sample[3] for sample in batch]

            camera2lidar=  [
                sample[4]
                for sample in batch
            ]

            camera2ego =  [
                sample[5]
                for sample in batch
            ]

            lidar2ego =  [
                sample[6]
                for sample in batch
            ]

            img_aug_matrix=  [
                sample[7]
                for sample in batch
            ]

            lidar_aug_matrix= [
                sample[8]
                for sample in batch
            ]

            #make tensors out of calib matrices 
            camera_intrinsics = torch.stack(camera_intrinsics) 
            camera2lidar=torch.stack(camera2lidar)
            camera2ego=torch.stack(camera2ego)
            lidar2ego=torch.stack(lidar2ego)
            img_aug_matrix=torch.stack(img_aug_matrix)
            lidar_aug_matrix=torch.stack(lidar_aug_matrix)


            return {
                "image": images,          # [B, 3, H, W]

                # lists because number differs per sample
                "points": points,        # list of [Ni, 4]
                "gt_boxes": gt_boxes,    # list of [Mi, box_dim]
                "camera_intrinsics": camera_intrinsics,
                "camera2lidar": camera2lidar,
                "camera2ego": camera2ego,
                "lidar2ego": lidar2ego,
                "img_aug_matrix": img_aug_matrix,
                "lidar_aug_matrix": lidar_aug_matrix

                # "camera_intrinsics": torch.stack(
                #     [sample["camera_intrinsics"] for sample in batch]
                # ),

                # "camera_extrinsics": torch.stack(
                #     [sample["camera_extrinsics"] for sample in batch]
                # ),
            }
