# Deliverable D3.1

This repository contains the software provided for the deliverable D3.1 of the HIDDEN project, Work Package (WP) 3 (AI-based advanced collective awareness).

The components follow the established HIDDEN system architecture (D2.2) and consider the previously defined specifications and requirements (D2.1).

## Repository structure

Each top-level folder contains the software of the components of one of the corresponding tasks of WP3.

| Folder | Task Name | Task Nr. |
| --- | --- | --- |
| [01_SAP](01_SAP) | Occlusion-aware single-agent object detection and tracking | Task 3.1 |
| [02_DGT](02_DGT) | Driver gaze tracking and status monitoring | Task 3.2 |
| [03_CA](03_CA) | Occlusion-aware failsafe collective awareness aided by the driver's attention area | Task 3.3 |

Inside each task folder, the subfolders, each corresponding to one component, are named after their specified name in the HIDDEN architecture.

```
D3.1/
├── 01_SAP/                     # Single-agent perception (Task 3.1)
│   ├── CTAG_lidar_ODTH
│   ├── DENSO_uwb_ODT
│   ├── Late_Fusion_ODT
│   ├── UFR_cam_ASEG
│   └── UPV_cam_lidar_ODTH
├── 02_DGT/                     # Driver gaze tracking and status monitoring (Task 3.2)
│   ├── CTAG_ATTEN
│   ├── LIBRA_camera_EM
│   ├── LIBRA_CIBOS_OD_GAZE
│   ├── Merge_DS
│   └── Merge_FR
├── 03_CA/                      # Collective awareness (Task 3.3)
│   ├── CUT_FUS_SP_DG
│   ├── ICCS_FUS_SP_DG_CA
│   └── UFR_FUS_SP_DG_CA
└── README.md
```
