from datetime import datetime
from omegaconf import DictConfig, OmegaConf

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import logging
from src.logging_config import setup_logging

root_logger = logging.getLogger()

print("Logging level:", root_logger.level)

for handler in root_logger.handlers:
    print(
        "handler:",
        type(handler),
        type(handler),
        getattr(handler, "baseFilename", None),
    )

import hydra
from hydra.utils import instantiate
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import logging

@hydra.main(
    config_path="configs",
    config_name="config"
)
def main(cfg:DictConfig):
    """
    train/validate
    
    For training
        - set stage to "fit" in datamodule setup
        - set configs/data to dairv2x_detection

    """
    setup_logging()  #python logging
    
    logging.info("Starting main training sequence...")

    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)

    datamodule = instantiate(cfg.data)
                  
    model = instantiate(cfg.model)   

    success = datamodule.setup(stage="fit")

    hparams = OmegaConf.to_container(
        cfg,
        resolve=True,
        throw_on_missing=True
    )


    logger = instantiate(cfg.logger)        #tensorboard logger
    logger.log_hyperparams(hparams)
    
    logger.experiment.add_text(
            "hydra_config",
            f"<pre>{OmegaConf.to_yaml(cfg)}</pre>",
            global_step=0,
    )

    checkpoint_callback = ModelCheckpoint(              #saves checkpoints (callback)
            dirpath="/mnt/e/dev/hidden/projects/detection_dairv2x/checkpoints",
            filename=f"epoch={{epoch:02d}}-val={{val:.4f}}-name={cfg.experiment_name}",
            monitor=cfg.modelcheckpoint_monitor,
            mode="min",
            save_top_k=-1,          # save all checkpoints
            every_n_epochs=10,
            save_on_train_epoch_end=False,
            save_last=True,
         )

 
    trainer = instantiate(
        cfg.trainer,
        callbacks=[checkpoint_callback],
        logger=logger,   #add extra hydra conf param
        log_every_n_steps=1,
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        overfit_batches=5,
        limit_train_batches=5,  # Run 5 training batches
        limit_val_batches=2,    # Run 2 validation batches
        num_sanity_val_steps=0, # Skip preliminary validation
        max_epochs=1, #for profiling
    )

    # print("Last model path:", checkpoint_callback.last_model_path)

    # with open("training_logs/checkpoint_info_{}.log".format(cfg.experiment_name), "a") as f:
    #     print("Callbacks:", file=f)

    #     for cb in trainer.callbacks:
    #         print(
    #             type(cb),
    #             getattr(cb, "dirpath", None),
    #             file=f,
    #         )

    #     print("Checkpoint dir:", checkpoint_callback.dirpath, file=f)
    #     print("Best model path:", checkpoint_callback.best_model_path, file=f)
    #     print("Last model path:", checkpoint_callback.last_model_path, file=f) #save callbacks 
    print("Lightning version:", pl.__version__)
    print("Working directory:", Path.cwd())
    print("Active logger:", trainer.logger)
    print("TensorBoard directory:", logger.log_dir)



    trainer.fit(model = model , 
                datamodule=datamodule)
    


    print("\nFinal Metrics:")
    for k, v in trainer.callback_metrics.items():
        print(f"{k}: {v.item():.6f}")
    print(trainer.callback_metrics)

    


if __name__ == "__main__":
        main()
