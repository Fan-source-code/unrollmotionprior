import os
import hydra
from hydra.core.hydra_config import HydraConfig

from omegaconf import DictConfig
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer

from models import LitUnrollReg
from utils.misc import MyModelCheckpoint

import pytorch_lightning as pl

import random
random.seed(7)


import logging
log = logging.getLogger(__name__)


@hydra.main(config_path="conf/train", config_name="config1res")
def main(cfg: DictConfig) -> None:
    model_dir = HydraConfig.get().run.dir

    # use only one GPU
    gpu_list = '0'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list


    # lightning model
    model = LitUnrollReg(**cfg)

    # configure logger
    logger = TensorBoardLogger(model_dir, name='log')

    # model checkpoint callback with ckpt metric logging
    ckpt_callback = pl.callbacks.ModelCheckpoint(save_last=True,
                                                 dirpath=f'{model_dir}/checkpoints/',
                                                 filename='{epoch}-{val_metrics/val_loss:.3f}',
                                                 monitor='val_metrics/val_loss',
                                                 mode='min',
                                                 save_top_k=-1,
                                                 verbose=True)

    trainer = pl.Trainer(default_root_dir=model_dir,
                         logger=logger,
                         callbacks=[ckpt_callback],
                         gradient_clip_val=0.1,
                         devices=[0],
                         precision=cfg.precision,
                         **cfg.training.trainer)

    # run training
    trainer.fit(model)
    ckpt_callback.best_model_path


if __name__ == "__main__":
    main()
    