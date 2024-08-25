"""Run model inference and save outputs for analysis"""
import os
import time
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.datasets import BrainMRInterSubj3D, CardiacMR2D, CardiacMR2D_MM, CardiacCMRA3D
from models.gradirns import LitGraDIRN
from models.unrollreg import LitUnrollReg
from models.unrollregls import LitUnrollRegls
from models.base import LitDLReg
from core.transformations import warp
from core.utils import create_img_pyramid, interpolate_nd
from utils.image_io import save_nifti, load_nifti
from utils.metrics import measure_metrics, MetricReporter
from utils.misc import setup_dir

import random
random.seed(7)

DATASETS = {'cardiac_cmra': CardiacCMRA3D}
DL_MODELS = {'unrollreg': LitUnrollReg}


def get_test_dataloader(cfg, pin_memory=False):
    dataset = DATASETS[cfg.data.type](**cfg.data.dataset)
    return DataLoader(dataset,
                      shuffle=False,
                      pin_memory=pin_memory,
                      **cfg.data.dataloader)


def get_test_model(cfg, device=torch.device('cpu')):
    if cfg.model.type in DL_MODELS.keys():
        model = DL_MODELS[cfg.model.type].load_from_checkpoint(cfg.model.ckpt_path, strict=True)
        model = model.to(device=device)
        model.eval()
        print(cfg.model.type)
    else:
        raise ValueError(f"Unknown inference model type: {cfg.model.type}")
    return model


def inference(cfg, model, dataloader, output_dir, model_type=None, device=torch.device('cpu')):
    print('---------------------')
    print("Running inference...")
    binindex = cfg.data.dataset.binindex
    loss = torch.nn.MSELoss()
    # model.eval()
    for idx, batch in enumerate(tqdm(dataloader)):
        for k, x in batch.items():
            # reshape data for inference
            # 2d: (batch_size=1, num_slices, H, W) -> (num_slices, batch_size=1, H, W)
            # 3d: (batch_size=1, 1, D, H, W) -> (1, batch_size=1, D, H, W) only works with batch_size=1
            batch[k] = x.transpose(0, 1).to(device=device)

        # model inference
        with torch.no_grad():
            if model_type == 'unrollreg':

                tars = [batch['tar']]
                srcs = [batch['src']]

                out = model(tars, srcs)

                
            else:
                out = model(batch['tar'], batch['src'])

        # save the outputs
        subj_id = dataloader.dataset.subject_list[idx]
        output_subj_dir = setup_dir(f'{output_dir}/{subj_id}')

        with torch.no_grad():

            disp_net = out[0][-1].cpu().numpy()


        save_nifti(disp_net, path=f'{output_subj_dir}/bin{binindex}disp3.nii.gz')




@hydra.main(config_path="conf/test", config_name="config")
def main(cfg: DictConfig) -> None:

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = torch.device("cuda:0")

    print("=============================")
    print("Device: ", device)
    print("=============================")



    # run inference
    run_dir = HydraConfig.get().run.dir

    for durationindex in range(1):
        cfg.data.dataset.durationindex = durationindex
        output_dir = setup_dir(f'{run_dir}/outputs/{int(durationindex + 1)}')
        for binindex in range(3):
            cfg.data.dataset.binindex = binindex

            # configure dataset & model
            test_dataloader = get_test_dataloader(cfg, pin_memory=(device is torch.device('cuda')))

            # run inference
            if cfg.inference:
                test_model = get_test_model(cfg, device=device)
                inference(cfg, test_model, test_dataloader, output_dir, model_type=cfg.model.type, device=device)



if __name__ == '__main__':
    main()
