import os
import torch
from core import losses
from core.networks import UnrollReg
from data.datasets import BrainMRInterSubj3D, CardiacMR2D, CardiacMR2D_MM, CardiacCMRA3D


def get_network(hparams):
    """Configure network"""
    network = UnrollReg(**hparams.network.config)
    return network


def get_loss_fn(hparams):
    """ Configure similarity and regularisation loss functions """
    sim_loss_fn = {'mse': torch.nn.MSELoss(), 'ncc': losses.ncc}[hparams.loss.sim_loss.type]
    sim_loss_fn = sim_loss_fn(**hparams.loss.sim_loss.config) if hparams.loss.sim_loss.config else sim_loss_fn
    reg_loss_fn = {'diffusion': losses.diffusion_loss}[
        hparams.loss.reg_loss.type] if hparams.loss.reg_loss.type else None
    return sim_loss_fn, reg_loss_fn


def get_loss_fn_unroll(hparams):
    # train_loss_fn = torch.nn.MSELoss(reduction='sum')
    train_loss_fn = torch.nn.MSELoss()
    reg_loss_fn = losses.diffusion_loss
    return train_loss_fn, reg_loss_fn


def get_datasets(hparams):
    assert os.path.exists(hparams.data.train_path), \
        f"Training data path does not exist: {hparams.data.train_path}"
    assert os.path.exists(hparams.data.val_path), \
        f"Validation data path does not exist: {hparams.data.val_path}"

    if hparams.data.name == 'brain_camcan':
        train_dataset = BrainMRInterSubj3D(data_dir=hparams.data.train_path,
                                           limit_data=hparams.data.train_limit_data,
                                           crop_size=hparams.data.crop_size,
                                           resample_size=hparams.data.resample_size,
                                           modality=hparams.data.modality,
                                           atlas_path=hparams.data.atlas_path)

        val_dataset = BrainMRInterSubj3D(data_dir=hparams.data.val_path,
                                         crop_size=hparams.data.crop_size,
                                         resample_size=hparams.data.resample_size,
                                         modality=hparams.data.modality,
                                         evaluate=True,
                                         atlas_path=hparams.data.atlas_path)

    elif hparams.data.name == 'cardiac_ukbb':
        train_dataset = CardiacMR2D(hparams.data.train_path,
                                    limit_data=hparams.data.train_limit_data,
                                    crop_size=hparams.data.crop_size,
                                    slicing=hparams.data.train_slicing,
                                    batch_size=hparams.data.batch_size
                                    )
        val_dataset = CardiacMR2D(hparams.data.val_path,
                                  crop_size=hparams.data.crop_size,
                                  slicing=hparams.data.val_slicing
                                  )

    elif hparams.data.name == 'cardiac_mm':
        train_dataset = CardiacMR2D_MM(hparams.data.train_path,
                                       crop_size=hparams.data.crop_size,
                                       spacing=hparams.data.spacing,
                                       original_spacing=hparams.data.original_spacing,
                                       slicing=hparams.data.train_slicing,
                                       limit_data=hparams.data.train_limit_data,
                                       batch_size=hparams.data.batch_size
                                       )
        val_dataset = CardiacMR2D_MM(hparams.data.val_path,
                                     crop_size=hparams.data.crop_size,
                                     slicing=hparams.data.val_slicing,
                                     spacing=hparams.data.spacing,
                                     original_spacing=hparams.data.original_spacing
                                     )

    elif hparams.data.name == 'cardiac_cmra':
        train_dataset = CardiacCMRA3D(hparams.data.train_path,
                                      limit_data=hparams.data.train_limit_data,
                                      crop_size=hparams.data.crop_size,
                                      # slice_range = hparams.data.train_slice_range,
                                      slicing=hparams.data.train_slicing,
                                      binindex=hparams.data.train_binindex,
                                      durationindex=hparams.data.train_durationindex,
                                      batch_size=hparams.data.batch_size
                                      )
        val_dataset = CardiacCMRA3D(hparams.data.val_path,
                                    crop_size=hparams.data.crop_size,
                                    slicing=hparams.data.val_slicing,
                                    binindex=hparams.data.val_binindex,
                                    durationindex=hparams.data.val_durationindex
                                    )
    else:
        raise ValueError(f'Dataset config ({hparams.data.name}) not recognised.')

    return train_dataset, val_dataset
