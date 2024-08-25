import torch
from pytorch_lightning import LightningModule
# from pytorch_lightning.loggers.base import merge_dicts
from torch.optim import Adam
from torch.utils.data import DataLoader

from core.transformations import warp
from core.utils import interpolate_nd
from models.utils import get_network, get_loss_fn, get_datasets, get_loss_fn_unroll
from utils.metrics import measure_metrics
#from utils.misc import worker_init_fn
from utils.visualise import visualise_result


class LitDLReg(LightningModule):
    """ DL registration base Lightning module"""
    def __init__(self, *args, **kwargs):
        super(LitDLReg, self).__init__()
        self.save_hyperparameters()

        self.network = get_network(self.hparams)
        self.train_dataset, self.val_dataset = get_datasets(self.hparams)

        self.train_loss_fn, self.reg_loss_fn = get_loss_fn_unroll(self.hparams)

        self.hparam_metrics = {f'hparam_metrics/{m}': 0.0 for m in self.hparams.hparam_metrics}

    def on_fit_start(self):
        # log dummy initial hparams w/ best metrics (for tensorboard HPARAMS tab)
        self.logger.log_hyperparams(self.hparams, metrics=self.hparam_metrics)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.hparams.data.batch_size,
                          shuffle=self.hparams.data.shuffle,
                          num_workers=self.hparams.data.num_workers,
                          pin_memory=self.on_gpu,
                          #worker_init_fn=worker_init_fn
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=1,
                          shuffle=False,
                          num_workers=self.hparams.data.num_workers,
                          pin_memory=self.on_gpu,
                          #worker_init_fn=worker_init_fn
                          )

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.training.lr)
        if self.hparams.training.lr_decay_step:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                        step_size=self.hparams.training.lr_decay_step,
                                                        gamma=0.8,
                                                        last_epoch=-1)
            return [optimizer], [scheduler]
        else:
            return [optimizer]

    def forward(self, tar, src):
        # pure network forward pass
        out = self.network(tar, src)
        return out

    def inference(self, batch):
        """ Forward pass and warping source image """

        raise NotImplementedError

    def loss_fn(self, outputs, batch):

        raise NotImplementedError

    def forward_and_loss(self, batch):
        """ Forward pass inference + compute loss """
        outputs = self.inference(batch)
        loss, losses = self.loss_fn(outputs, batch)
        return loss, losses, outputs

    def training_step(self, batch, batch_idx):
        train_loss, train_losses, train_outputs = self.forward_and_loss(batch)
        self._log_train_metrics(batch, train_loss, train_losses, train_outputs)

        return train_loss

    def _log_train_metrics(self, batch, train_loss, train_losses, train_outputs):
        self.log('train_loss/train_loss', train_loss)
        self.log_dict({f'train_loss/{k}': loss for k, loss in train_losses.items()})



    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            for k, x in batch.items():
                # 2D: cardiac-only (1, N_slices, H, W) -> (N_slices, 1, H, W)
                # 3D: (1, 1, D, H, W)
                batch[k] = x.transpose(0, 1)
            val_loss, val_losses, val_outputs = self.forward_and_loss(batch)
        self.log('val_metrics/val_loss', val_loss)
        val_losses = {k: loss.cpu() for k, loss in val_losses.items()}

        return {'val_loss': val_loss.cpu(), **val_losses}

