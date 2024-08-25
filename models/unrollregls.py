import torch

import matplotlib.pyplot as plt

from core.transformations import warp, warp_fn
from core.utils import create_img_pyramid
from models.base import LitDLReg
from utils.visualise import visualise_seq_results


class LitUnrollRegls(LitDLReg):
    def __init__(self, *args, **kwargs):
        super(LitUnrollRegls, self).__init__(*args, **kwargs)

    def inference(self, batch):
        num_resolutions = len(self.hparams.network.config.DC_num_blocks)
        tars = create_img_pyramid(batch['tar'], num_resolutions)
        srcs = create_img_pyramid(batch['src'], num_resolutions)
        out, zs = self.forward(tars, srcs)
        return {'disp': out, 'tars': tars, 'srcs': srcs, 'z': zs}

    def loss_fn(self, outputs, batch):
        # only compute loss on the last output
        # disp_label = batch['disp_gt'].squeeze().view(1,2,208,208)
        tar = outputs['tars'][-1]
        src = outputs['srcs'][-1]
        disp = outputs['disp'][-1]
        disp_label = batch['disp_gt'].squeeze().view(disp.shape)
        z = outputs['z'][-1]
        grid = getattr(self.network, f'grid_lvl{self.network.num_resolutions-1}')
        warped_src = warp_fn(src, disp, grid)
        losses = {}
        #reguweight_ = self.network.reg_blocks[-1][-1].reguweight_
        # reguweight_ = self.network.regu_weight
        reguweight_ = self.network.reg_blocks[-1][-1].reguweight_
        # cost = (self.train_loss_fn(tar, warped_src) + reguweight_*self.train_loss_fn(disp, z))/tar.shape[0]
        costdc = self.train_loss_fn(tar, warped_src)
        costreg = self.train_loss_fn(disp, z)
        losses['cost'] = costdc+reguweight_*costreg
        losses['cost_dc'] = costdc
        losses['cost_reg'] = costreg
        
        loss = self.train_loss_fn(disp, disp_label)
        losses['train_loss'] = loss
        
        # sim_loss = self.train_loss_fn(tar, warped_src) * self.hparams.loss.sim_loss.weight
        # losses['sim_loss'] = sim_loss     
        # loss = sim_loss
        # reg_loss = self.reg_loss_fn(disp) * self.hparams.loss.reg_loss.weight
        # losses['reg_loss'] = reg_loss
        # loss = loss + reg_loss
        
        # (dis-)similarity loss
        # sim_loss = self.sim_loss_fn(tar, warped_src) * self.hparams.loss.sim_loss.weight
        # losses['sim_loss'] = sim_loss
        # loss = sim_loss
        # # regularisation loss
        # if self.reg_loss_fn:
        #     reg_loss = self.reg_loss_fn(disp) * self.hparams.loss.reg_loss.weight
        #     losses['reg_loss'] = reg_loss
        #     loss = loss + reg_loss
        return loss, losses

    def _log_train_metrics(self, batch, train_loss, train_losses, train_outputs):
        super(LitUnrollRegls, self)._log_train_metrics(batch, train_loss, train_losses, train_outputs)
        # tau_list = [DispUpdate.tau_ for dc_blocks_lvl in self.network.DC_blocks for DClayer in dc_blocks_lvl for DispUpdate in DClayer.dispupdate_blocks ]
        # self.log_dict({f'tau/block_{n}': tau_ for n, tau_ in enumerate(tau_list)})
        # regu_list = [reg_blocks_lvl[0].reguweight_ for reg_blocks_lvl in self.network.reg_blocks]
        # self.log_dict({f'regu_weight/block_{n}': weight_ for n, weight_ in enumerate(regu_list)})
        regu_list = [self.network.reg_blocks[lvl][-1].reguweight_ for lvl in range(self.network.num_resolutions)]
        self.log_dict({f'regu_weight/resolution_{n}': weight_ for n, weight_ in enumerate(regu_list)})

    # def _log_validation_visual(self, batch_idx, batch, val_outputs):
    #     # (optional) log validation visual
    #     if batch_idx == 0 and (self.current_epoch+1) % self.hparams.training.log_visual_every_n_epoch == 0:
    #         self._log_visual(batch, val_outputs, stage='val')

    # def _log_visual(self, batch, outputs, stage='val'):
    #     with torch.no_grad():
    #         # log images and transformation
    #         # visual_data = {k: list()
    #         #                for k in ['tars', 'srcs', 'warped_srcs', 'tar_segs', 'src_segs',
    #         #                          'warped_src_segs', 'errors', 'errors_seg', 'disps']}
    #         visual_data = {k: list()
    #                        for k in ['tars', 'srcs', 'warped_srcs', 'zs',
    #                                   'errors', 'disps']}

    #         # generate segmentation pyramids
    #         #tar_segs = create_img_pyramid(batch['tar_seg'].cpu(), lvls=self.network.num_resolutions, label=True)
    #         #src_segs = create_img_pyramid(batch['src_seg'].cpu(), lvls=self.network.num_resolutions, label=True)

    #         # iterator to iterate through the list of disps
    #         disps_iter = iter(outputs['disp'])
    #         zs_iter = iter(outputs['z'])

    #         for lvl in range(self.network.num_resolutions):
    #             for _ in range(self.network.DC_num_blocks[lvl] * self.network.DC_num_repeat[lvl]):
    #                 tar = outputs['tars'][lvl].detach().cpu()
    #                 src = outputs['srcs'][lvl].detach().cpu()
    #                 z = next(zs_iter).detach().cpu()
    #                 #tar_seg = tar_segs[lvl]
    #                 #src_seg = src_segs[lvl]
    #                 disp = next(disps_iter).detach().cpu()
    #                 warped_src = warp(src, disp)
    #                 error = tar - warped_src
    #                 #warped_src_seg = warp(src_seg, disp, interp_mode='nearest')
    #                 #error_seg = tar_seg - warped_src_seg

    #                 # visual_data_n = {'tars': tar, 'srcs': src, 'warped_srcs': warped_src,
    #                 #                  'tar_segs': tar_seg, 'src_segs': src_seg, 'warped_src_segs': warped_src_seg,
    #                 #                  'errors': error, 'errors_seg': error_seg, 'disps': disp}
                    
    #                 visual_data_n = {'tars': tar, 'srcs': src, 'warped_srcs': warped_src, 'zs': z,
    #                                  'errors': error, 'disps': disp}
    #                 assert visual_data_n.keys() == visual_data.keys()
    #                 for k in visual_data.keys():
    #                     visual_data[k].append(visual_data_n[k].numpy())

    #     fig = visualise_seq_results(visual_data)
    #     self.logger.experiment.add_figure(f'{stage}_visual', fig, global_step=self.global_step, close=True)

    # def _log_energy(self, batch, stage='val'):
    #     # log energy
    #     self.network.compute_energy = True
    #     # run forward pass again to populate network.energy
    #     _, _, _ = self.forward_and_loss(batch)
    #     energy = self.network.get_energy()
    #     self.network.compute_energy = False
    #     fig_energy, ax = plt.subplots()
    #     ax.plot(range(len(energy)), [e.cpu() for e in energy], 'b^--')
    #     self.logger.experiment.add_figure(f'{stage}_energy', fig_energy, global_step=self.global_step, close=True)


