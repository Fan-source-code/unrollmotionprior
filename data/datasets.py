import os
import random
import numpy as np

from torch.utils.data import Dataset
from data.utils import _load2d, _load2dxz, _load3d, _load3d_mrca, _load3d_mrca_p2,  _crop_and_pad, _resample, _normalise_intensity, _clean_seg
from data.utils import _magic_slicer, _to_tensor, _to_ndarry
from scipy.io import loadmat


class _BaseDataset(Dataset):
    def __init__(self, data_dir, limit_data: float = 1., batch_size=1):
        super(_BaseDataset, self).__init__()
        self.data_dir = data_dir
        assert os.path.exists(data_dir), f"Data dir does not exist: {data_dir}"
        self.data_path_dict = dict()
        self.subject_list = self._set_subj_list(limit_data, batch_size)

    def _set_subj_list(self, limit_data, batch_size):
        assert limit_data <= 1., f'Limit data ratio ({limit_data}) must be <= 1 '
        subj_list = sorted(os.listdir(self.data_dir))
        if limit_data < 1.:
            num_subj = len(subj_list)
            subj_list = subj_list[:int(num_subj * limit_data)]  # select the subset
            subj_list *= (int(1 / limit_data) + 1)  # repeat to fill
            subj_list = subj_list[:num_subj]
        return subj_list * batch_size  # replicate batch_size times # normalise by batch size

    def _set_path(self, index):
        """ Set the paths of data files to load and the keys in data_dict"""
        raise NotImplementedError

    def __getitem__(self, index):
        """ Load data and pre-process """
        raise NotImplementedError

    def __len__(self):
        return len(self.subject_list)


class BrainMRInterSubj3D(_BaseDataset):
    def __init__(self,
                 data_dir,
                 crop_size=(176, 192, 176),
                 resample_size=(128, 128, 128),
                 limit_data=1.,
                 batch_size=1,
                 modality='t1t1',
                 evaluate=False,
                 atlas_path=None):
        super(BrainMRInterSubj3D, self).__init__(data_dir, limit_data=limit_data, batch_size=batch_size)
        self.crop_size = crop_size
        self.resample_size = resample_size
        self.img_keys = ['tar', 'src']
        self.evaluate = evaluate
        self.modality = modality
        self.atlas_path = atlas_path
        # Note: original data spacings are 1mm in all dimensions
        #self.spacing = [rsz / csz for rsz, csz, in zip(self.resample_size, self.crop_size)] #ori code
        self.spacing = [rsz / csz for rsz, csz, in list(zip(self.resample_size, self.crop_size))]

    def _set_path(self, index):
        # choose the target and source subjects/paths
        if self.atlas_path is None:
            self.tar_subj_id = self.subject_list[index]
            self.tar_subj_path = f'{self.data_dir}/{self.tar_subj_id}'
        else:
            self.tar_subj_path = self.atlas_path

        if self.evaluate:
            self.src_subj_id = self.subject_list[(index+1) % len(self.subject_list)]
        else:
            self.src_subj_id = random.choice(self.subject_list)
        self.src_subj_path = f'{self.data_dir}/{self.src_subj_id}'

        # target and source paths
        self.data_path_dict['tar'] = f'{self.tar_subj_path}/T1_brain.nii.gz'
        if self.modality == 't1t1':
            self.data_path_dict['src'] = f'{self.src_subj_path}/T1_brain.nii.gz'
        elif self.modality == 't1t2':
            self.data_path_dict['src'] = f'{self.src_subj_path}/T2_brain.nii.gz'
        else:
            raise ValueError(f'Modality ({self.modality}) not recognised.')

        if self.evaluate:
            self.img_keys.append('src_ref')
            self.data_path_dict['src_ref'] = f'{self.src_subj_path}/T1_brain.nii.gz'
            self.data_path_dict['tar_seg'] = f'{self.tar_subj_path}/T1_brain_MALPEM_tissues.nii.gz'
            self.data_path_dict['src_seg'] = f'{self.src_subj_path}/T1_brain_MALPEM_tissues.nii.gz'

    def __getitem__(self, index):
        self._set_path(index)
        data_dict = _load3d(self.data_path_dict)
        data_dict = _crop_and_pad(data_dict, self.crop_size)
        data_dict = _normalise_intensity(data_dict, self.img_keys)
        data_dict = _to_tensor(data_dict)
        if self.crop_size != self.resample_size:
            data_dict = _resample(data_dict, size=tuple(self.resample_size))
        return data_dict
    
    
class CardiacCMRA2D(_BaseDataset):
    def __init__(self,
                data_dir,
                limit_data=1,
               batch_size=1,
               slice_range=None,
               slicing=None,
               binindex=None,
               crop_size=(208,208),
               spacing=(1.5,1.5),
               original_spacing=(1.5,1.5)
                ):
        super(CardiacCMRA2D,self).__init__(data_dir, limit_data=limit_data, batch_size=batch_size)
        self.crop_size = crop_size
        self.img_keys = ['tar', 'src', 'src_ref']
        self.slice_range = slice_range
        self.slicing = slicing
        self.binindex=binindex
        self.spacing = spacing
        self.original_spacing = original_spacing
        # self.alldata = self._loadalldata(batch_size)  #assume 2D motion along RL and FH directions
        self.alldata = self._loadalldataxz(batch_size) #assume 2D motion along AP and FH directions
        
    def _loadalldata(self, batch_size):
        length = len(self.subject_list)/batch_size
        #length = min(length, 3)
        alldata_dict = dict()
        for k in range(int(length)):
            data_path_dict = dict()
            subj_id = self.subject_list[k] 
            subj_path = f'{self.data_dir}/{subj_id}'
            # data_path_dict['tar'] = f'{subj_path}/vs0.0_ifftReconbinnum0.999994.mat'
            # data_path_dict['src0'] = f'{subj_path}/vs0.0_ifftReconbinnum0.999991.mat'
            # data_path_dict['src1'] = f'{subj_path}/vs0.0_ifftReconbinnum0.999992.mat'
            # data_path_dict['src2'] = f'{subj_path}/vs0.0_ifftReconbinnum0.999993.mat'
            data_path_dict['tar'] = f'{subj_path}/5min/sg_iSENSEReconbinnum4iter3.mat'
            data_path_dict['src0'] = f'{subj_path}/5min/sg_iSENSEReconbinnum1iter3.mat'
            data_path_dict['src1'] = f'{subj_path}/5min/sg_iSENSEReconbinnum2iter3.mat'
            data_path_dict['src2'] = f'{subj_path}/5min/sg_iSENSEReconbinnum3iter3.mat'
            data_path_dict['motion0'] = f'{subj_path}/motion2/dispbin1.mat'
            data_path_dict['motion1'] = f'{subj_path}/motion2/dispbin2.mat'
            data_path_dict['motion2'] = f'{subj_path}/motion2/dispbin3.mat'
            data_path_dict['input0'] = f'{subj_path}/5min/motionv2/dispbin1.mat'
            data_path_dict['input1'] = f'{subj_path}/5min/motionv2/dispbin2.mat'
            data_path_dict['input2'] = f'{subj_path}/5min/motionv2/dispbin3.mat'
            # coilmap_path = f'{subj_path}/rawdata/coilmaphighRes.mat'
            # # coilmap = loadmat(coilmap_path)['coilmap'].transpose(2, 1, 0, 3)
            # coilmap = loadmat(coilmap_path)['coilmap'].transpose(2, 0, 1, 3) #nz*nx*ny*nc
            alldata_dict[subj_id] = _load2d(data_path_dict, 0)
        return alldata_dict
    
    
    def _loadalldataxz(self, batch_size):
        length = len(self.subject_list)/batch_size
        #length = min(length, 3)
        alldata_dict = dict()
        for k in range(int(length)):
            data_path_dict = dict()
            subj_id = self.subject_list[k] 
            subj_path = f'{self.data_dir}/{subj_id}'
            # data_path_dict['tar'] = f'{subj_path}/vs0.0_ifftReconbinnum0.999994.mat'
            # data_path_dict['src0'] = f'{subj_path}/vs0.0_ifftReconbinnum0.999991.mat'
            # data_path_dict['src1'] = f'{subj_path}/vs0.0_ifftReconbinnum0.999992.mat'
            # data_path_dict['src2'] = f'{subj_path}/vs0.0_ifftReconbinnum0.999993.mat'
            data_path_dict['tar'] = f'{subj_path}/5min/sg_iSENSEReconbinnum4iter3.mat'
            data_path_dict['src0'] = f'{subj_path}/5min/sg_iSENSEReconbinnum1iter3.mat'
            data_path_dict['src1'] = f'{subj_path}/5min/sg_iSENSEReconbinnum2iter3.mat'
            data_path_dict['src2'] = f'{subj_path}/5min/sg_iSENSEReconbinnum3iter3.mat'
            data_path_dict['motion0'] = f'{subj_path}/motion2xz/xzdispbin1.mat'
            data_path_dict['motion1'] = f'{subj_path}/motion2xz/xzdispbin2.mat'
            data_path_dict['motion2'] = f'{subj_path}/motion2xz/xzdispbin3.mat'
            data_path_dict['input0'] = f'{subj_path}/5min/motionv2xz/xzdispbin1.mat'
            data_path_dict['input1'] = f'{subj_path}/5min/motionv2xz/xzdispbin2.mat'
            data_path_dict['input2'] = f'{subj_path}/5min/motionv2xz/xzdispbin3.mat'
            # coilmap_path = f'{subj_path}/rawdata/coilmaphighRes.mat'
            # # coilmap = loadmat(coilmap_path)['coilmap'].transpose(2, 1, 0, 3)
            # coilmap = loadmat(coilmap_path)['coilmap'].transpose(2, 0, 1, 3) #nz*nx*ny*nc
            alldata_dict[subj_id] = _load2dxz(data_path_dict, 0)
        return alldata_dict
    
    
    def _getDataList(self, start, end, binindex):
        if binindex == 'random':          
            z = random.randint(start, end)
            # print('bin:', z)
        else:
            z = binindex
        if ((z < start) or (z > end)):
            raise ValueError(f'Data bin {z} not exist.')
        else:
            data_list = ['tar', f'src{z}', f'motion{z}', f'input{z}']
        return data_list
    
    
    def _set_path(self, index):
        self.subj_id = self.subject_list[index] 
        self.subj_path = f'{self.data_dir}/{self.subj_id}'
        # print(self.subj_path)
        #self.data_path_dict['tar'] = f'{self.subj_path}/vs0.0_ifftReconbinnum0.999994.mat'
        #self.data_path_dict['src'] = f'{self.subj_path}/vs0.0_ifftReconbinnum0.999991.mat'
        #self.data_path_dict['src_ref'] = self.data_path_dict['src']
        self.data_list =self. _getDataList(0, 2, self.binindex)
        #self.coilmap_path = f'{self.subj_path}/rawdata/coilmaphighRes.mat'
        
        
    def _getdata(self, data, data_list):
        data_dict = dict()
        for name in data_list:
            if name == 'tar':
                data_dict[name] = data[name]
            elif 'motion' in name:
                data_dict['disp_gt'] = data[name].astype(np.float32)
            elif 'input' in name:
                data_dict['disp_input'] = data[name].astype(np.float32)
            else:
                data_dict['src'] = data[name]
                data_dict['src_ref'] = data[name]
                
        return data_dict
            
        
    def __getitem__(self, index):
        self._set_path(index)
        # print('subject:', index)
        #self.coilmap = loadmat(self.coilmap_path)['coilmap'].transpose(2, 1, 0, 3)
        #data_dict = _load2d(self.data_path_dict, self.coilmap)
        data_dict = self. _getdata(self.alldata[self.subject_list[index] ], self.data_list)
        data_dict = _magic_slicer(data_dict, slice_range=self.slice_range, slicing=self.slicing)
        data_dict = _normalise_intensity(data_dict, self.img_keys)
        data_dict = _to_tensor(data_dict)
        return data_dict


class CardiacCMRA3D(_BaseDataset):
    def __init__(self,
                 data_dir,
                 limit_data=1,
                 batch_size=1,
                 slice_range=None,
                 slicing=None,
                 binindex=None,
                 durationindex=None,
                 Testing=False,
                 crop_size=(208, 208),
                 spacing=(1.5, 1.5),
                 original_spacing=(1.5, 1.5)
                 ):
        super(CardiacCMRA3D, self).__init__(data_dir, limit_data=limit_data, batch_size=batch_size)
        # self.crop_size = crop_size
        # self.img_keys = ['tar', 'src', 'src_ref']
        self.img_keys = ['tar', 'src', 'targt', 'srcgt']
        # self.slice_range = slice_range
        self.slicing = slicing
        self.binindex = binindex
        self.durationindex = durationindex
        self.Testing = Testing
        # self.spacing = spacing
        # self.original_spacing = original_spacing
        self.batch_size = batch_size
        self.alldata = self._loadalldata3D(batch_size)  #assume 2D motion along RL and FH directions
        # self.alldata = self._loadalldataxz(batch_size)  # assume 2D motion along AP and FH directions

    def _loadalldata3D(self, batch_size):
        length = len(self.subject_list) / batch_size
        # length = min(length, 3)
        alldata_dict = dict()
        for k in range(int(length)):
            data_path_dict = dict()
            subj_id = self.subject_list[k]
            subj_path = f'{self.data_dir}/{subj_id}'

            data_path_dict['motion0'] = f'{subj_path}/rawdata/R1v2/softgating/label/3Dmotionv4/dispbin1.mat'
            data_path_dict['motion1'] = f'{subj_path}/rawdata/R1v2/softgating/label/3Dmotionv4/dispbin2.mat'
            data_path_dict['motion2'] = f'{subj_path}/rawdata/R1v2/softgating/label/3Dmotionv4/dispbin3.mat'

            alldata_dict[subj_id] = _load3d_mrca(data_path_dict, 0)
            # alldata_dict[subj_id] = []
        return alldata_dict

    def _loadalldata3Dimg(self, idxduration, subjindex, binidx):
        alldata_dict = dict()
        data_path_dict = dict()
        subj_id = self.subject_list[subjindex]
        subj_path = f'{self.data_dir}/{subj_id}'

        # if self.Testing:
        #     idxduration = self.durationindex

        data_path_dict[
            f'src{int(idxduration)}{binidx}'] = f'{subj_path}/rawdata/R6v2/softgating/{int(idxduration)}/sgiSENSEReconv2/Echo2_sg_iSENSEReconbinnum{binidx+1}.mat'
        data_path_dict[
            f'tar{int(idxduration)}'] = f'{subj_path}/rawdata/R6v2/softgating/{int(idxduration)}/sgiSENSEReconv2/Echo2_sg_iSENSEReconbinnum4.mat'

        # data_path_dict[
        #     f'src{int(idxduration)}{binidx}'] = f'{subj_path}/rawdata/R1v2/softgating/sgiSENSEReconv2/Echo2_sg_iSENSEReconbinnum{binidx + 1}.mat'
        # data_path_dict[
        #     f'tar{int(idxduration)}'] = f'{subj_path}/rawdata/R1v2/softgating/sgiSENSEReconv2/Echo2_sg_iSENSEReconbinnum4.mat'

        # data_path_dict[
        #     f'srcgt{binidx}'] = f'{subj_path}/rawdata/R1v2/softgating/sgiSENSEReconv2/Echo2_sg_iSENSEReconbinnum{binidx + 1}.mat'
        # data_path_dict[
        #     f'targt'] = f'{subj_path}/rawdata/R1v2/softgating/sgiSENSEReconv2/Echo2_sg_iSENSEReconbinnum4.mat'

        alldata_dict[subj_id] = _load3d_mrca(data_path_dict, 0)
        return alldata_dict

    def _getDataList(self, binstart, binend, binindex, durationstart, durationend, durationindex):
        if binindex == 'random':
            z = random.randint(binstart, binend)
            # print('bin:', z)
        else:
            z = binindex

        if durationindex == 'random':
            s = random.randint(durationstart, durationend)
        else:
            s = int(durationindex + 1)

        # data_list = [f'tar{s}', f'src{s}{z}', f'motion{z}', f'input{s}{z}']
        data_list = [f'tar{s}', f'src{s}{z}', f'motion{z}']
        # data_list = [f'tar{s}', f'src{s}{z}']
        # data_list = [f'tar{s}', f'src{s}{z}', 'targt', f'srcgt{z}']
        return data_list, z, s

    def _set_path(self, index):
        self.subj_id = self.subject_list[index]
        self.subj_path = f'{self.data_dir}/{self.subj_id}'
        # print(self.subj_path)
        # self.data_path_dict['tar'] = f'{self.subj_path}/vs0.0_ifftReconbinnum0.999994.mat'
        # self.data_path_dict['src'] = f'{self.subj_path}/vs0.0_ifftReconbinnum0.999991.mat'
        # self.data_path_dict['src_ref'] = self.data_path_dict['src']
        self.data_list, self.binidx, self.durationidx = self._getDataList(0, 2, self.binindex, 1, 11, self.durationindex)
        # self.coilmap_path = f'{self.subj_path}/rawdata/coilmaphighRes.mat'
        # self.alldata[self.subj_id] = self._loadalldata3Dp2(self.alldata[self.subj_id], binidx, durationidx, self.subj_path)

    # def _getdata(self, data, data_list):
    #     data_dict = dict()
    #     for name in data_list:
    #         if 'tar' in name:
    #             data_dict['tar'] = data[name]
    #         elif 'motion' in name:
    #             data_dict['disp_gt'] = data[name].astype(np.float32)
    #         elif 'input' in name:
    #             data_dict['disp_input'] = data[name].astype(np.float32)
    #         else:
    #             data_dict['src'] = data[name]
    #             # data_dict['src_ref'] = data[name]
    #
    #     return data_dict

    def _getdata(self, motion_dict, img_dict, data_list):
        data_dict = dict()
        for name in data_list:
            if 'targt' in name:
                data_dict['targt'] = img_dict[name]
            elif 'tar' in name:
                data_dict['tar'] = img_dict[name]
            elif 'motion' in name:
                data_dict['disp_gt'] = motion_dict[name].astype(np.float32)
            # elif 'input' in name:
            #     data_dict['disp_input'] = data[name].astype(np.float32)
            elif 'srcgt' in name:
                data_dict['srcgt'] = img_dict[name]
            else:
                data_dict['src'] = img_dict[name]
                # data_dict['src_ref'] = data[name]

        return data_dict

    def __getitem__(self, index):
        self._set_path(index)
        # print('subject:', index)
        # self.coilmap = loadmat(self.coilmap_path)['coilmap'].transpose(2, 1, 0, 3)
        # data_dict = _load2d(self.data_path_dict, self.coilmap)
        img_dict = self._loadalldata3Dimg(self.durationidx, index, self.binidx)
        data_dict = self._getdata(self.alldata[self.subject_list[index]], img_dict[self.subject_list[index]], self.data_list)
        data_dict = _normalise_intensity(data_dict, self.img_keys)
        # data_dict = _magic_slicer(data_dict, slicing=self.slicing)
        data_dict = _to_tensor(data_dict)
        return data_dict
        
        
class CardiacMR2D(_BaseDataset):
    def __init__(self,
                 data_dir,
                 limit_data=1.,
                 batch_size=1,
                 slice_range=None,
                 slicing=None,
                 crop_size=(160, 160),
                 spacing=(1.8, 1.8),
                 original_spacing=(1.8, 1.8)
                 ):
        super(CardiacMR2D, self).__init__(data_dir, limit_data=limit_data, batch_size=batch_size)
        self.crop_size = crop_size
        self.img_keys = ['tar', 'src', 'src_ref']
        self.slice_range = slice_range
        self.slicing = slicing
        self.spacing = spacing
        self.original_spacing = original_spacing

    def _set_path(self, index):
        self.subj_id = self.subject_list[index]
        self.subj_path = f'{self.data_dir}/{self.subj_id}'
        self.data_path_dict['tar'] = f'{self.subj_path}/sa_ED.nii.gz'
        self.data_path_dict['src'] = f'{self.subj_path}/sa_ES.nii.gz'
        self.data_path_dict['src_ref'] = self.data_path_dict['src']
        self.data_path_dict['tar_seg'] = f'{self.subj_path}/label_sa_ED.nii.gz'
        self.data_path_dict['src_seg'] = f'{self.subj_path}/label_sa_ES.nii.gz'

    def __getitem__(self, index):
        self._set_path(index)
        data_dict = _load2d(self.data_path_dict)
        data_dict = _magic_slicer(data_dict, slice_range=self.slice_range, slicing=self.slicing)
        if self.original_spacing != self.spacing:
            # resample if spacing changes
            data_dict = _to_tensor(data_dict)
            scale_factor = tuple([os / s for (os, s) in zip(self.original_spacing, self.spacing)])
            data_dict = _resample(data_dict, scale_factor=scale_factor)
            data_dict = _to_ndarry(data_dict)
        data_dict = _crop_and_pad(data_dict, self.crop_size)
        data_dict = _normalise_intensity(data_dict, self.img_keys)
        data_dict = _to_tensor(data_dict)
        return data_dict


class CardiacMR2D_MM(CardiacMR2D):
    def __init__(self, *args, **kwargs):
        super(CardiacMR2D_MM, self).__init__(*args, **kwargs)

    def _set_path(self, index):
        self.subj_id = self.subject_list[index]
        self.subj_path = f'{self.data_dir}/{self.subj_id}'
        self.data_path_dict['tar'] = f'{self.subj_path}/ED_img.nii.gz'
        self.data_path_dict['src'] = f'{self.subj_path}/ES_img.nii.gz'
        self.data_path_dict['src_ref'] = self.data_path_dict['src']
        self.data_path_dict['tar_seg'] = f'{self.subj_path}/ED_seg.nii.gz'
        self.data_path_dict['src_seg'] = f'{self.subj_path}/ES_seg.nii.gz'

    def __getitem__(self, index):
        data_dict = super(CardiacMR2D_MM, self).__getitem__(index)
        if self.slicing != 'random':
            data_dict = _to_tensor(_clean_seg(_to_ndarry(data_dict)))
        return data_dict
