from torch.utils.data import Dataset, ConcatDataset, random_split
import torch
import os
import numpy as np
import random
from utils.config import get_config
import nibabel as nib
from torchvision import transforms
import torchio as tio



config = get_config()

data_root = config.train.dataset
train_val_ratio = config.train.train_val_ratio
test_path = config.test.dataset

torch.manual_seed(202109)
torch.cuda.manual_seed_all(202109)
np.random.seed(202109)


class Patient(Dataset):
    def __init__(self, root):
        self.root = root
        self.data = []
        self.cp = self.read_cp()
        self.read_data()

    def __getitem__(self, item):
        data = self.data[item]
        pos = data.get('position')           
        T10 = np.array([data.get('T10'),])    
        raw_data = data.get('dce_data')
        cp = self.cp
        #Ktrans_all = np.load('/media/sail/Elements/BME_Grad_Project/Joanne/BBB_DL/paper/new/Ktrans_High_recreate_Th_npy/' + self.root[-5:] + '.npy', allow_pickle=True)
        #Ktrans_p = Ktrans_all[pos[0], pos[1], pos[2]]
        #param = np.array([Ktrans_p, Ktrans_p, Ktrans_p])
        param = data.get('param')

##################### add 0429 #########################
        T10_pcr99 = np.load('/media/sail/Elements/BME_Grad_Project/Joanne/BBB_DL/demo/data/DL_input/T10_prc99/' + self.root[-5:] + '.npy', allow_pickle=True)

        T10 = T10 / T10_pcr99
        dce_spatial = raw_data / T10_pcr99
        cp = cp / T10_pcr99

        extra_data = np.ones_like(cp)
        extra_data = extra_data*T10

        cp = cp[np.newaxis, ...]

        data_ref = np.concatenate((cp, np.reshape(extra_data,(1,cp.shape[1]))), axis=0)
        dce_spatial = dce_spatial[np.newaxis, ...]  # 1, 7, 7, 84
        return dce_spatial.astype(np.float), data_ref.astype(np.float), param.astype(np.float32), pos           

    def __len__(self):
        return len(self.data)

    def read_cp(self):
        info_file = os.path.join(self.root, 'CA.npy')
        cp = np.load(info_file, allow_pickle=True)
        return cp

    def read_data(self):
        eTofts_files = os.listdir(self.root)
        eTofts_files.remove('CA.npy')

        eTofts_files.sort(key=lambda x: int((os.path.splitext(x)[0]).split('_')[1]))
        for file in eTofts_files:
            data = np.load(os.path.join(self.root, file), allow_pickle=True)
            self.data.extend(data)

def patients_dataset(fold_df, f):
    testing_subject = fold_df.loc[fold_df['fold'] == f, ['subject']]['subject'].tolist()
    training_subject = fold_df['subject'].tolist()
    print(training_subject)


    patients = []
    test_set = []
    print('Testing path: ', test_path)
    print('Training path: ', data_root)
    for pats in training_subject:
        if pats in testing_subject:
            print('Loading testing subjects: ', pats)
            test_set.append(Patient(os.path.join(test_path, pats)))          
        else:
            print('Loading training + validation subjects: ', pats)
            patients.append(Patient(os.path.join(data_root, pats)))

    test = ConcatDataset(test_set)
    sets = ConcatDataset(patients)                                                         
    train_len = int(sets.__len__()*train_val_ratio)                                                 
    train, valid = random_split(sets, [train_len, sets.__len__()-train_len])               
    return train, valid, test




def dataAug(data):
    dce_spatial = data
    dce_Aug = np.zeros(dce_spatial.shape)
    # affine_transform = tio.RandomAffine()
    training_transform = tio.Compose({
                                        tio.OneOf({
                                                    tio.RandomBlur(),                 
                                                    tio.RandomNoise()
                                                    }, p=0.5),
                                        # tio.RandomMotion(p=0.5),
                                        # tio.RandomAffine(p=0.5),

        })

    for b in range(dce_spatial.shape[0]):
        dce_spatial_4D = dce_spatial[b]   # (1, x, y, t)
        dce_spatial_4D_Aug = training_transform(dce_spatial_4D)
        dce_Aug[b, :, :, :, :] = dce_spatial_4D_Aug
    # dce_spatial_4D = dce_spatial[np.newaxis, ...]
    # dce_spatial_4D_Aug = training_transform(dce_spatial_4D)
    return torch.tensor(dce_Aug)
