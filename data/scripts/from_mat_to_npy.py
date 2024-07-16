import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
import os
import argparse
#from model.eTofts import fit_eTofts, full_eTofts, s
import random
import nibabel as nib
import matplotlib.pyplot as plt
from scipy import ndimage

parser = argparse.ArgumentParser()
#parser.add_argument('--input', required=True, type=str)
parser.add_argument('--save_path', required=True, type=str)
parser.add_argument('--output', required=True, choices=['OP_NM_CAlow', 'OP_NM_CAhigh', 'WB_CAhigh', 'WB_CAlow'], type=str)
parser.add_argument('--GT', default='High', choices=['High', 'Low'], type=str)
parser.add_argument('--ratio', nargs='+', default=[1,1], type=int)
parser.add_argument('--patch_steps', default=1, type=int)

args = parser.parse_args()
#pats = args.input
save_path = args.save_path   # ST_pad_InRaw_GtRaw_CAlow
output = args.output
GT = args.GT
ratio = args.ratio
Input = save_path.split('In')[1].split('_')[0]
Output = save_path.split('Gt')[1][:-1].split('_')[0]    #ST_pad_InSmoothed_GtRaw
CA_type = output.split('CA')[1]
patch_steps = args.patch_steps
print('Input type: ', Input)
print('Output type: ', Output)
#OP_ratio = 6
#NM_ratio = 4
OP_ratio = ratio[0]
NM_ratio = ratio[1]
ratio = NM_ratio/OP_ratio
print(ratio)

print('output: {}, GT: {}'.format(output, GT) )
    
def WB_npy(cp, dce_data, T1, brain, ktrans, OP_mask):
    WB_datas = []
    for idx in range(OP_mask.shape[2]):
        for i in range(OP_mask.shape[0]):
            for j in range(OP_mask.shape[1]):
                if brain[i, j, idx] == 1:    

                    #raw_dce = dce_data[i-3:i+4, j-3:j+4, idx, :]
                    #dce = raw_dce

                    data = {'T10': T1, 'dce_data': dce_data, 'position': (i, j, idx), 'ktrans': ktrans}
#                    data = {'T10': T1[i, j, idx], 'dce_data': dce, 'position': (i, j, idx), 'ktrans': ktrans[i, j, idx]}
                    x = pool.apply_async(update_data, args=(data, cp, queue))
                    data = queue.get()
                    WB_datas.append(data)

    WB_count = len(WB_datas)
    print(pat, 'WB', WB_count)
    np.save(os.path.join(save_dir, 'WB_{}'.format(WB_count)), WB_datas)
    #return WB_datas, WB_count

def OP_NM_npy(cp, dce_data, T1, brain, ktrans, OP_mask, patch_steps):
    OP_datas = []
    NM_datas = []
    NM_coor = []
    for idx in range(OP_mask.shape[2]):
        for i in range(0, OP_mask.shape[0], patch_steps):
            for j in range(0, OP_mask.shape[1], patch_steps):
                if OP_mask[i, j, idx] == 1 and brain[i, j, idx] == 1:                   
                    raw_dce = dce_data[i-3:i+4, j-3:j+4, idx, :]
                    #dce = raw_dce
                    
#                    data = {'T10': T1, 'dce_data': dce_data, 'position': (i, j, idx), 'ktrans': ktrans}
                    data = {'T10': T1[i, j, idx], 'dce_data': raw_dce, 'position': (i, j, idx), 'ktrans': ktrans[i, j, idx]}
                    x = pool.apply_async(update_data, args=(data, cp, queue))

                    data = queue.get()
                    OP_datas.append(data)

                elif OP_mask[i, j, idx] == 0 and brain[i, j, idx] == 1:
                    NM_coor.append([i,j,idx]) 

    OP_count = len(OP_datas)  
    NM_count = int(OP_count*ratio)

    if NM_count > len(NM_coor):
        NM_count = len(NM_coor)
        
    normal_datas_random = random.sample(NM_coor, NM_count)
    for X,Y,Z in normal_datas_random:
        raw_dce = dce_data[X-3:X+4, Y-3:Y+4, Z, :]
        #dce = raw_dce
#        data = {'T10': T1, 'dce_data': dce_data, 'position': (X, Y, Z), 'ktrans': ktrans}
        data = {'T10': T1[X, Y, Z], 'dce_data': raw_dce, 'position': (X, Y, Z), 'ktrans': ktrans[X, Y, Z]}
        x = pool.apply_async(update_data, args=(data, cp, queue))

        data = queue.get()
        NM_datas.append(data)
           
    print(pat, 'OP:', OP_count, ', NM:', len(NM_coor), ' -> ', len(NM_datas))
    np.save(os.path.join(save_dir, 'OP_{}'.format(OP_count)), OP_datas)
    np.save(os.path.join(save_dir, 'NM_{}'.format(NM_count)), NM_datas)
    #return OP_datas, NM_datas, OP_count, NM_count




def update_data(data,cp,q):
    T10, signal, ktrans = data['T10'], data['dce_data'], data['ktrans']		#original
    par = np.array([ktrans, ktrans, ktrans])	#add
    data.update({'param': par})
    data.update({'dce_data': signal})
    q.put(data)

pool = multiprocessing.Pool()
queue = multiprocessing.Manager().Queue(1024)

data_root = '/media/sail/Elements/BME_Grad_Project/Joanne/BBB_DL/demo/data/'
project_root = '/media/sail/Elements/BME_Grad_Project/Joanne/BBB_DL/demo/data/DL_input/'

for pat_name in os.listdir(os.path.join(data_root, 'raw/BrainMask')):
    pat = pat_name[44:49]

    if output.split('_CA')[0] == 'OP_NM':
        save_dir = os.path.join(project_root, save_path, '{}_{}_{}_k{}'.format(output, OP_ratio, NM_ratio, GT), '{}'.format(pat))
    else:
        save_dir = os.path.join(project_root, save_path, '{}_k{}'.format(output, GT), '{}'.format(pat))

    os.makedirs(save_dir, exist_ok=True)
    if CA_type == 'low':
        cp = scio.loadmat(os.path.join(data_root, 'raw_process/Muscle_CA_Low', pat+'_CA_Low.mat'))['CA_Low'][:, 0]
    else:
        cp = scio.loadmat(os.path.join(data_root, 'raw_process/Muscle_CA_High', pat+'_CA_High.mat'))['CA_High'][:, 0]

    np.save(os.path.join(save_dir, 'CA.npy'), cp)
    brain = nib.load(os.path.join(data_root, 'raw/BrainMask', pat_name)).get_fdata()
    OP_mask = nib.load(os.path.join(data_root, 'raw/openingMask_High', pat + '_HighDose_ktrans_mask.nii.gz')).get_fdata()
    OP_mask = OP_mask * brain

    if Input == 'RawLowDCE': 
        print('Input raw low dose DCE')
        dce_data = nib.load(os.path.join(data_root, 'raw/DCE_Low_4d', pat+'_LowDose_4d.nii.gz')).get_fdata()      
        T1 = nib.load(os.path.join(data_root, 'raw_process/T10_LowDose', pat+'_T10_LowDose.nii.gz')).get_fdata()   
#		T1_pcr99 = np.percentile(T1[np.logical_and(brain == 1 , T1 > 0)], 99)
#		print(T1_pcr99)
#		T1 = T1 / T1_pcr99
    elif Input == 'RawHighDCE':
        print('Input raw high dose DCE')
        dce_data = nib.load(os.path.join(data_root, 'raw/DCE_High_4d', pat+'_HighDose_4d.nii.gz')).get_fdata()      
        T1 = nib.load(os.path.join(data_root, 'raw_process/T10_HighDose', pat+'_T10_HighDose.nii.gz')).get_fdata()   
    else:
        print('Wrong input')
        exit()


    if Output == 'Raw':
        print('Output raw')
        ktrans = nib.load(os.path.join(data_root, 'raw_process/ktrans_th', pat+'_HighDose_ktrans_th.nii.gz')).get_fdata()
#		ktrans = nib.load(os.path.join(data_root, 'Raw/Ktrans_High', pat+'_HighDose_ktrans.nii.gz')).get_fdata()
    else:
        print('Wrong output')
        exit()


    T10_WB = T1 * brain
    T10_pcr99 = np.percentile(T10_WB[brain == 1], 99)
    print(T10_pcr99)
    os.makedirs(os.path.join(project_root, 'T10_prc99'), exist_ok=True)
    np.save(os.path.join(project_root, 'T10_prc99', pat), T10_pcr99)

 #   plt.subplot(2,3,1)
 #   plt.plot(cp)

 #   plt.subplot(2,3,2)
 #   plt.imshow(dce_data[:,:,10,83])

 #   plt.subplot(2,3,3)
	# plt.imshow(T1[:,:,10])

 #   plt.subplot(2,3,4)
 #   plt.imshow(brain[:,:,10])

 #   plt.subplot(2,3,5)
 #   plt.imshow(ktrans[:,:,10])

 #   plt.subplot(2,3,6)
 #   plt.imshow((OP_mask*ktrans)[:,:,10])
	# plt.show()


    if output.split('_CA')[0] == 'WB':
       WB_npy(cp, dce_data, T1, brain, ktrans, OP_mask)
    elif output.split('_CA')[0] == 'OP_NM':
       OP_NM_npy(cp, dce_data, T1, brain, ktrans, OP_mask, patch_steps)
    else:
       exit()

print('save_dir: ',save_dir)

#python3 from_mat_to_npy.py --save_path=ST_pad_InRawLowDCE_GtRaw_KtransTh/ --output='OP_NM_CAlow' --ratio 1 1
#python3 from_mat_to_npy.py --save_path=ST_pad_InRawLowDCE_GtRaw_KtransTh_patchsteps3/ --output='OP_NM_CAlow' --ratio 1 4 --patch_steps 3
#python3 from_mat_to_npy.py --save_path=ST_pad_InRawLowDCE_GtRaw_KtransTh/ --output='WB_CAlow' 

#python3 from_mat_to_npy.py --save_path=ST_pad_InRawHighDCE_GtRaw_KtransTh/ --output='OP_NM_CAhigh' --ratio 1 1
#python3 from_mat_to_npy.py --save_path=ST_pad_InRawHighDCE_GtRaw_KtransTh_patchsteps3/ --output='OP_NM_CAhigh' --ratio 1 4 --patch_steps 3
#python3 from_mat_to_npy.py --save_path=ST_pad_InRawHighDCE_GtRaw_KtransTh/ --output='WB_CAhigh' 



#python3 from_mat_to_npy_Joanne_ST.py --save_path=ST_pad_InSmoothed_GtRaw/ --output='OP_NM' --ratio 1 1
#python3 from_mat_to_npy_Joanne_ST.py --save_path=ST_pad_InRaw_GtSmoothed/ --output='OP_NM' --ratio 1 1
#python3 from_mat_to_npy_Joanne_ST.py --save_path=ST_pad_InSmoothed_GtSmoothed/ --output='OP_NM' --ratio 1 1
#python3 from_mat_to_npy_Joanne_ST.py --save_path=ST_pad_InRaw_GtRaw/ --output='OP_NM' --ratio 1 1

#python3 from_mat_to_npy_Joanne_ST.py --save_path=ST_pad_InSmoothed_GtRaw/ --output='WB'
#python3 from_mat_to_npy_Joanne_ST.py --save_path=ST_pad_InRaw_GtSmoothed/ --output='WB'
#python3 from_mat_to_npy_Joanne_ST.py --save_path=ST_pad_InSmoothed_GtSmoothed/ --output='WB'
#python3 from_mat_to_npy_Joanne_ST.py --save_path=ST_pad_InRaw_GtRaw/ --output='WB'


#python3 from_mat_to_npy_Joanne_ST.py --save_path=ST_pad_InRaw_GtMedianTh/ --output='OP_NM' --ratio 1 4
#python3 from_mat_to_npy_Joanne_ST.py --save_path=ST_pad_InRaw_GtRaw_nonoverlap/ --output='OP_NM' --ratio 1 1

#python3 from_mat_to_npy_Joanne_ST_new.py --save_path=ST_pad_InRaw_GtRaw_Low/ --output='OP_NM_CAhigh' --ratio 1 1
#python3 from_mat_to_npy_Joanne_ST_new.py --save_path=ST_pad_InRaw_GtRaw_Low/ --output='OP_NM_CAlow' --ratio 1 1
#python3 from_mat_to_npy_Joanne_ST_new.py --save_path=ST_pad_InRaw_GtRaw_Low/ --output='WB_CAhigh' 
#python3 from_mat_to_npy_Joanne_ST_new.py --save_path=ST_pad_InRaw_GtRaw_Low/ --output='WB_CAlow' 

#python3 from_mat_to_npy.py --save_path=ST_pad_InRaw_GtRaw_KtransTh/ --output='OP_NM_CAlow' --ratio 1 1
#python3 from_mat_to_npy_Joanne_ST_new.py --save_path=ST_pad_InRaw_GtRaw_KtransTh_T10nor_patchsteps3/ --output='OP_NM_CAlow' --ratio 1 4 --patch_steps 3
#python3 from_mat_to_npy_Joanne_ST_new.py --save_path=ST_pad_InRaw_GtRaw_KtransTh/ --output='WB_CAlow' 

#python3 from_mat_to_npy_Joanne_ST_new.py --save_path=ST_pad_InRaw_GtRaw_KtransTh_patchsteps3_pos/ --output='OP_NM_CAlow' --ratio 1 4 --patch_steps 3
#python3 from_mat_to_npy_Joanne_ST_new.py --save_path=ST_pad_InRaw_GtRaw_KtransTh_pos/ --output='WB_CAlow' 

