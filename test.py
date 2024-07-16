import argparse
import datetime
import os
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from model.network_model_ST import CNN_ST
from utils.evaluate import concordance_correlation_coefficient as ccc
from utils.evaluate import KL
from utils.dataset_ST import patients_dataset, Patient

import toml
from easydict import EasyDict
import nibabel as nib
import csv
import pandas as pd
import scipy.io as scio
from scipy import ndimage
import scipy.stats
import skimage
from skimage import metrics
from skimage.metrics import normalized_root_mse as nrmse
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, type=str)
args = parser.parse_args()
model_path = os.path.join('exp/',args.model_path)
config_path = os.path.join(model_path, 'config.toml')


with open(config_path, 'r', encoding='utf-8') as f:
	config = toml.load(f)
	default_config = EasyDict(config)

fold_df = default_config.train.fold_excel
fold_df = pd.read_excel(fold_df, dtype={'subject':str, 'fold':int})
test_config = default_config.test
test_config.update(args.__dict__)

gpu = torch.device('cuda:{}'.format(default_config.train.gpu))
torch.cuda.set_device(gpu)
torch.multiprocessing.set_sharing_strategy('file_system')
batch = test_config.batch

str_time = time.strftime('%m-%d_%H-%M-%S', time.localtime(time.time()))

fast_m = CNN_ST().cuda()
test_path = test_config.dataset
print('test_path',test_path)

loss_type = default_config.train.loss_type
if loss_type == 'MSE':
	loss_func = nn.MSELoss()
elif loss_type == 'MAE':
	loss_func = nn.L1Loss()
elif loss_type == 'SmoothL1':
	loss_func = nn.SmoothL1Loss()

def test(dataset, fast_m, name='test'):

	res_x = []
	res_y = []

	k_trans_x = []
	k_trans_y = []

	with torch.no_grad():
		fast_m.eval()
		test_load = DataLoader(dataset=dataset, batch_size=batch, shuffle=False, num_workers=4)

		img_name = name + '_HighDose_ktrans.nii.gz'
		img = nib.load(os.path.join('/media/sail/Elements/BME_Grad_Project/Joanne/BBB_DL/demo/data/raw/Ktrans_High', img_name))
		test_img = np.zeros(img.shape)
		GT_img = np.zeros(img.shape)

		for count, data in enumerate(test_load):
			dce_spetial, data_ref, paramters, pos = data   
			dce_spetial = dce_spetial.cuda()
			data_ref = data_ref.cuda()

			pre = fast_m(dce_spetial, data_ref)

			test_img[pos[0].numpy(), pos[1].numpy(), pos[2].numpy()] = pre[:, 0].cpu().numpy()
			GT_img[pos[0].numpy(), pos[1].numpy(), pos[2].numpy()] = paramters[:, 0].cpu().numpy()

	PredictionImg = nib.Nifti1Image(test_img, img.affine, img.header)
	BrainMask_ID = {'ID002':'20201214_212326_IVIM_and_DeepC_FUS_BBB_Scan_ID002_L1_001_1_1_BrainMask',
					'ID003':'20201215_214853_IVIM_and_DeepC_FUS_BBB_Scan_ID003_L2_001_1_1_BrainMask',
					'ID004':'20201215_235940_IVIM_and_DeepC_FUS_BBB_Scan_ID004_L3_001_1_1_BrainMask',
					'ID005':'20210205_151949_IVIM_and_DeepC_FUC_BBB_Scan_ID005_L1_001_1_1_BrainMask',
					'ID007':'20210213_184916_IVIM_and_DeepC_FUS_BBB_Scan_ID007_L1_001_1_1_BrainMask',
					'ID008':'20210213_205053_IVIM_and_DeepC_FUS_BBB_Scan_ID008_L1_001_1_1_BrainMask',
					'ID009':'20210214_181057_IVIM_and_DeepC_FUS_BBB_Scan_ID009_L1_001_1_1_BrainMask',
					'ID012':'20210221_210445_IVIM_and_DeepC_FUS_BBB_Scan_ID012_L2_001_1_1_BrainMask',
					'ID014':'20210311_184341_IVIM_and_DeepC_FUS_BBB_Scan_ID014_001_1_1_BrainMask',
					'ID016':'20210311_220532_IVIM_and_DeepC_FUS_BBB_Scan_ID016_001_1_1_BrainMask',
					'ID019':'20211202_161605_IVIM_and_DeepC_FUS_BBB_Scan_ID019_001_1_1_BrainMask',
					'ID020':'20211202_161623_IVIM_and_DeepC_FUS_BBB_Scan_ID020_001_1_1_BrainMask'
					}

	BrainMask = nib.load(os.path.join('/media/sail/Elements/BME_Grad_Project/Joanne/BBB_DL/demo/data/raw/BrainMask', BrainMask_ID[name] + '.nii.gz')).get_fdata()
	test_WB = test_img * BrainMask
	GT_WB = GT_img * BrainMask


	kernel_size = (3,3,3)
	test_WB_smoothed = ndimage.median_filter(test_WB, size = kernel_size)
	GT_WB_smoothed = ndimage.median_filter(GT_WB, size = kernel_size)
	GT_WB_smoothed[GT_WB_smoothed > 0.05] = 0

	GT_WB_max = np.max(GT_WB_smoothed[BrainMask == 1])
	test_WB_max = np.max(test_WB_smoothed[BrainMask == 1])

	GT_WB_smoothed = GT_WB_smoothed #/ GT_WB_max
	test_WB_smoothed = test_WB_smoothed #/ test_WB_max

	th = 0
	# INCRESE TH TO 0.2 AND DO COMPARISON (for opening area)
	GT_WB_smoothed[GT_WB_smoothed < th] = 0
	# GT_WB_smoothed[GT_WB_smoothed > 1] = 0
	test_WB_smoothed[test_WB_smoothed < th] = 0
	# test_WB_smoothed[test_WB_smoothed > 1] = 0


	GT_WB_smoothed_array = GT_WB_smoothed[BrainMask == 1]
	test_WB_smoothed_array = test_WB_smoothed[BrainMask == 1]


	Prediction_vs_GT_SCC, p_value = scipy.stats.spearmanr(test_WB_smoothed_array, GT_WB_smoothed_array, nan_policy='omit')
	Prediction_vs_GT_PCC, p_value = np.corrcoef(GT_WB_smoothed_array, test_WB_smoothed_array)[1]
	Prediction_vs_GT_ccc = ccc(GT_WB_smoothed_array, test_WB_smoothed_array)
	Prediction_vs_GT_nrmse = nrmse(GT_WB_smoothed_array, test_WB_smoothed_array, normalization='euclidean')
	Prediction_vs_GT_KL = KL(test_WB_smoothed_array, GT_WB_smoothed_array, BrainMask, 128)

	Prediction_vs_GT_PSNR_sum = 0
	Prediction_vs_GT_SSIM_sum = 0
	for i in range(test_img.shape[2]):
		GT_WB_smoothed_slice = GT_WB_smoothed[:,:,i]
		test_WB_smoothed_slice = test_WB_smoothed[:,:,i]
		BrainMask_slice = BrainMask[:, :, i]

		slice_psnr = PSNR(GT_WB_smoothed_slice[BrainMask_slice == 1], test_WB_smoothed_slice[BrainMask_slice == 1])
		slice_ssim = SSIM(GT_WB_smoothed_slice[BrainMask_slice == 1], test_WB_smoothed_slice[BrainMask_slice == 1])

		Prediction_vs_GT_PSNR_sum += slice_psnr
		Prediction_vs_GT_SSIM_sum += slice_ssim
	Prediction_vs_GT_PSNR = Prediction_vs_GT_PSNR_sum / test_img.shape[2]
	Prediction_vs_GT_SSIM = Prediction_vs_GT_SSIM_sum / test_img.shape[2]

	# Prediction_vs_GT_SSIM = SSIM(GT_WB_smoothed_array, test_WB_smoothed_array)
	# Prediction_vs_GT_SSIM = 0
	# for i in range(18):
	# 	t = test_WB_smoothed[i]
	# 	g = GT_WB_smoothed[i]
	# 	b = BrainMask[i]
	# 	Prediction_vs_GT_SSIM = SSIM(t[b==1], g[b==1])#, datarange=test_WB_smoothed_array.max()-test_WB_smoothed_array.min())
	# 	print(Prediction_vs_GT_SSIM)

	loss_dict = {'SCC': Prediction_vs_GT_SCC, 'PCC': Prediction_vs_GT_PCC, 'CCC': Prediction_vs_GT_ccc, 'NRMSE': Prediction_vs_GT_nrmse, 'KL': Prediction_vs_GT_KL, 'PSNR': Prediction_vs_GT_PSNR, 'SSIM': Prediction_vs_GT_SSIM}
	loss_info = '{}, SCC:{}, PCC:{}, CCC:{}, NRMSE:{}, KL:{}, PSNR:{}, SSIM:{}'.format(
				name, Prediction_vs_GT_SCC, Prediction_vs_GT_PCC, Prediction_vs_GT_ccc, Prediction_vs_GT_nrmse, Prediction_vs_GT_KL, Prediction_vs_GT_PSNR, Prediction_vs_GT_SSIM
	)

	return loss_info, PredictionImg, loss_dict

if not os.path.exists(model_path):
	raise FileNotFoundError(f'{model_path} not exists!')

test_time = str(datetime.datetime.now())
# dict1 = {'SCC': 0, 'PCC': 0, 'CCC': 0, 'NRMSE': 0, 'KL': 0, 'PSNR': 0, 'SSIM': 0}
loss_all = {}

for f in range(1, len(set(fold_df['fold']))+1):
	testing_subject = fold_df.loc[fold_df['fold'] == f, ['subject']]['subject'].tolist()
	if not os.path.exists(os.path.join(model_path, 'Fold' + str(f), 'best_patient.tar')):
		continue

	param_path = os.path.join(model_path, 'Fold' + str(f), 'best_patient.tar')
	best_model = torch.load(param_path, map_location=gpu)
	fast_m.load_state_dict(best_model)

	# test_result = []
	
	save_path = os.path.join(model_path,'test')
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	for pat_dir in testing_subject:
		test_pat_data = Patient(os.path.join(test_path, pat_dir))
		pat_result, PredictionImg, loss_dict = test(test_pat_data, fast_m, pat_dir)
		loss_all[pat_dir] = loss_dict
		save_name = 'Fold' + str(f) + '_' + pat_dir + '_test.nii.gz'
		nib.save(PredictionImg, os.path.join(save_path, save_name))

		# test_result.append(pat_result)
		# for metrics, val in dict1.items():
		# 	val += loss_dict[metrics]
		# 	dict1[metrics] = val

print('------------Test Result------------')


# df_loss_all = pd.DataFrame(loss_all).T
# for metrics, val in dict1.items():
# 	val /= len(df_loss_all)
# 	dict1[metrics] = val
# dict1 = {'Average': dict1}
# df_dict1 = pd.DataFrame(dict1).T
# df = pd.concat([df_loss_all, df_dict1])
# print(df)
df = pd.DataFrame(loss_all).T
loss_all['Avg'] = {}
for k, v in df.items():
    avg = v.mean()
    loss_all['Avg'][k] = avg

df = (pd.DataFrame(loss_all).T).sort_index()
df.to_excel(os.path.join(model_path, 'test_result.xlsx'))
print(df)
print('Test start at', test_time)
print('End at', str(datetime.datetime.now()), f'result path:{model_path}')
