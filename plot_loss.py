import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
#from utils.config import get_config
import numpy as np
import toml
from easydict import EasyDict

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, type=str)
args = parser.parse_args()
model_path = args.model_path
if os.path.isfile(os.path.join('exp', model_path, 'config_loadCKPT.toml')):
    config_path = os.path.join('exp', model_path, 'config_loadCKPT.toml')
elif os.path.isfile(os.path.join('exp', model_path, 'config.toml')):
    config_path = os.path.join('exp', model_path, 'config.toml')

with open(config_path, 'r', encoding='utf-8') as f:
    config = toml.load(f)
    default_config = EasyDict(config)
fold_df = default_config.train.fold_excel
fold_df = pd.read_excel(fold_df, dtype={'subject':str, 'fold':int})

plt.figure()
r, c = 2, len(set(fold_df['fold']))
for i in range(1, len(set(fold_df['fold'])) + 1):
	# if os.path.isfile(os.path.join('exp/', model_path, 'models/Fold'+str(i), 'learning_curve.csv')):
	# 	train_loss = pd.read_csv(os.path.join('exp/',model_path,'models/Fold'+str(i), 'learning_curve.csv'))
	if os.path.isfile(os.path.join('exp/', model_path, 'Fold'+str(i), 'learning_curve.csv')):
		train_loss = pd.read_csv(os.path.join('exp/',model_path,'Fold'+str(i), 'learning_curve.csv'))
		Pear_train_avg_list = train_loss['Train Pearson Corr'].tolist()
		Pear_val_avg_list = train_loss['Val Pearson Corr'].tolist()
		Pear_test_avg_list = train_loss['Test Pearson Corr'].tolist()
		loss_type_val = train_loss.columns[2]
		loss_type_train = train_loss.columns[1]
		loss_type_Test = train_loss.columns[3]

		Err_val_avg_list = train_loss[loss_type_val].tolist()
		Err_train_avg_list = train_loss[loss_type_train].tolist()
		Err_Test_avg_list = train_loss[loss_type_Test].tolist()

		best_epoch = train_loss[loss_type_val].idxmin()
		#best_epoch = len(Err_val_avg_list)-1001
		print('epoch: {}, MSE(T): {}, PR(T): {}, MSE(V): {}, PR(V): {}'.format(best_epoch, Err_train_avg_list[best_epoch],Pear_train_avg_list[best_epoch], Err_val_avg_list[best_epoch], Pear_val_avg_list[best_epoch]))


		
		plt.subplot(r, c, i)
		plt.plot(Err_train_avg_list,'r')
		plt.plot(Err_val_avg_list,'b')
		plt.plot(Err_Test_avg_list,'g')
		plt.axvline(best_epoch, color='y')
		plt.legend(['Train','Validation', 'Test', 'Best Epoch='+str(best_epoch)])
		plt.title('Fold'+ str(i)+ ' '+ 'Loss')

		plt.subplot(r, c, i + c)
		plt.plot(Pear_train_avg_list,'r')
		plt.plot(Pear_val_avg_list,'b')
		plt.plot(Pear_test_avg_list,'g')
		plt.axvline(best_epoch, color='y')
		plt.legend(['Train','Validation', 'Test', 'Best Epoch='+str(best_epoch)])
		plt.title('Pearson Correlation')
		plt.suptitle(model_path.split('/')[-1])
plt.show()
