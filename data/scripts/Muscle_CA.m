clc;clear;close all;
addpath(genpath('/media/sail/Elements1/BME_Grad_Project/Joanne/FUS-BBBOpenning_Project/DCE/scripts/1 Within-scan Preprocessing/matlab_toolbox/'))
addpath(genpath('/media/sail/Elements1/BME_Grad_Project/Joanne/FUS-BBBOpenning_Project/DCE/scripts/1 Within-scan Preprocessing/matlab_toolbox/NIfTI_20140122'))
addpath(genpath('matlab_toolbox'))
pathdef

%% muscle contrast agent concentraiton with medfilt2
data_path = '/media/sail/Elements/BME_Grad_Project/Joanne/BBB_DL/demo/data/';
dir_Low_4d_list=dir(strcat('/media/sail/Elements/BME_Grad_Project/Joanne/BBB_DL/demo/data/raw/DCE_Low_4d/'));

%data_path = '/media/sail/Elements/BME_Grad_Project/Joanne/BBB_DL/data_recreate/';
%dir_Low_4d_list=dir(strcat('/media/sail/Elements/BME_Grad_Project/Joanne/BBB_DL/data_recreate/Raw/DCE_Low_4d/'));
%%
for dir_i = 3 : length(dir_Low_4d_list)
    Low_folder = dir_Low_4d_list(dir_i);
	Low_name = Low_folder.name;
    subject_ID = Low_name(1:5);
    
	t4d_Low_str = load_nii(strcat(data_path, 'raw/DCE_Low_4d/', Low_name));
	t4d_Low_vol = double(t4d_Low_str.img);  
	t4d_High_str = load_nii(strcat(data_path, 'raw/DCE_High_4d/', subject_ID, '_HighDose_4d.nii.gz'));
	t4d_High_vol = double(t4d_High_str.img);  
    
    muscle_str = strcat(subject_ID, '_HighDose_muscle_mask.nii.gz');
	muscle_mask_str = load_nii(strcat(data_path, 'raw/MuscleMask_High/', subject_ID, '_HighDose_muscle_mask.nii.gz'));
	muscle_mask_img = double(muscle_mask_str.img);

	for i=1:84
        t4d_Low_vol_3d = t4d_Low_vol(:,:,:,i);
        t4d_High_vol_3d = t4d_High_vol(:,:,:,i);
        mean_muscle_Low = mean(t4d_Low_vol_3d(muscle_mask_img > 0));
        mean_muscle_High = mean(t4d_High_vol_3d(muscle_mask_img > 0));
        
        CA_Low(i,1) = mean_muscle_Low;
        CA_High(i,1) = mean_muscle_High;
	end
    save_path_Low=strcat(data_path, 'raw_process/Muscle_CA_Low/');
    mkdir(save_path_Low)
    save_path_High=strcat(data_path, 'raw_process/Muscle_CA_High/');
    mkdir(save_path_High)
    save(strcat(save_path_Low, subject_ID, '_CA_Low'), 'CA_Low')
    save(strcat(save_path_High, subject_ID, '_CA_High'), 'CA_High')

    min_y = min(min(CA_Low(:)), min(CA_High(:))) - 100;
    max_y = max(max(CA_Low(:)), max(CA_High(:))) + 100;
    
    figure
    subplot(121)
    plot(CA_Low,'o-')
    ylim([min_y, max_y])
    t.FontSize=14;
    t.VerticalAlignment='bottom';
    title([subject_ID, 'Low'])
    subplot(122)
    plot(CA_High,'o-')
    ylim([min_y, max_y])
    t.FontSize=14;
    t.VerticalAlignment='bottom';
    title([subject_ID, 'High'])
end
