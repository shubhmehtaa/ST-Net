clc;clear;close all;
addpath(genpath('/media/sail/Elements1/BME_Grad_Project/Joanne/FUS-BBBOpenning_Project/DCE/scripts/1 Within-scan Preprocessing/matlab_toolbox/'))
addpath(genpath('/media/sail/Elements1/BME_Grad_Project/Joanne/FUS-BBBOpenning_Project/DCE/scripts/1 Within-scan Preprocessing/matlab_toolbox/NIfTI_20140122'))
addpath(genpath('matlab_toolbox'))
pathdef

%%
data_path = '/media/sail/Elements/BME_Grad_Project/Joanne/BBB_DL/demo/data/';
%dir_4d_list_High=dir(strcat(data_path,'DCE_High_4d/'));
%dir_4d_list_Low=dir(strcat(data_path,'data_recreate/Raw/DCE_Low_4d/'));
dir_brainmask_list=dir(strcat(data_path,'raw/BrainMask/'));
save_path=strcat(data_path,'raw_process/');

%data_path = '/media/sail/Elements/BME_Grad_Project/Joanne/BBB_DL/';
%dir_4d_list_High=dir(strcat(data_path,'data_recreate/Raw/DCE_High_4d/'));
%dir_4d_list_Low=dir(strcat(data_path,'data_recreate/Raw/DCE_Low_4d/'));
%dir_brainmask_list=dir(strcat(data_path,'data_recreate/Raw/BrainMask/'));
%save_path=strcat(data_path,'data_recreate/');

%% DCE
for dir_i = 3: length(dir_brainmask_list) %[4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    if dir_i > 0
        BrainMask_ID = dir_brainmask_list(dir_i);   % '20201214_212326_IVIM_and_DeepC_FUS_BBB_Scan_ID002_L1_001_1_1_BrainMask.nii.gz'
        BrainMask_ID_name = BrainMask_ID.name       
        subject_ID = BrainMask_ID_name(45 : 45 + 4);	% subject_ID=IDXXX     
        
        %% Brain Mask
        mask_str=load_nii(strcat(data_path, 'raw/BrainMask/', BrainMask_ID_name));
        mask = double(mask_str.img);
        mask(mask == 2) = 1;
        
        %% High 4d DCE signal (with medfilt2)
        t4d_str=load_nii(strcat(data_path, 'raw/DCE_High_4d/', subject_ID, '_HighDose_4d.nii.gz'));
        t4d_vol = double(t4d_str.img);
        smoothed_t4d_str = t4d_str;
        %st4d_str_redo = t4d_str;
        size_all = size(t4d_vol);
        slice = size_all(3);
        acq = size_all(4);
%---------
        t4d_str_redo = t4d_str;
        t4d_vol_redo = zeros(size(t4d_vol));
        smoothed_t4d_vol = zeros(size(t4d_vol));
        mask_4d = smoothed_t4d_vol;
        for i = 1: acq
            t4d_3d = t4d_vol(:, :, :, i);
            for j = 1:slice
                t4d_2d = t4d_3d(:, :, j);
                smoothed_t4d_vol(:, :, j, i) = medfilt2(t4d_2d);
                t4d_vol_redo(:,:,j,i) = t4d_2d;
            end
            mask_4d(:,:,:,i) = mask;
        end
        smoothed_t4d_str.img = smoothed_t4d_vol;
        t4d_str_redo.img = t4d_vol_redo;
        
        %figure(1)
        %st4d_vol_th = st4d_vol;
        %histogram(st4d_vol_th(mask_4d==1 & st4d_vol_th~=0), 100, 'DisplayStyle', 'stair')
        %% Ktrans           
        ktrans_str = load_nii(strcat(data_path, 'raw/Ktrans_High/', subject_ID, '_HighDose_ktrans.nii.gz'));
        ktrans_vol = double(ktrans_str.img);
        ktrans_str_th = ktrans_str;
        ktrans_vol_th = ktrans_vol;
        ktrans_vol_th(ktrans_vol_th < -0.02) = 0;
        ktrans_vol_th(ktrans_vol_th > 0.05) = 0;
        ktrans_str_th.img = ktrans_vol_th;
        smoothed_ktrans_vol = ktrans_vol;
        ktrans_size = size(ktrans_vol);
        smoothed_ktrans_str = ktrans_str;
        for i = 1: slice                           
            smoothed_ktrans_vol(:,:,i) = medfilt2(ktrans_vol(:,:,i));
        end
        smoothed_ktrans_str.img = smoothed_ktrans_vol;

        %% T10: Low 4d DCE signal (with medfilt2)        
        t4d_Low_str = load_nii(strcat(data_path,'raw/DCE_Low_4d/', subject_ID, '_LowDose_4d.nii.gz'));
        t4d_Low_vol = double(t4d_Low_str.img);
        
        t4d_Low_str_redo = t4d_Low_str;
        t4d_Low_vol_redo = zeros(size(t4d_Low_vol));
        for i=1: acq
            t4d_Low_3d = t4d_Low_vol(:, :, :, i);
            for j=1:slice
                t4d_Low_2d = t4d_Low_3d(:, :, j);
                t4d_Low_vol_redo(:,:,j,i) = t4d_Low_2d;
            end
        end
        t4d_Low_str_redo.img = t4d_Low_vol_redo;
        
        T10 = mean(t4d_Low_vol(:,:,:,1:4),4);
        T10_medfilt = T10;
        T10_str = ktrans_str;
        T10_str.img = T10;
        smoothed_T10_str = ktrans_str;
        for i = 1: slice
            T10_medfilt(:,:,i) = medfilt2(T10(:,:,i));
        end
        %T10_str.img = T10;
        smoothed_T10_str.img = T10_medfilt;
        
        %% T10: High 4d DCE signal (with medfilt2)        
        t4d_High_str = load_nii(strcat(data_path,'raw/DCE_High_4d/', subject_ID, '_HighDose_4d.nii.gz'));
        t4d_High_vol = double(t4d_High_str.img);
        
        t4d_High_str_redo = t4d_High_str;
        t4d_High_vol_redo = zeros(size(t4d_High_vol));
        for i=1: acq
            t4d_High_3d = t4d_High_vol(:, :, :, i);
            for j=1:slice
                t4d_High_2d = t4d_High_3d(:, :, j);
                t4d_High_vol_redo(:,:,j,i) = t4d_High_2d;
            end
        end
        t4d_High_str_redo.img = t4d_High_vol_redo;
        
        T10_High = mean(t4d_High_vol(:,:,:,1:4),4);
        T10_High_medfilt = T10_High;
        T10_High_str = ktrans_str;
        T10_High_str.img = T10_High;
        smoothed_T10_str_High = ktrans_str;
        for i = 1: slice
            T10_High_medfilt(:,:,i) = medfilt2(T10_High(:,:,i));
        end
        %T10_str.img = T10;
        smoothed_T10_str_High.img = T10_High_medfilt;
                       
                       
        %% Ktrans Opening Mask
        OP_mask_str=load_nii(strcat(data_path, 'raw/openingMask_High/', subject_ID, '_HighDose_ktrans_mask.nii.gz'));
        OP_mask_vol = double(OP_mask_str.img); 
        muscle_mask_str=load_nii(strcat(data_path, 'raw/MuscleMask_High/', subject_ID, '_HighDose_muscle_mask.nii.gz'));
        muscle_mask_vol = double(muscle_mask_str.img); 
        
        mask_str_redo = mask_str;
        mask_redo = zeros(size(mask));
        OP_mask_str_redo = mask_str;
        OP_mask_redo = zeros(size(OP_mask_vol));
        muscle_mask_str_redo = muscle_mask_str;
        muscle_mask_redo = zeros(size(muscle_mask_vol));
        for i = 1: slice
            mask_redo(:, :, i) = mask(:, :, i);
            OP_mask_redo(:, :, i) = OP_mask_vol(:, :, i);
            muscle_mask_redo(:, :, i) = muscle_mask_vol(:, :, i);
        end
        mask_str_redo.img = mask_redo;
        OP_mask_str_redo.img = OP_mask_redo;
        muscle_mask_str_redo.img = muscle_mask_redo;
        
        % Update Brain Mask
        mask_update_str = mask_str;
        mask_update = mask;
        mask_update(smoothed_ktrans_vol < -0.02) = 0;
        mask_update(smoothed_ktrans_vol > 0.05) = 0;
        mask_update_str.img = mask_update;
                       
        mask_4d_update = mask_4d;
        for i=1: acq
            mask_4d_update(:, :, :, i) = mask_update;
        end
        
        %figure
        %for i =1:16
        %    subplot(3, 6, i)
        %    imagesc(mask(:, :, i))
        %end
        %title('BM')
        %figure
        %for i = 1:18
        %    subplot(3, 6, i)
        %    imagesc(mask_update(:, :, i))
        %end
        %title('BM update')
        
        figure
        subplot(521)
        imagesc(t4d_vol(:, :, 11, 84))
        axis('square')
        subplot(522)
        imagesc(smoothed_t4d_vol(:, :, 11, 84))
        axis('square')

        subplot(523)
        imagesc(T10(:, :, 11))
        axis('square')
        subplot(524)
        imagesc(T10_medfilt(:, :, 11))
        axis('square')
        
        subplot(525)
        imagesc(ktrans_vol(:, :, 11), [0 0.05])
        axis('square')
        subplot(526)
        imagesc(smoothed_ktrans_vol(:, :, 11), [0 0.05])
        axis('square')
        
        subplot(527)
        imagesc(OP_mask_vol(:, :, 11))
        axis('square')
        
        subplot(528)
        imagesc(muscle_mask_vol(:, :, 11))   
        axis('square')
        
        subplot(529)
        t4d_vol_th = t4d_vol;
        smoothed_t4d_vol_th = smoothed_t4d_vol;
        histogram(t4d_vol_th(mask_4d == 1 & t4d_vol_th ~= 0), 100, 'DisplayStyle', 'stair')
        hold on
        histogram(smoothed_t4d_vol_th(mask_4d == 1 & smoothed_t4d_vol_th ~= 0), 100, 'DisplayStyle', 'stair')
        axis('square')
        
        subplot(5, 2, 10)
        imagesc(mask(:, :, 11))
        axis('square')
        
        mkdir(strcat(save_path, 'ktrans_th/'));
        save_nii(ktrans_str_th, strcat(save_path, 'ktrans_th/', subject_ID, '_HighDose_ktrans_th.nii.gz'))    
        %save_nii(t4d_str_redo, strcat(save_path, subject_ID, '_HighDose_4d.nii.gz'))    
        %save_nii(t4d_Low_str_redo, strcat(save_path, subject_ID, '_LowDose_4d.nii.gz')) 
        %save_nii(mask_str_redo, strcat(save_path, BrainMask_ID_name))    
        %save_nii(OP_mask_str_redo, strcat(save_path, subject_ID, '_HighDose_ktrans_mask.nii.gz'))    
        %save_nii(muscle_mask_str_redo, strcat(save_path, subject_ID, '_HighDose_muscle_mask.nii.gz')) 
        mkdir(strcat(save_path, 'T10_LowDose/'));
        save_nii(T10_str, strcat(save_path, 'T10_LowDose/', subject_ID, '_T10_LowDose.nii.gz')) 
        mkdir(strcat(save_path, 'T10_HighDose/'));
        save_nii(T10_High_str, strcat(save_path, 'T10_HighDose/', subject_ID, '_T10_HighDose.nii.gz'))
    end
end

%%
%{
figure(1)
subplot(121)
imagesc(t4d_vol(:,:,11,84))
subplot(122)
imagesc(st4d_vol(:,:,11,84))
figure(2)
subplot(121)
ktrans_vol = ktrans_vol;
imagesc(ktrans_vol(:,:,11), [0 0.04])
subplot(122)
sktrans_vol = sktrans_vol;
imagesc(sktrans_vol(:,:,11), [0 0.04])

figure(3)
subplot(121)
imagesc(T10(:,:,11))
subplot(122)
imagesc(T10_medfilt(:,:,11))
figure(4)
subplot(121)
imagesc(OP_mask(:,:,11))
subplot(122)
imagesc(mask(:,:,11))
%}