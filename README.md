# ST-Net
This is the official code for the IEEE ISBI 2024 paper "DEEP LEARNING ENABLES REDUCED GADOLINIUM DOSE FOR CONTRAST-ENHANCED BLOOD-BRAIN BARRIER OPENING QUANTITATIVE MEASUREMENT" available at: https://ieeexplore.ieee.org/document/10635626
doi: 10.1109/ISBI56570.2024.10635626

How to run the model: 
1. Open a terminal, conda activate your environment (mine is "Schi"), and run "python train.py", the results will be saved in "exp" folder
2. config.toml: set hyperparameters
	- ckpt_path: If you want to train the model from scratch, please set this value as 0. If you want to load checkpoint, please indicate the checkpoint folder.
	- fold_excel: Generate a .xlsx file to set cross validation data fold
	- curr_fold: Indicate which fold you are going to run
	- Augmentation: If you want to use augmentation, please set this value as 1; otherwise, please set it as 0.
3. utils/dataset_ST.py: dataloader
4. The deep learning architecture is in the "model" folder and the results are saved in "exp" folder
5. plot_loss.py: open a terminal and run $ python plot_loss.py --model_path='{your model folder name in the exp folder}'
6. How to run the testing using pretrained weight: When you train the model, a config.toml will save in the result folder and will be used in the testing. Run $ python test.py --model_path='{your model folder name in the exp folder}'. Terminal will also show the performance for each testing data.
7. If you have already run the test.py and only want to print out the testing performance, run $ python print_metrics.py

Note: Please send an email to either sm5321@columbia.edu or jg3400@columbia.edu for the data files that were used. We will also be happy to provide you with a video tutorial of how to run the ST-Net with our data. 
