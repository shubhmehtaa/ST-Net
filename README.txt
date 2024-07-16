1. How to run the model: open a terminal, conda activate your envieonment (mine is "Schi"), and run "python train.py", the results will be saved in exp folder
2. config.toml: set hyperparameters
	- ckpt_path: If you want to train the modek from scratch, please set this value as 0. If you want to load checkpoint, please indicate the checkpoint folder.
	- fold_excel: Generate an .xlsx file to set cross validation data fold
	- curr_fold: Indicate which fold you are going to run
	- Augmentation: If you want to use aygmentation, please set this value as 1; otherwise, please set it as 0.
3. utils/dataset_ST.py: dataloader
4. The deep learning architecture is in model folder and the results are saved in exp folder
5. plot_loss.py: open a terminal and run $ python plot_loss.py --model_path='{your model folder name in the exp folder}'
6. How to run the testing using pretrained weight: When you train the model, a config.toml will save in the result folder and will be used in the testing. Run $ python test.py --model_path='{your model folder name in the exp folder}'. Terminal will also show the performance for each testing data.
7. If you have already run the test.py and only want to print out the testing performance, run $ python print_metrics.py



