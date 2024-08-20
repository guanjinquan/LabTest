# LabTest
Classification Task on Pathology Images

## Environment Setup
You can setup the environment by following the steps below.
### Install Conda
```bash
pip install requirements.txt
```

If there is any error, you can install the packages manually.


## Download Images
Download data from Baidu Netdisk, link is given in the word file.


## Preprocess Pathology Images 
In `LabTest/Misc/preprocess_data.py`, you should change the `input_dir` global variable to downloaded data folder. 

After that, run the script to preprocess the data. 

```bash
python preprocess_data.py
```


## Train Model
Run the training script to train the model.  e.g.

```bash
bash ./Scripts/train_vit_small_pathology.sh
```

Then, you can monitor the training process in `./Results/<model>/<runs_id>/results.png` and `./Results/<model>/<runs_id>/log.txt`.

