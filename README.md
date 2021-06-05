# The effect of phased recurrent units in the classification of multiple catalogs of astronomical lightcurves
## The L+P architecture

This repository contains the implementation of the light curve classifier described in C. Donoso-Oliva et al., 2021.
![alt text](https://github.com/cridonoso/plstm_tf2/blob/master/scores.png?raw=true)

**NOTE: A new version of the L+P is being developed at this link. It will contain new experimental features and code best practices. (not yet evaluated)**

## Requirements
The faster way to run the code is creating a [conda](https://docs.conda.io/en/latest/miniconda.html) environment as follows:
```
conda env create -f environment.yml
```
the first line on the YAML file defines the name of the environment. Feel free to change it.
<br><br>
Optionally, you can use the `requirements.txt` file to install dependencies using pip
```
pip install -r requirements.txt
```

## Folders and files
- `./models/`: This folder includes the LSTM and Phased LSTM model class.
  - `plstm.py`: Phased LSTM classifier following the architecture explained in the paper.
  - `lstm.py`: LSTM classifier following the architecture explained in the paper.
  - `./layers/`: Custom layers used on this work
    - `phased.py`: Phased LSTM unit. It consists in a LSTM + time gate
- `data.py`: Contains relevant function for loading and creating [records](https://www.tensorflow.org/tutorials/load_data/tfrecord).
- `get_data.py`: Script to download data
- `main.py`: Main script that loads the data and instances the models (use --help to see running options) for training
- `predict`: Prediction script which receive 4 (sys.argv) arguments: 
  - dataset  : record path
  - rnn_type : either plstm or lstm 
  - fold_n   : according to our kfold
  - norm     : nomalization technique -i.e., n1 or n2
- `{train, test}_script.py`: Code routines for hyperparameter tuning
### Additional not included folders
For storage reasons we do not upload `experiments` and `results` folders which contain adjusted models and metrics/figures, respectively. Although they are not in this repo, you can download them using this link. Once downloaded paste the folders into the root directory.

## Downloading data
You can download data in their raw format or preprocessed record. Getting records implies using the same folds as the author.
<br>
### Commands
- `python get_data.py --help`: to see script options.
- `python get_data.py --dataset <name>`: to download raw dataset.
- e.g. Downloading LINEAR records, 
  ```
    python get_data.py --dataset linear --record 
  ```
 Alternatively, you can download directly from [google drive](https://drive.google.com/drive/folders/1m2fXqn25LYSyG5jEbpM3Yfdbpx9EQCDo?usp=sharing)

## Creating your own data
For custom training, please convert your data to `tf.Record` using the standard function `create_record(light_curves, labels, path='')` in `./data.py`
