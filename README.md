# An Ensemble Recurrent Neural Architecture for the Classification of Unfolded Light Curves

In this repository, you can find the code for the light curve classifier described in C. Donoso Oliva et al., 20xx.

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
- `main.py`: The principal script that loads the data and instances the models (use --help to see running options)
- `get_data.py`: Script to download data
- `data.py`: Contains relevant function for loading and creating [records](https://www.tensorflow.org/tutorials/load_data/tfrecord).
- `./models/`: This folder includes the LSTM and Phased LSTM model class.
  - `plstm.py`: Phased LSTM classifier following the architecture explained in the paper.
  - `lstm.py`: LSTM classifier following the architecture explained in the paper.
  - `./layers/`: Custom layers used on this work
    - `phased.py`: Phased LSTM unit. It consists in a LSTM + time gate

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
