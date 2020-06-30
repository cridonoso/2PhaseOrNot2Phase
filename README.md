# An Ensemble Recurrent Neural Architecture for the Classification of Unfolded Light Curves

In this repository you can find the code for the light curve classifier described in Donoso et.al. 20xx. 

## Requirements
The faster way to run the code is creating a [conda](https://docs.conda.io/en/latest/miniconda.html) enviroment as follows:
```
conda env create -f environment.yml
```
the first line on the YAML file defines the name of the enviorment. Feel free to change it. 
<br><br>
Additionaly, you can use the `requirements.txt` file to install dependencies using pip
```
pip install -r requirements.txt
```

## Folders and files
- `main.py`: It is the principal script that loads the data and instances the models (use --help to see running options)
- `data.py`: Contains relevant function for loading and creating [records](https://www.tensorflow.org/tutorials/load_data/tfrecord).
- `./models/`: this folder includes the LSTM and Phased LSTM model class
  - `plstm.py`: Phased LSTM classifier following the architecture exaplined on the paper.
  - `lstm.py`: LSTM classifier following the architecture explained on the paper.
  - `./layers/`: Additional layers used on this work
    - `phased.py`: Phased LSTM unit. It consists in a LSTM + time gate 
