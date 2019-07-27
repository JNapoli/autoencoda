# Hit Predict

Hit Predict is a Python library for predicting the success of your music.

## Overview

This repo contains Python scripts for building a data ingestion,
pre-processing, and modeling pipeline to build a model that predicts the
probability of an audio sample appearing on the Billboard Hot 100 chart,
a ranking of the top 100 songs of the week. 

Here, audio samples are featurized using information from their Mel
spectrograms (example below), which contain information about the time-evolution of the
different frequencies present in the music. 

![ScreenShot](/figs/spectrogram.png)

## Installation [MacOS and Linux]

The code uses python 3.7.3 and can be installed via the following steps:

1. Clone this repo using:
```bash
git clone https://github.com/JNapoli/autoencoda.git
cd autoencoda/
```

2. Create a virtual environment (ensure that [Venv](https://docs.python.org/3.6/library/venv.html#module-venv) is available):
```bash
python3 -m venv my-env
source my-env/bin/activate
```

3. Install required packages via pip3:
```bash
pip3 install -r requirements.txt
```

## Usage

Scripts to execute the full project pipeline can be found in the autoencoda subdirectory of the repo root directory. Each script requires arguments, which can be listed by running:

```bash
python SCRIPT-NAME.py --help
```

The most common usage will be to make a prediction on a new track. This
can be done by providing the full mp3 file path to the predict.py script:

```bash
python predict.py --path_track /FULL/PATH/TO/AUDIO/FILE/track.mp3
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
[MIT](https://choosealicense.com/licenses/mit/)
