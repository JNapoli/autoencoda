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

## Installation

Clone the Hit Predict repo:
```bash
git clone https://github.com/JNapoli/autoencoda.git
```

Dependencies can be installed using the package manager
[pip](https://pip.pypa.io/en/stable/) and the requirements.txt file in
the repo root directory:
```bash
pip install -r requirements.txt
```

## Usage

Each python script here requires arguments. List all script arguments
and short descriptions of each argument by running:

```bash
python SCRIPT-NAME.py --help
```

The most common usage will be to make a prediction on a new track. This
can be done by running:

```bash
python predict.py --path_track /FULL/PATH/TO/AUDIO/FILE/track.mp3
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
[MIT](https://choosealicense.com/licenses/mit/)
