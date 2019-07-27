# Hit Predict

Hit Predict is a Python library for predicting the success of your music.

## Overview

This repo contains scripts for building a data ingestion,
pre-processing, and modeling pipeline to build a model that predicts the
probability of an audio sample appearing on the Billboard Hot 100 chart,
a ranking of the top 100 songs of the week. 

Here, audio samples are featurized using information from their Mel
spectrograms (example below), which contain information about the time-evolution of the
different frequencies present in the music. 

![ScreenShot](/figs/spectrogram.png)

## Installation 

The following steps have been verified to be reproducible on MacOS. The code uses python 3.7.3 and can be installed as follows:

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

### Minimal use case

The most common use case will be to make a prediction on a new mp3 file. This
may be done by providing the mp3 file's full path to the prediction script:

```bash
python predict.py --path_track /FULL/PATH/TO/AUDIO/FILE/track.mp3
```

### Full pipeline

For each script in the [autoencoda](/autoencoda/) subdirectory, arguments and their descriptions can be viewed as follows:

```bash
python SCRIPT-NAME.py --help
```

The full pipeline can be executed as follows: 

1. Run [billboard_query.py](/autoencoda/billboard_query.py), which uses the [billboard.py](https://github.com/guoguo12/billboard-charts) package to scrape songs from the [Billboard](https://www.billboard.com/charts/hot-100) Hot-100 chart. This generates a set of tracks that appeared on the chart from a user-specified date to the present.

2. Run [ingest.py](/autoencoda/ingest.py) to get mp3 samples for each track in the set using the [Spotipy](https://spotipy.readthedocs.io/en/latest/) package and to featurize them using [librosa](https://github.com/librosa). A Spotify [Client ID](https://developer.spotify.com/documentation/general/guides/app-settings/) and secret key will be required to make requests via the Spotify API, both of which can be passed as arguments to the script.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## License
[MIT](https://choosealicense.com/licenses/mit/)
