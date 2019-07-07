# Hit Predict

Hit Predict is a Python library for predicting the success of your music.


## Overview

This repo contains Python scripts for automating an ingestion,
preprocessing, and modeling pipeline to build a model that predicts the
probability of an audio sample appearing on the Billboard Hot 100 chart,
a ranking of the top 100 songs of the week. 


![Alt text](/figs/spectrogram.jpg?raw=true "Optional Title")




TODO: example spectrogram. 


## Installation

Clone the Hit Predict repo:
```bash
git clone https://github.com/JNapoli/hitpredict.git
```

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install package dependencies:
```bash
pip install -r requirements.txt
```

## Usage


```python
python autoencoda/billboard_query.py --path_save /Users/joe/Desktop/test-dependencies/autoencoda/data/billboard/billboard-scrape.p
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
