import argparse
import librosa
import os

import preprocess

from keras.models import model_from_json


def main(args):
    assert os.path.exists(args.path_track),
           "Can't find your mp3 file! Please provide a valid path."
    assert args.path_track[-3:] == 'mp3',
           'Please provide the audio file in mp3 format.'
    audio, sr = librosa.load(args.path_track)
    sg = librosa.feature.melspectrogram(y=audio, sr=sr)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict the popularity of your track!'
    )
    parser.add_argument('--path_track',
                        type=str,
                        required=True,
                        help='mp3 file containing your music.')
    args = parser.parse_args()
    main(args)
