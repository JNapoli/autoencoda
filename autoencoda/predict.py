import argparse
import librosa
import logging

import numpy as np
import os.path as path
import tensorflow as tf

from tensorflow.keras.models import load_model


def print_result(result, thresh):
    """Prints output message according to the result and the
       prediction threshold.
    """
    if result >= thresh:
        print('Congrats, it looks like you have a hit! ' + \
              'This track is a hit with {:.2f} probability.'.format(result))
    else:
        print("Sorry, this doesn't look like a hit. Better luck next time!")
    return None


def main(args):
    path_full_self = path.realpath(__file__)
    path_base_self = path.dirname(path_full_self)
    path_model = path.join(path_base_self, '..', 'model', 'model.h5')
    if not path.exists(path_model):
        raise IOError('Model file could not be located.')
    if not path.exists(args.path_track):
        raise IOError("Can't find your mp3 file! Please provide a valid path to an mp3.")
    if not args.path_track.endswith('.mp3'):
        raise ValueError('I expect an mp3 file.')
    else:
        model = load_model(path_model)
        audio, sr = librosa.load(args.path_track)
        sg = np.array([librosa.feature.melspectrogram(y=audio, sr=sr).mean(axis=1)])
        X = sg / sg.max()
        result = model.predict(X)[0][0]
        print_result(result, args.threshold)


if __name__ == '__main__':
    # Take care of future deprecation warning
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Predict the popularity of your track!'
    )
    parser.add_argument('--path_track',
                        type=str,
                        required=True,
                        help='mp3 file containing your music.')
    parser.add_argument('--threshold',
                        type=float,
                        required=False,
                        default=0.5,
                        help='Probability threshold to use for inference.')
    args = parser.parse_args()

    # Run inference
    main(args)
