import argparse
import librosa
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import os.path as path

from librosa.display import specshow


def read_pickle(pickle_path):
    """Lightweight wrapper function to load pickled data.

    Args:
        pickle_path (str): Path to the pickle file to load.

    Returns:
        pickle file contents
    """
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)


def get_tracks_in_directory(path_dir):
    """Load all track pickles in the provided directory.

    Args:
        path_dir (str): Path to the directory containing the track pickles.

    Returns:
        tracks (list): List of dictionaries containing track data.
    """
    tracks = []
    dir_contents = [elem for elem in os.listdir(path_dir) if elem[-2:] == '.p']
    for fn in dir_contents:
        path_full = os.path.join(path_dir, fn)
        tracks.append(read_pickle(path_full))
    return tracks


def normalize_spectra(spectra):
    """Normalize spectra.

    Args:
        spectra (np.ndarray): np.ndarray with shape [n_spectra, n_frequency]

    Returns:
        spectra: The original array, but normalized to unit area across axis 1
    """
    assert all([elem.shape == spectra[0].shape for elem in spectra]), \
           'Arrays are of inconsistent shape.'
    # Normalize spectra to unit area
    spectra = spectra / np.trapz(spectra, axis=1)[:, np.newaxis]
    assert np.trapz(spectra, axis=1).all(), 'Arrays not normalized to unit area.'
    return spectra


def main(args):
    path_full_self = path.realpath(__file__)
    path_base_self = path.dirname(path_full_self)
    path_log = path.join(path_base_self,
                         '..',
                         'logs',
                         'preprocess.log')

    # Create directory to save our results
    if not os.path.exists(args.preprocessed): os.mkdir(args.preprocessed)

    # Set verbosity level for debugging
    logging.basicConfig(filename=path_log, level=logging.DEBUG)

    # Load raw data
    tracks_billboard = get_tracks_in_directory(args.tracks_billboard)
    tracks_not_billboard = get_tracks_in_directory(args.tracks_not_billboard)

    # Extract average (time-independent) spectra
    spectra_billboard = np.array([elem['spectrogram'].mean(axis=1)
                                  for elem in tracks_billboard])
    spectra_not_billboard = np.array([elem['spectrogram'].mean(axis=1)
                                     for elem in tracks_not_billboard])

    # Extract full (time-dependent) spectra
    # Each element of these arrays has dimensions [n_time, n_mel_freq]
    spectra_billboard_full = np.array([elem['spectrogram'].T
                                      for elem in tracks_billboard])
    spectra_not_billboard_full = np.array([elem['spectrogram'].T
                                          for elem in tracks_not_billboard])

    # Free up memory
    del tracks_billboard
    del tracks_not_billboard

    # Normalize time-independent spectra
    spectra_billboard = normalize_spectra(spectra_billboard)
    spectra_not_billboard = normalize_spectra(spectra_not_billboard)

    # Scale time-independent spectra
    spectra_billboard = spectra_billboard / \
                        np.max(spectra_billboard, axis=1)[:, np.newaxis]
    spectra_not_billboard = spectra_not_billboard / \
                            np.max(spectra_not_billboard, axis=1)[:, np.newaxis]

    # Scale time-dependent spectra
    spectra_billboard_full = np.array([
        elem / np.max(elem, axis=1)[:, np.newaxis]
        for elem in spectra_billboard_full
        if elem.shape == spectra_billboard_full[0].shape
    ])
    spectra_not_billboard_full = np.array([
        elem / np.max(elem, axis=1)[:, np.newaxis]
        for elem in spectra_not_billboard_full
        if elem.shape == spectra_not_billboard_full[0].shape
    ])

    # Save preprocessed data
    np.save(os.path.join(args.preprocessed,
                         'preprocessed-billboard-no-subtract-scaled.npy'),
                         spectra_billboard)
    np.save(os.path.join(args.preprocessed,
                         'preprocessed-not-billboard-no-subtract-scaled.npy'),
                         spectra_not_billboard)
    np.save(os.path.join(args.preprocessed,
                         'preprocessed-billboard-full-no-subtract-scaled.npy'),
                         spectra_billboard_full)
    np.save(os.path.join(args.preprocessed,
                         'preprocessed-not-billboard-full-no-subtract-scaled.npy'),
                         spectra_not_billboard_full)

    # Verbosity
    logging.info("Preprocessed {:d} Billboard tracks and {:d} \
                 non-Billboard tracks.".format(spectra_billboard.shape[0],
                                               spectra_not_billboard.shape[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess and store the data for my model.'
    )
    parser.add_argument('tracks_billboard',
                        type=str,
                        help='Path containing tracks that appeared on Billboard.')
    parser.add_argument('tracks_not_billboard',
                        type=str,
                        help='Path containing tracks that did not appear on Billboard.')
    parser.add_argument('preprocessed',
                        type=str,
                        help='Path to the directory in which to save preprocessed data.')
    args = parser.parse_args()
    main(args)
