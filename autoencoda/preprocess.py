import argparse
import librosa
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from librosa.display import specshow


def read_pickle(pickle_path):
    """Lightweight wrapper function to load pickled data.
    """
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)


def get_tracks_in_directory(path_dir):
    tracks = []
    dir_contents = os.listdir(path_dir)
    # Make sure directory contains only pickled tracks.
    assert all([elem[-2:] == '.p' for elem in dir_contents])
    for fn in dir_contents:
        path_full = os.path.join(path_dir, fn)
        tracks.append(read_pickle(path_full))
    return tracks


def normalize_spectra(spectra):
    assert all([elem.shape == spectra[0].shape for elem in spectra]), \
           'Arrays are of inconsistent shape.'
    # Normalize spectra to unit area
    spectra = spectra / np.trapz(spectra, axis=1)[:, np.newaxis]
    assert np.trapz(spectra, axis=1).all(), 'Arrays not normalized to unit area.'
    return spectra


def main(args):
    # Set verbosity level for debugging
    logging.basicConfig(filename='preprocess.log', level=logging.DEBUG)

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
    #assert (spectra_billboard_full <= 1.0).all(), 'Data scaling failed.'
    #assert (spectra_not_billboard_full <= 1.0).all(), 'Data scaling failed.'

    # Save preprocessed data
    np.save(os.path.join(args.preprocessed, 'preprocessed-billboard-no-subtract-scaled.npy'),
            spectra_billboard)
    np.save(os.path.join(args.preprocessed, 'preprocessed-not-billboard-no-subtract-scaled.npy'),
            spectra_not_billboard)
    np.save(os.path.join(args.preprocessed, 'preprocessed-billboard-full-no-subtract-scaled.npy'),
            spectra_billboard_full, allow_pickle=True)
    np.save(os.path.join(args.preprocessed, 'preprocessed-not-billboard-full-no-subtract-scaled.npy'),
            spectra_not_billboard_full, allow_pickle=True)

    # Verbosity
    logging.info("Preprocessed {:d} Billboard tracks and {:d} \
                 non-Billboard tracks.".format(spectra_billboard.shape[0],
                                               spectra_not_billboard.shape[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Preprocess and store the data for my model.'
    )
    parser.add_argument('--tracks_billboard',
                        type=str,
                        required=False,
                        default='../data/cache_tracks_billboard/',
                        help='Path containing tracks that appeared on Billboard.')
    parser.add_argument('--tracks_not_billboard',
                        type=str,
                        required=False,
                        default='../data/cache_tracks_not_billboard/',
                        help='Path containing tracks that did not appear on Billboard.')
    parser.add_argument('--preprocessed',
                        type=str,
                        required=False,
                        default='../data/preprocessed/',
                        help='Path containing preprocessed data.')
    args = parser.parse_args()
    main(args)
