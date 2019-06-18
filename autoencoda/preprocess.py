import argparse
import librosa
import logging
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

from librosa.display import specshow


def read_pickle(full_path):
    with open(full_path, 'rb') as f:
        return pickle.load(f)


def get_tracks_in_directory(path_dir):
    tracks = []
    dir_contents = os.listdir(path_dir)
    # Make sure directory contains only pickled tracks.
    assert [elem[-2:] == '.p' for elem in dir_contents]
    assert all([elem[-2:] == '.p' for elem in dir_contents])
    for fn in dir_contents:
        path_full = os.path.join(path_dir, fn)
        tracks.append(read_pickle(path_full))
    return tracks


def normalize_spectra(spectra):
    assert all([elem.size == spectra[0].size for elem in spectra])
    # Normalize spectra to unit alrea
    # Normalize spectra to unit area
    spectra = np.array(spectra)
    spectra = spectra / np.trapz(spectra, axis=1)[:, np.newaxis]
    assert np.trapz(spectra, axis=1).all()
    # Subtract mean
    spectra -= spectra.mean(axis=0)
    return spectra


def main(args):
    # Set verbosity level for debugging.
    logging.basicConfig(filename='preprocess.log',
                        level=logging.DEBUG)
    # Load raw data.
    tracks_billboard = get_tracks_in_directory(args.tracks_billboard)
    tracks_not_billboard = get_tracks_in_directory(args.tracks_not_billboard)
    # Extract spectra.
    spectra_billboard = [elem['spectrogram'].mean(axis=1)
                         for elem in tracks_billboard]
    spectra_not_billboard = [elem['spectrogram'].mean(axis=1)
                             for elem in tracks_not_billboard]
    # Normalize.
    spectra_billboard = normalize_spectra(spectra_billboard)
    spectra_not_billboard = normalize_spectra(spectra_not_billboard)
    np.save(os.path.join(args.preprocessed, 'preprocessed-billboard.npy'),
            spectra_billboard)
    np.save(os.path.join(args.preprocessed, 'preprocessed-not-billboard.npy'),
            spectra_not_billboard)
    logging.info("Preprocessed {:d} Billboard tracks and {:d} \
                 non-Billboard tracks.".format(spectra_billboard.shape[0],
                                               spectra_not_billboard.shape[1]))


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
    parser.add_argument('--images_billboard',
                        type=str,
                        required=False,
                        default='../data/preprocessed_images/billboard/',
                        help='Path containing spectrogram images of the Billboard \
                        tracks.')
    parser.add_argument('--images_not_billboard',
                        type=str,
                        required=False,
                        default='../data/preprocessed_images/not_billboard/',
                        help='Path containing spectrogram images of tracks that \
                        are not in Billboard set.')
    parser.add_argument('--preprocessed',
                        type=str,
                        required=False,
                        default='../data/preprocessed/',
                        help='Path containing preprocessed data.')
    args = parser.parse_args()
    main(args)
