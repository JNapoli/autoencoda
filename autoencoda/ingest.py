import argparse
import librosa
import os
import spotipy
import sys
import time
import wget

import numpy as np
import spotipy.util as su


parser = argparse.ArgumentParser(
    description='Use Spotify API to assemble dataset.'
)
parser.add_argument('--spotify_client_id',
                    type=str,
                    required=True,
                    help='Required credential to access Spotify API.')
parser.add_argument('--spotify_client_secret',
                    type=str,
                    required=True,
                    help='Required secret key to access Spotify API.')
parser.add_argument('--path_artists',
                    type=str,
                    required=True,
                    default='../data/artist-keys.txt',
                    help='Path to file containing list of Spotify URIs, one for \
                    each artist to build into the dataset. Example: \
                    "spotify:artist:4tZwfgrHOc3mvqYlEYSvVi" for Daft Punk :).')
parser.add_argument('--path_data_storage',
                    type=str,
                    required=True,
                    default='../data/data_store',
                    help='Directory in which to store mp3 files and other track \
                    data.')
args = parser.parse_args()


def main():
    pass


if __name__ == '__main__':
    pass
