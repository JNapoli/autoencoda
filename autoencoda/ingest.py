import argparse
import librosa
import logging
import os
import spotipy
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


def get_spotify_instance(client_id, client_secret):
    """Get a Spotify instance that can pull information from the Spotify API.

    Args:
        client_id (str): Client ID for accessing Spotify API.
        client_secret (str): Secret client ID (key) for accessing Spotify API.

    Returns:
        S: A Spotify instance that can be used to query the Spotify API.
    """
    token = su.oauth2.SpotifyClientCredentials(
        client_id=client_id,
        client_secret=client_secret
    )
    cache_token = token.get_access_token()
    try:
        S = spotipy.Spotify(cache_token)
    except SpotifyOauthError:
        logging.exception('Make sure you are using the right Spotify \
                          credentials.')
    return S


def get_tracks_with_previews(artist_id, spotify):
    """
    """
    # Get all albums for specified artist.
    albums = spotify.artist_albums(artist_id)['items']

    # Properly format album IDs for Spotify API.
    album_IDs = [
        'spotify:album:{}'.format(elem['id']) for elem in albums
    ]

    # Figure out which tracks actually have previews.
    # We only want those.
    tracks_with_previews = []
    for AID in album_IDs:
        tracks = spotify.album_tracks(AID)['items']
        tracks_with_previews.extend([
            t for t in tracks if t['preview_url'] is not None
        ])

    return [
        {
            'track_id': 'spotify:track:{}'.format(elem['id']),
            'artist_id': artist_id,
            'preview_url': elem['preview_url']
        }
        for elem in tracks_with_previews
    ]


def main():
    logging.basicConfig(filename='ingestion.log',
                        level=logging.DEBUG)


if __name__ == '__main__':
    pass
