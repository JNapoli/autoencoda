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


def add_spotify_audio_features(track_list, spotify):
    """
    """
    for track in track_list:

        # Get precomputed audio features.
        audio_features = spotify.audio_features(track['track_id'])[0]
        track['popularity'] = spotify.track(track['track_id'])['popularity']

        # Add numerical features to the data.
        for feature, value in audio_features.items():
            if not feature in track:
                track[feature] = value

    return track_list


def fetch_mp3_files(track_list, dir_save_base):
    """
    """
    if not os.path.exists(dir_save_base):
        os.mkdir(dir_save_base)

    for track in track_list:
        dir_track_data = os.path.join(dir_save_base,
                                      'track_{}'.format(track['track_id']))
        if not os.path.exists(dir_track_data):
            os.mkdir(dir_track_data)
        path_mp3 = os.path.join(dir_track_data, 'preview.mp3')
        wget.download(track['preview_url'], path_mp3)
        track['path_mp3'] = path_mp3

    return track_list


def compute_spectrograms(track_list, **kwargs_spec):
    """
    """
    for track in track_list:
        audio, sr = librosa.load(track['path_mp3'])
        sg = librosa.feature.melspectrogram(y=audio, sr=sr, **kwargs_spec)
        track['spectrogram'] = sg
        track['sr'] = sr

    return track_list


def main(args):
    logging.basicConfig(filename='ingestion.log',
                        level=logging.DEBUG)
    with open(args.path_artists, 'r') as f:
        artist_URIs = [
            line.strip() for line in f.readlines()
        ]
    try:
        assert len(artist_URIs) > 0
    except AssertionError:
        logging.exception('The artist list is empty.')
    logging.info('Initializing Spotify instance...')
    spotify = get_spotify_instance(args.spotify_client_id,
                                   args.spotify_client_secret)
    logging.info('Done.')
    t0 = time.time()
    n_tracks_processed = 0
    # Fetch data for each artist in our list.
    # Monitor time it takes per artist in order to keep an eye on cost.
    for i, artist_URI in enumerate(artist_URIs, start=1):
        # Subselect tracks that have previews.
        artist_tracks = get_tracks_with_previews(artist_URI, spotify)
        n_tracks_processed += len(artist_tracks)
        # Add some precomputed features by Spotify.
        artist_tracks = add_spotify_audio_features(artist_tracks, spotify)
        # Get mp3s from the web and save them.
        artist_tracks = fetch_mp3_files(artist_tracks,
                                        args.path_data_storage)
        # Compute the spectrograms.
        artist_tracks = compute_spectrograms(artist_tracks)
        # Timing per track processed:
        elapsed_per_track = (time.time() - t0) / float(n_tracks_processed)
    logging.info('Processing took {:.2f} seconds per track.'.format(elapsed_per_track))


if __name__ == '__main__':
    main(args)
