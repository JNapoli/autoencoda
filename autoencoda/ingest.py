import argparse
import librosa
import logging
import os
import pickle
import spotipy
import sys
import time
import wget

import spotipy.util as su


parser = argparse.ArgumentParser(
    description='Use Spotify API to assemble dataset.'
)
parser.add_argument('--path_raw_dat_billboard',
                    type=str,
                    required=False,
                    default='../data/raw/billboard-scrape.p',
                    help='Path to file containing Billboard scrape result.')
parser.add_argument('--path_data_mp3',
                    type=str,
                    required=False,
                    default='../data/data_mp3/',
                    help='Directory in which to store mp3 files and other track \
                    data.')
parser.add_argument('--spotify_client_id',
                    type=str,
                    required=True,
                    help='Required credential to access Spotify API.')
parser.add_argument('--spotify_client_secret',
                    type=str,
                    required=True,
                    help='Required secret key to access Spotify API.')
parser.add_argument('--path_data_set_base',
                    required=False,
                    default='../data/processed/',
                    help='Base directory containing data sets as json entries, \
                    one for each artist.')
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


def get_spotify_from_billboard(bb_track, bb_artist, spotify):
    # Get the track
    track_items = spotify.search(bb_track)['tracks']['items']
    track_URI, artist_URI = None, None
    if len(track_items) > 0:
        for item in track_items:
            if item['name'] == bb_track:
                track_URI = 'spotify:track:{}'.format(item['id'])
                item_artists = item['artists']
                for i_artist in item_artists:
                    if i_artist['name'] == bb_artist:
                        artist_URI = 'spotify:artist:{}'.format(i_artist['id'])
            if track_URI and artist_URI:
                break
    return track_URI, artist_URI


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

        # Add numerical features to the data.
        for feature, value in audio_features.items():
            if not feature in track:
                track[feature] = value

        track['popularity'] = spotify.track(track['track_id'])['popularity']

    return track_list


def fetch_mp3_files(track_list, dir_mp3s):
    """
    """
    if not os.path.exists(dir_mp3s):
        logging.info('Creating base directory for mp3s at {}.'.format(dir_mp3s))
        os.mkdir(dir_mp3s)

    for track in track_list:
        path_mp3 = os.path.join(dir_mp3s,
                                'track_{}.mp3'.format(track['track_id']))
        if not os.path.exists(path_mp3):
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


def cache_artist_data(artist_tracks, dir_base):
    """
    """
    # Save data for each artist separately. Makes it easier for us to query
    # by genre or time period.
    assert len(artist_tracks) > 0
    if not os.path.exists(dir_base): os.mkdir(dir_base)
    artist_id = artist_tracks[0]['artist_id']
    path_pickle_artist = os.path.join(dir_base,
                                      artist_id + '.p')
    with open(path_pickle_artist, 'wb') as f_save:
        pickle.dump(path_pickle_artist,
                    f_save,
                    protocol=pickle.HIGHEST_PROTOCOL)
    return None


def has_mp3_preview(track_URI, spotify):
    return spotify.track(track_URI)['preview_url'] is not None


def flag_billboard_entries(track_list, billboard_entry_URIs):
    for track in track_list:
        if track['track_id'] in billboard_entry_URIs:
            track['billboard'] = True
        else:
            track['billboard'] = False
    return track_list


def main(args):
    # Set verbosity level for debugging.
    logging.basicConfig(filename='ingestion.log',
                        level=logging.DEBUG)

    # Get Spotify instance for querying Spotify API.
    logging.info('Initializing Spotify instance...')
    spotify = get_spotify_instance(args.spotify_client_id,
                                   args.spotify_client_secret)
    logging.info('Done.')

    # Load raw Billboard 100 data.
    with open(args.path_raw_dat_billboard, 'rb') as f:
        billboard_raw = pickle.load(f)

    # Get Spotify URIs from raw strings queried from Billboard.
    billboard_items_with_URIs = [
        get_spotify_from_billboard(elem[0], elem[1], spotify)
        for elem in billboard_raw
    ]

    # Filter out all elements that have None values.
    billboard_items_with_URIs = [
        elem for elem in billboard_items_with_URIs if not None in elem
    ]

    # Filter out all elements that don't have a preview associated with them.
    billboard_items_with_URIs = [
        elem for elem in billboard_items_with_URIs if has_mp3_preview(elem[0],
                                                                      spotify)
    ]

    # For each artist that appears on Billboard, we want to get all available
    # mp3 previews.
    artist_URIs = [
        elem[1] for elem in billboard_items_with_URIs
    ]
    track_URIs_in_billboard = [
        elem[0] for elem in billboard_items_with_URIs
    ]
    try:
        assert len(artist_URIs) > 0
    except AssertionError:
        logging.exception('The artist list is empty.')

    # Keep an eye on processing time.
    t0 = time.time()
    n_tracks_processed = 0

    # Fetch data for each artist in our list.
    # Monitor time it takes per artist.
    for artist_URI in artist_URIs:
        # Authentication tokens expire after 1 hour. This is a first-order
        # hack for that.
        if os.path.exists(os.path.join(args.path_data_set_base,
                                       'artist_URI' + '.p')):
            logging.info('Skipping {} bc already processed.'.format(artist_URI))
            continue

        # Subselect tracks that have previews.
        artist_tracks = get_tracks_with_previews(artist_URI, spotify)
        n_tracks_processed += len(artist_tracks)

        # Add some precomputed features by Spotify.
        artist_tracks = add_spotify_audio_features(artist_tracks, spotify)

        # Get mp3s from the web and save them.
        artist_tracks = fetch_mp3_files(artist_tracks, args.path_data_mp3)

        # Compute the spectrograms.
        artist_tracks = compute_spectrograms(artist_tracks)

        # Get flags for Tracks that appear on billboard. Use these for 
        # logistic regression.
        artist_tracks = flag_billboard_entries(artist_tracks,
                                               track_URIs_in_billboard)

        # Cache our results.
        cache_artist_data(artist_tracks, args.path_data_set_base)

    # Timing per track processed
    elapsed_per_track = (time.time() - t0) / float(n_tracks_processed)
    logging.info(
        'Processing {:d} tracks took {:.2f} seconds per track.'.format(
            n_tracks_processed,
            elapsed_per_track
        )
    )


if __name__ == '__main__':
    main(args)
