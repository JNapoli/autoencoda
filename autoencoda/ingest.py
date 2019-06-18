import argparse
import librosa
import logging
import os
import pickle
import random
import spotipy
import sys
import time
import wget

import numpy as np
import spotipy.util as su


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
    """Get Spotify URIs, given a track and artist name. This function
    searches Spotify for available URI data.

    Args:
        bb_track (str): Track name
        bb_artist (str): Track artist
        spotify (Spotify): Spotify instance to query

    Returns:
        URIs (tuple): Tuple whose first element is the track URI and whose second
                      element is the artist URI. Either or both can be None.
    """
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


def add_spotify_audio_features(track_list, spotify):
    """Get precomputed audio features from Spotify.
    """
    for track in track_list:
        # Get precomputed audio features.
        audio_features = spotify.audio_features(track['track_id'])[0]
        # Add numerical features to the data.
        for feature, value in audio_features.items():
            if not feature in track:
                track[feature] = value
        time.sleep(1.0)
        track['popularity'] = spotify.track(track['track_id'])['popularity']
    return track_list


def compute_spectrogram(track, **kwargs_spec):
    """Obtain a mel spectrogram from the raw track audio.
    """
    audio, sr = librosa.load(track['path_mp3'])
    sg = librosa.feature.melspectrogram(y=audio, sr=sr, **kwargs_spec)
    track['spectrogram'] = sg
    track['sr'] = sr
    return track


def has_mp3_preview(track_URI, spotify):
    """Function to return whether the requested track has an mp3 preview available.
    """
    return spotify.track(track_URI)['preview_url'] is not None


def build_track(track_URI, artist_URI, spotify, path_data_mp3, billboard=False):
    """ Put together a dictionary containing track information.
    """
    track = {
        'track_id': track_URI,
       'artist_id': artist_URI,
     'preview_url': spotify.track(track_URI)['preview_url']
    }
    assert track['preview_url'] is not None
    path_mp3 = os.path.join(path_data_mp3, track_URI + '.mp3')
    # Download and save path for mp3.
    wget.download(track['preview_url'], path_mp3)
    track['path_mp3'] = path_mp3
    track = compute_spectrogram(track)
    track['billboard'] = billboard
    return track


def main(args):
    # Set verbosity level for debugging.
    logging.basicConfig(filename='ingestion.log',
                        level=logging.DEBUG)

    # Get Spotify instance for querying Spotify API.
    logging.info('Initializing Spotify instance...')
    spotify = get_spotify_instance(args.spotify_client_id,
                                   args.spotify_client_secret)
    logging.info('Done.')

    # Option to continue.
    if args.ingest_billboard:

        # Load raw Billboard 100 data entries.
        with open(args.path_raw_dat_billboard, 'rb') as f:
            billboard_raw = pickle.load(f)

        # Randomize entries.
        billboard_raw = list(billboard_raw)
        np.random.shuffle(billboard_raw)
        URIs_billboard_tracks, URIs_billboard_artists = [], []

        # Obtain and store each track separately.
        for item in billboard_raw:
            try:
                # Verbosity
                logging.info('Processing {:s} by {:s}'.format(item[0], item[1]))

                # Get Spotify tags.
                bill_item = get_spotify_from_billboard(item[0],
                                                        item[1],
                                                        spotify)

                # Only process tracks with Spotify IDs and previews.
                if None in bill_item or not has_mp3_preview(bill_item[0], spotify):
                    continue

                # Spotify IDs
                URI_track, URI_artist = bill_item[0], bill_item[1]

                # Build track
                path_tracks_billboard = '../data/cache_tracks_billboard/'
                assert os.path.exists(path_tracks_billboard)
                path_track = os.path.join(path_tracks_billboard,
                                          URI_track + '.p')

                # Only build tracks we haven't built before.
                if os.path.exists(path_track): continue
                track_to_save = build_track(URI_track,
                                            URI_artist,
                                            spotify,
                                            args.path_data_mp3,
                                            billboard=True)

                # Cache track.
                with open(path_track, 'wb') as f:
                     pickle.dump(track_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)

                # Keep track of artists and tracks in Billboard set.
                URIs_billboard_tracks.append(URI_track)
                URIs_billboard_artists.append(URI_artist)

            # Sometimes connection or token exceptions happen.
            except spotipy.client.SpotifyException:
                logging.info('Caught expired token.')
                spotify = get_spotify_instance(args.spotify_client_id,
                                               args.spotify_client_secret)
                logging.info('Successfully reinstantiated.')
                continue

            except:
                logging.info('Caught some error.')
                spotify = get_spotify_instance(args.spotify_client_id,
                                               args.spotify_client_secret)
                logging.info('Successfully reinstantiated.')
                continue

        # Keep track of what URIs we have in our data set.
        with open('../data/cache/cache-URI-lists.p', 'wb') as f:
            pickle.dump((URIs_billboard_tracks, URIs_billboard_artists), f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    # Now get some tracks put out by the artists on Billboard that did not
    # make it to Billboard themselves.
    if args.ingest_non_hits:
        logging.info('Loading cached URI lists.')

        # Load which URIs we have processed from Billboard.
        with open('../data/cache/cache-URI-lists.p', 'rb') as f:
            URIs = pickle.load(f)
        URIs_billboard_tracks, URIs_billboard_artists = URIs
        # Sanity check
        assert 'track' in URIs_billboard_tracks[0]
        assert 'artist' in URIs_billboard_artists[0]

        artists_processed = []
        to_build = []

        for artist_URI in URIs_billboard_artists:
            if not artist_URI in artists_processed:
                try:
                    # Get top tracks for each Billboard artist.
                    top_tracks = spotify.artist_top_tracks(artist_URI)['tracks']
                    for t in top_tracks:
                        t_URI = 'spotify:track:{}'.format(t['id'])

                        # Now we are only interested in getting non-billboard hits
                        # that have mp3 previews.
                        if not t_URI in URIs_billboard_tracks and has_mp3_preview(t_URI, spotify):
                            track_to_save = build_track(t_URI,
                                                        artist_URI,
                                                        spotify,
                                                        args.path_data_mp3,
                                                        billboard=False)
                            path_tracks_not_billboard = '../data/cache_tracks_not_billboard/'
                            assert os.path.exists(path_tracks_not_billboard)
                            path_track = os.path.join(path_tracks_not_billboard,
                                                      t_URI + '.p')
                            # Cache track
                            with open(path_track, 'wb') as f:
                                pickle.dump(track_to_save, f,
                                            protocol=pickle.HIGHEST_PROTOCOL)

                except spotipy.client.SpotifyException:
                    logging.info("Caught expired token.")
                    spotify = get_spotify_instance(args.spotify_client_id,
                                                   args.spotify_client_secret)
                    logging.info("Successfully reinstantiated.")
                    continue

                except:
                    logging.info("Caught some error")
                    spotify = get_spotify_instance(args.spotify_client_id,
                                                   args.spotify_client_secret)
                    logging.info("Successfully reinstantiated.")
                    continue

if __name__ == '__main__':
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
    parser.add_argument('--ingest_billboard',
                        type=int,
                        default=0,
                        required=False,
                        help='Whether to ingest Billboard hot 100 entries.')
    parser.add_argument('--ingest_non_hits',
                        type=int,
                        default=0,
                        required=False,
                        help='Whether to ingest songs that did not appear on \
                             billboard.')
    args = parser.parse_args()
    main(args)