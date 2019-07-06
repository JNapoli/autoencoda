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

import preprocess

import numpy as np
import os.path as path
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


def compute_spectrogram(track, **kwargs_spec):
    """Obtain a mel spectrogram from the raw track audio.

    Args:
        track (dict): Dictionary whose attributes correspond to features associated
                      with the track.

    Returns:
        track (dict): Track with additional spectral features computed using
                      librosa.
    """
    audio, sr = librosa.load(track['path_mp3'])
    sg = librosa.feature.melspectrogram(y=audio, sr=sr, **kwargs_spec)
    track['spectrogram'] = sg
    track['sr'] = sr
    track['centroid'] = librosa.feature.spectral_centroid(y=audio, sr=sr)
    track['bandwidth'] = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    track['flatness'] = librosa.feature.spectral_flatness(y=audio)
    track['rolloff'] = librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=0.80)
    track['tonnetz'] = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sr)
    track['zero_cross'] = librosa.feature.zero_crossing_rate(audio)
    track['rms'] = librosa.feature.rms(y=audio)
    return track


def has_mp3_preview(track_URI, spotify):
    """Function to return whether the requested track has an mp3 preview available.

    Args:
        track_URI (str): Track URI for searching Spotify
        spotify (Spotify): Spotify instance to query

    Returns:
        (bool): Boolean whose value indicates whether an mp3 preview is
                available for the track on Spotify
    """
    return spotify.track(track_URI)['preview_url'] is not None


def build_track(track_URI, artist_URI, spotify, path_data_mp3, billboard=False):
    """ Put together a dictionary containing track information.
    """
    track_info_from_spotify = spotify.track(track_URI)
    track = {
        'track_id': track_URI,
       'artist_id': artist_URI,
     'preview_url': track_info_from_spotify['preview_url'],
     'popularity': track_info_from_spotify['popularity']
    }
    assert track['preview_url'] is not None, 'Track preview did not exist.'
    path_mp3 = os.path.join(path_data_mp3, track_URI + '.mp3')
    # Download and save path for mp3.
    wget.download(track['preview_url'], path_mp3)
    track['path_mp3'] = path_mp3
    track['billboard'] = billboard
    track = compute_spectrogram(track)
    track = compute_chromogram(track)
    return track


def main(args):
    path_full_self = path.realpath(__file__)
    path_base_self = path.dirname(path_full_self)
    path_log = path.join(path_base_self,
                         '..',
                         'logs',
                         'ingestion.log')
    # Set verbosity level for debugging.
    logging.basicConfig(filename=path_log,
                        level=logging.DEBUG)

    # Get Spotify instance for querying Spotify API.
    logging.info('Initializing Spotify instance...')
    spotify = get_spotify_instance(args.spotify_client_id,
                                   args.spotify_client_secret)
    logging.info('Done.')

    # Create directory to store mp3s
    if not os.path.exists(args.path_data_mp3):
        os.mkdir(args.path_data_mp3)

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
                if not os.path.exists(args.path_cache_billboard):
                    os.mkdir(args.path_cache_billboard)
                path_track = os.path.join(args.path_cache_billboard, URI_track + '.p')

                # Keep track of artists and tracks in Billboard set.
                URIs_billboard_tracks.append(URI_track)
                URIs_billboard_artists.append(URI_artist)

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
        path_URI_list = os.path.join(path_base_self, '..', 'tmp', 'URIs-BB.p')
        with open(path_URI_list, 'wb') as f:
            pickle.dump((URIs_billboard_tracks, URIs_billboard_artists), f,
                        protocol=pickle.HIGHEST_PROTOCOL)

    # Now get some tracks put out by the artists on Billboard that did not
    # make it to Billboard themselves.
    if args.ingest_non_hits:
        logging.info('Loading cached URI lists.')

        # Load which URIs we have processed from Billboard.
        path_URI_list = os.path.join(path_base_self, '..', 'tmp', 'URIs-BB.p')
        with open(path_URI_list, 'rb') as f:
            URIs = pickle.load(f)
        URIs_billboard_tracks, URIs_billboard_artists = URIs
        # Sanity check
        assert 'track' in URIs_billboard_tracks[0], \
               "'Track' should be in the track URIs!"
        assert 'artist' in URIs_billboard_artists[0], \
               "'Artist' should be in the track URIs!"

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
                            if not os.path.exists(args.path_cache_not_billboard):
                                os.mkdir(args.path_cache_not_billboard)
                            path_track = os.path.join(args.path_cache_not_billboard,
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
    parser.add_argument('path_raw_dat_billboard',
                        type=str,
                        help='Path to file containing Billboard scrape result.')
    parser.add_argument('path_data_mp3',
                        type=str,
                        help='Directory in which to store mp3 files and other track \
                        data.')
    parser.add_argument('path_cache_billboard',
                        type=str,
                        help='Directory in which to store Billboard tracks.')
    parser.add_argument('path_cache_not_billboard',
                        type=str,
                        help='Directory in which to store not-Billboard tracks.')
    parser.add_argument('spotify_client_id',
                        type=str,
                        help='Required credential to access Spotify API.')
    parser.add_argument('spotify_client_secret',
                        type=str,
                        help='Required secret key to access Spotify API.')
    parser.add_argument('-ingest_billboard',
                        type=int,
                        default=0,
                        required=False,
                        help='Whether to ingest Billboard hot 100 entries.')
    parser.add_argument('-ingest_non_hits',
                        type=int,
                        default=0,
                        required=False,
                        help='Whether to ingest songs that did not appear on \
                             billboard.')
    args = parser.parse_args()
    main(args)
