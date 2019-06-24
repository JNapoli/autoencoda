import argparse
import os

def main(args):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Predict the popularity of your track!'
    )
    parser.add_argument('--path_track',
                        type=str,
                        required=True,
                        help='mp3 file containing your music.')
    args = parser.parse_args()
    main(args)
