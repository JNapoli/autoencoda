import argparse
import billboard
import logging
import pickle
import requests
import time


def get_hot_100_set(stop_date, chart):
    """Get list of (title, artist) tuples for entries on the Billboard
       Hot 100 chart from stop_date - present.

    Args:
        stop_date (str): Date at which to stop entry scraping. Date must be
                         provided in format YYYY-MM-DD.
        chart (ChartData): A billboard ChartData object containing the Hot 100
                           entries.

    Returns:
        appeared_hot_100 (list): List of tuples of artists/songs that appeared
                                 on the Hot 100 chart.
    """
    appeared_hot_100 = [
        (elem.title, elem.artist) for elem in chart
    ]
    while chart.previousDate > stop_date:
        logging.info('Date {:s}'.format(chart.previousDate))
        try:
            chart = billboard.ChartData('hot-100', date=chart.previousDate)
            appeared_hot_100.extend([
                (elem.title, elem.artist) for elem in chart
            ])
        except requests.exceptions.ReadTimeout:
            logging.info('Logging a caught timeout exception and continuing.')
    return appeared_hot_100


def main(args):
    logging.basicConfig(filename='billboard.log',
                        level=logging.DEBUG)
    t0 = time.time()
    appeared_hot_100 = get_hot_100_set(args.end_date,
                                       billboard.ChartData('hot-100'))
    set_appeared_hot_100 = set(appeared_hot_100)
    elapsed = (time.time() - t0) / 60.0
    logging.info("Processing took {:.2f} minutes for ending \
                  date {:s}.".format(elapsed, args.end_date))
    with open(args.path_save, 'wb') as f:
        pickle.dump(set_appeared_hot_100, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Use Billboard python package to get tracks that appear on \
                     the Hot 100 list for a specific date range.'
    )
    parser.add_argument('--end_date',
                        type=str,
                        required=False,
                        default='2019-01-01',
                        help='Date format: YYYY-MM-DD')
    parser.add_argument('--path_save',
                        type=str,
                        required=False,
                        default='../data/raw/billboard-scrape.p',
                        help='File path for saving Hot 100 set.')
    args = parser.parse_args()
    main(args)
