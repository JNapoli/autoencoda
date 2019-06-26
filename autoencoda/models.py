import argparse
import logging
import sys

import numpy as np
import tensorflow.keras as k

from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, \
                                    train_test_split
from sklearn.svm import SVC
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Activation, BatchNormalization, \
                                    Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential


def LSTM_keras(X_trn, Y_trn,
               optimizer=k.optimizers.Adam(lr=0.001),
               list_metrics=['accuracy']):
    """Baseline LSTM model for spectrogram classification.
    """
    # Data set dimensions
    _, timesteps, data_dim = X_trn.shape

    # Expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(60))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='accuracy',
                  optimizer=optimizer,
                  metrics=list_metrics)
    return model


def deep_logistic_keras(X_trn, Y_trn,
                        nodes_per_layer=[50, 20, 1],
                        loss_type='binary_crossentropy',
                        opt_type='Adam',
                        list_metrics=['accuracy'],
                        do_batch_norm=True,
                        do_dropout=None,
                        activation='relu'):
    """Builds a deep NN model to predict binary output.
    """
    # Initialize model
    model = Sequential()
    # Construct model layers
    N_layers = len(nodes_per_layer)

    for ilayer in range(N_layers):
        nodes = nodes_per_layer[ilayer]

        # Handles each kind of layer (input, output, hidden) appropriately
        if ilayer == 0:
            model.add(Dense(nodes))
        elif ilayer == N_layers - 1:
            try:
                assert nodes == 1, 'Output layer should have 1 node.'
            except AssertionError:
                logging.error('Binary classification should have 1 output node.')
            model.add(Dense(nodes, activation='sigmoid'))
        else:
            model.add(Dense(nodes))

        # Activation
        if not ilayer == N_layers - 1:
            if do_dropout is not None:
                assert do_dropout < 1.0 and do_dropout > 0.0, 'Dropout must be \
                       fraction between 0.0 and 1.0.'
                model.add(Dropout(do_dropout))
            # Optional batch norm
            if do_batch_norm:
                model.add(BatchNormalization())
            model.add(Activation(activation))
    # Compile
    model.compile(loss=loss_type,
                  optimizer=opt_type,
                  metrics=['accuracy'])
    return model


def logistic_regression_keras(X_trn, Y_trn,
                              loss_type='binary_crossentropy',
                              opt_type='Adam',
                              list_metrics=['accuracy'],
                              print_summary=False,
                              **kwargs):
    """Logistic regression baseline model
    """
    # Initialize model
    model = Sequential()
    model.add(Dense(1, activation='sigmoid'))
    # Compile and fit
    model.compile(loss=loss_type,
                  optimizer=opt_type,
                  metrics=list_metrics)
    if print_summary:
        print('Logistic regression model summary:')
        print(model.summary())
    return model


def keras_fit_model_wrapper(X_trn, Y_trn, model, **kwargs):
    """Wrapper function that calls fit() and passes kwargs.
    """
    history = model.fit(X_trn, Y_trn, **kwargs)
    return model


def main(args):
    np.random.seed(args.seed)
    # Load data
    X_1 = np.load('../data/preprocessed/preprocessed-billboard.npy')
    np.random.shuffle(X_1)
    Y_1 = np.ones(X_1.shape[0])
    N_samples_max = X_1.shape[0]
    X_0 = np.load('../data/preprocessed/preprocessed-not-billboard.npy')
    Y_0 = np.zeros(X_0.shape[0])
    np.random.shuffle(X_0)
    X = np.vstack((X_1, X_0))
    Y = np.hstack((Y_1, Y_0)).astype(int)
    assert Y.size == X.shape[0], 'Train and test sets should have same # of elements.'

    # Get train and test sets
    X_trn, X_tst, Y_trn, Y_tst = train_test_split(X, Y,
                                                  train_size=args.train_test_split,
                                                  random_state=args.seed)
    logging.info('Training on {:d} samples, validate on {:d} samples.'.format(
                 X_trn.shape[0], X_tst.shape[0]
    ))
    # Train classifier
    model = deep_logistic_keras(X_trn, Y_trn,
                                nodes_per_layer=[50, 20, 10, 1],
                                do_dropout=args.fraction_dropout)
    model = keras_fit_model_wrapper(X_trn, Y_trn, model,
                                    validation_data=[X_tst, Y_tst],
                                    epochs=args.epochs,
                                    batch_size=100,
                                    verbose=True,
                                    callbacks=[TensorBoard(log_dir=args.tensor_board)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build and tune models.'
    )
    parser.add_argument('--train_test_split',
                        type=float,
                        required=False,
                        default=0.8,
                        help='Fraction of data to use for training.')
    parser.add_argument('--fraction_dropout',
                        type=float,
                        required=False,
                        default=0.3,
                        help='Fraction of nodes to drop for deep classifier.')
    parser.add_argument('--epochs',
                        type=int,
                        required=False,
                        default=3000,
                        help='Number of epochs for training.')
    parser.add_argument('--tensor_board',
                        type=str,
                        required=False,
                        default='/tmp/tensorboard/',
                        help='Scratch directory for Tensor Board.')
    parser.add_argument('--seed',
                        type=int,
                        required=False,
                        default=12345,
                        help='Seed for numpy RNG.')
    args = parser.parse_args()
    main(args)
