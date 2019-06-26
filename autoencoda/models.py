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


def LSTM_keras(X, Y,
               loss_type='binary_crossentropy',
               N=32,
               optimizer=k.optimizers.Adam(lr=0.01),
               do_dropout=None,
               list_metrics=['accuracy']):
    """Baseline LSTM model for spectrogram classification.
    """
    # Data set dimensions
    _, n_timesteps, n_features = X.shape

    # Expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(N, input_shape=(n_timesteps, n_features), return_sequences=True))
    model.add(LSTM(N))
    if do_dropout is not None:
        model.add(Dropout(do_dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=loss_type,
                  optimizer=optimizer,
                  metrics=list_metrics)
    return model


def deep_logistic_keras(X_trn, Y_trn,
                        nodes_per_layer=[50, 20, 1],
                        loss_type='binary_crossentropy',
                        optimizer=k.optimizers.Adam(lr=0.001),
                        list_metrics=['accuracy'],
                        do_batch_norm=True,
                        do_dropout=None,
                        activation_type='relu'):
    """Builds a deep NN model to predict binary output.
    """
    # Initialize model
    model = Sequential()

    # Construct model layers
    N_layers = len(nodes_per_layer)

    # Construct all laters
    for ilayer in range(N_layers):
        nodes = nodes_per_layer[ilayer]
        last_layer = ilayer == (N_layers - 1)

        # Handles each kind of layer (input, output, hidden) appropriately
        if ilayer == 0:
            model.add(Dense(nodes, input_dim=X_trn.shape[1]))
        elif ilayer == N_layers - 1:
            assert nodes == 1, 'Output layer should have 1 node.'
            model.add(Dense(nodes, activation='sigmoid'))
        else:
            model.add(Dense(nodes))

        # Optional batch norm and dropout
        if not last_layer:
            if do_dropout is not None:
                assert do_dropout < 1.0 and do_dropout >= 0.0, \
                       'Dropout must be fraction between 0.0 and 1.0.'
                model.add(Dropout(do_dropout))

            # Optional batch norm
            if do_batch_norm:
                model.add(BatchNormalization())

            # Add activation Function
            model.add(Activation(activation_type))

    # Compile
    model.compile(loss=loss_type,
                  optimizer=optimizer,
                  metrics=list_metrics)

    return model


def logistic_regression_keras(X, Y,
                              loss_type='binary_crossentropy',
                              optimizer=k.optimizers.Adam(lr=0.01),
                              list_metrics=['accuracy'],
                              print_summary=False,
                              **kwargs):
    """Logistic regression baseline model
    """
    # Initialize model
    model = Sequential()
    model.add(Dense(1, input_dim=X.shape[1], activation='sigmoid'))
    # Compile and fit
    model.compile(loss=loss_type,
                  optimizer=optimizer,
                  metrics=list_metrics)
    # For debugging
    if print_summary:
        print('Logistic regression model summary:')
        print(model.summary())
    return model


def kfold_wrap(X, Y, model, args,
               k=10,
               batch=300,
               verbose=True,
               seed=1234,
               val_split=0.2):
    cv_scores = []
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    for trn, tst in kfold.split(X, Y):
        model.fit(X[trn], Y[trn],
                  validation_data=[X[tst], Y[tst]],
                  epochs=args.epochs,
                  batch_size=batch,
                  verbose=1,
                  callbacks=[TensorBoard(log_dir=args.tensor_board)]
        )
        model.save(args.path_save_model + 'model-latest.h5')
        scores_tst = model.evaluate(X[tst], Y[tst], verbose=0)
        scores_trn = model.evaluate(X[trn], Y[trn], verbose=0)
        assert model.metrics_names[1] == 'acc'
        cv_scores.append([scores_trn[1]*100, scores_tst[1]*100])
        logging.info('Trn acc: {:.2f}, Tst acc {:.2f}\n\n'.format(
            result[:,0].mean(), result[:,0].std(),
            result[:,1].mean(), result[:,1].std()
        ))
    return np.array(cv_scores)


def kfold_wrap_scikit(X, Y, model, args, k=10, batch=300, seed=1234):
    cv_scores = []
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for trn, tst in kfold.split(X, Y):
        model.fit(X[trn], Y[trn])
        cv_scores.append([model.score(X[trn], Y[trn]),
                          model.score(X[tst], Y[tst])])
    return np.array(cv_scores)


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
