import argparse
import logging
import sys

import numpy as np
import tensorflow.keras as k

from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
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


def load_data(path_bb_data, path_not_bb_data):
    X_1 = np.load(path_bb_data)
    X_0 = np.load(path_not_bb_data)
    X = np.concatenate((X_1, X_0))
    assert X.shape[0] == (X_1.shape[0] + X_0.shape[0]), \
           'Concatenate is not doing the right thing.'
    Y = np.hstack((np.ones(X_1.shape[0]),
                   np.zeros(X_0.shape[0])))
    assert X.shape[0] == Y.size, \
           'Train and test sets should have same # of elements.'
    return X, Y


def log_data_summary(X, Y):
    N_total = Y.size
    N_1 = Y.sum()
    N_0 = N_total - N_1
    logging.info("{:d} total examples. {:.1f} % of training set is 1.".format(
        N_total, 100 * float(N_1) / float(N_total)
    ))
    return None


def main(args):
    logging.basicConfig(filename='models.log', level=logging.DEBUG)
    np.random.seed(args.seed)

    # Get data
    X, Y = load_data(args.path_bb_data, args.path_not_bb_data)
    split = StratifiedShuffleSplit(n_splits=1,
                                   test_size=1-args.fraction_train,
                                   random_state=args.seed)
    splits = [(trn, tst) for trn, tst in split.split(X, Y)]
    trn, tst = splits[0][0], splits[0][1]

    # Print summary of data
    log_data_summary(X, Y)

    # Test models
    if args.do_logistic:
        model = logistic_regression_keras(X, Y)
        history = model.fit(X[trn], Y[trn],
                  epochs=args.epochs,
                  validation_data=[X[tst], Y[tst]],
                  batch_size=300,
                  verbose=1,
                  callbacks=[TensorBoard(log_dir=args.tensor_board)]
        )
        model.save(args.path_save_model + 'model-logistic.h5')
        y_pred = model.predict_classes(X[tst]).flatten()
        train_acc = history.history['acc']
        val_acc = history.history['val_acc']
        np.save(args.path_save_model + 'train_acc_logistic.npy', train_acc)
        np.save(args.path_save_model + 'val_acc_logistic.npy', val_acc)
        print('Logistic regression classification report:')
        print(classification_report(Y[tst], y_pred))
        logging.info('Saved model to disk.')
        np.save(args.path_save_model + 'logistic-Y-for-ROC.npy', Y[tst])
        np.save(args.path_save_model + 'logistic-Y-pred-for-ROC.npy',
                model.predict(X[tst]).flatten())
    if args.do_SVM:
        for c in [0.01, 0.1, 1.0, 10.0]:
            model = SVC(C=c, verbose=True)
            result = kfold_wrap_scikit(X, Y, model, args, k=10, seed=args.seed)
            with open('results-svm.txt', 'a+') as f:
                f.write('Trn acc: {:.2f}, Trn stdev: {:.2f}, Tst acc {:.2f}, Tst stdev {:.2f}\n\n'.format(
                    result[:,0].mean(), result[:,0].std(),
                    result[:,1].mean(), result[:,1].std()
                ))
    if args.do_LSTM:
        assert len(X.shape) == 3
        X = X[:, :500, :]
        model = LSTM_keras(X, Y)
        history = model.fit(X[trn], Y[trn],
                  epochs=args.epochs,
                  validation_data=[X[tst], Y[tst]],
                  batch_size=300,
                  verbose=1,
                  callbacks=[TensorBoard(log_dir=args.tensor_board)]
        )
        model.save(args.path_save_model + 'model-LSTM.h5')
        y_pred = model.predict_classes(X[tst]).flatten()
        train_acc = history.history['acc']
        val_acc = history.history['val_acc']
        np.save(args.path_save_model + 'train_acc_LSTM.npy', train_acc)
        np.save(args.path_save_model + 'val_acc_LSTM.npy', val_acc)
        print('LSTM classification report:')
        print(classification_report(Y[tst], y_pred))
        logging.info('Saved model to disk.')
        sys.exit()
        # checkpoint
        filepath = args.path_save_model + \
                   "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath,
            monitor='val_acc',
            verbose=1,
            save_best_only=True,
            mode='max'
        )
        callbacks_list = [checkpoint]
        # Fit the model
        model.fit(X, Y,
            validation_split=0.15,
            epochs=args.epochs,
            batch_size=100,
            callbacks=callbacks_list,
            verbose=1
        )
        model_json = model.to_json()
        with open(args.path_save_model + 'model-LSTM.json', 'w') as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights(args.path_save_model + 'model-LSTM.h5')
        logging.info('Saved LSTM model to disk')
    if args.do_NN:
        if not args.explore_models:
            #arch = [200, 100, 1]
            #arch = [100, 1]
            arch = [50, 1]
            #arch = [400, 200, 100, 1]
            act = 'sigmoid'
            model = deep_logistic_keras(X[trn], Y[trn],
                nodes_per_layer=arch,
                do_dropout=args.fraction_dropout,
                activation_type=act
            )
            history = model.fit(X[trn], Y[trn],
                      epochs=args.epochs,
                      validation_data=[X[tst], Y[tst]],
                      batch_size=300,
                      verbose=1,
                      callbacks=[TensorBoard(log_dir=args.tensor_board)]
            )
            model.save(args.path_save_model + 'model-NN.h5')
            y_pred = model.predict_classes(X[tst]).flatten()
            train_acc = history.history['acc']
            val_acc = history.history['val_acc']
            np.save(args.path_save_model + 'train_acc_NN.npy', train_acc)
            np.save(args.path_save_model + 'val_acc_NN.npy', val_acc)
            np.save(args.path_save_model + 'NN-Y-for-ROC.npy', Y[tst])
            np.save(args.path_save_model + 'NN-Y-pred-for-ROC.npy',
                    model.predict(X[tst]).flatten())
            print('NN classification report:')
            print(classification_report(Y[tst], y_pred))
            logging.info('Saved model to disk.')
        else:
            # Build model
            for arch in [
                [100, 1],
                [50, 1]
            ]:
                for drop in [0.2]:
                    for act in ['sigmoid']:
                        model = deep_logistic_keras(X, Y,
                            nodes_per_layer=arch,
                            do_dropout=drop,
                            activation_type=act
                        )
                        # K-fold cross-validation
                        result = kfold_wrap(X, Y, model, args, k=5, seed=args.seed)
                        print(result)
                        to_write = 'Arch [ ' + ', '.join(map(str, arch)) + '], Drop ' + \
                            str(drop) + ', Act ' + act + '\n'
                        with open('results.txt', 'a+') as f:
                            f.write(to_write)
                            f.write('Trn acc: {:.2f}, Trn stdev: {:.2f}, Tst acc {:.2f}, Tst stdev {:.2f}\n\n'.format(
                                result[:,0].mean(), result[:,0].std(),
                                result[:,1].mean(), result[:,1].std()
                            ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build and tune models.'
    )
    parser.add_argument('--fraction_train',
                        type=float,
                        required=False,
                        default=0.9,
                        help='Fraction of data to use for training.')
    parser.add_argument('--fraction_dropout',
                        type=float,
                        required=False,
                        default=0.3,
                        help='Fraction of nodes to drop for deep classifier.')
    parser.add_argument('--epochs',
                        type=int,
                        required=False,
                        default=2000,
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
    parser.add_argument('--do_LSTM',
                        type=int,
                        required=False,
                        default=0,
                        help='Fit LSTM')
    parser.add_argument('--do_NN',
                        type=int,
                        required=False,
                        default=0,
                        help='Fit NN')
    parser.add_argument('--do_logistic',
                        type=int,
                        required=False,
                        default=0,
                        help='Fit logistic regression model')
    parser.add_argument('--do_SVM',
                        type=int,
                        required=False,
                        default=0,
                        help='Fit SVM classification model')
    parser.add_argument('--explore_models',
                        type=int,
                        required=False,
                        default=0,
                        help='Whether to explore and cross validate models.')
    parser.add_argument('--path_bb_data',
                        type=str,
                        required=False,
                        default='../data/preprocessed/preprocessed-billboard-no-subtract-scaled.npy',
                        help='Path to preprocessed Billboard data.')
    parser.add_argument('--path_not_bb_data',
                        type=str,
                        required=False,
                        default='../data/preprocessed/preprocessed-not-billboard-no-subtract-scaled.npy',
                        help='Path to preprocessed not-Billboard data.')
    parser.add_argument('--path_save_model',
                        type=str,
                        required=False,
                        default='../models/',
                        help='Where to save the trained model.')
    args = parser.parse_args()
    main(args)
