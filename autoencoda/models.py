import argparse
import logging
import os
import sys

import numpy as np
import os.path as path
import tensorflow.keras as k

from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.svm import SVC
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import Activation, BatchNormalization, \
                                    Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential, load_model


def LSTM_keras(X, Y,
               N=32,
               loss_type='binary_crossentropy',
               optimizer=k.optimizers.Adam(lr=0.001),
               do_dropout=None,
               metrics_list=['accuracy']):
    """Build an LSTM model in Keras.

    Args:
        X (np.ndarray): Array with shape [n_examples, n_timesteps, n_features]
                        containing data examples.
        Y (np.ndarray): Array with size [n_examples].
        N (int): Number of units for LSTM.
        loss_type (str): The loss function to minimize.
        optimizer (Keras optimizer): Keras optimizer with which to compile model.
        do_dropout (float/None): Dropout fraction to use.
        metrics_list (list of str): Metrics to calculate during training.

    Returns:
        model (Keras model): Compiled Keras model.
    """
    # Data set dimensions
    _, n_timesteps, n_features = X.shape

    # Expected input data shape: (batch_size, timesteps, data_dim)
    model = Sequential()
    model.add(LSTM(N, input_shape=(n_timesteps, n_features), return_sequences=True))
    model.add(LSTM(N))
    if do_dropout is not None:
        if not (do_dropout >= 0.0 and do_dropout < 1.0):
            raise ValueError('Dropout fraction specification is not correct.')
        model.add(Dropout(do_dropout))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss=loss_type,
                  optimizer=optimizer,
                  metrics=metrics_list)
    return model


def deep_logistic_keras(X,
                        nodes_per_layer=[50, 20, 1],
                        loss_type='binary_crossentropy',
                        optimizer=k.optimizers.Adam(lr=0.001),
                        metrics_list=['accuracy'],
                        do_batch_norm=True,
                        do_dropout=None,
                        activation_type='relu',
                        initializer=k.initializers.RandomNormal(mean=0.0, stddev=0.05)):
    """Build a deep NN classifier in Keras.

    Args:
        X (np.ndarray): Array with shape [n_examples, n_features]
                        containing data examples.
        nodes_per_layer (list of int): Number of nodes in each layer.
        loss_type (str): The loss function to minimize.
        optimizer (Keras optimizer): Keras optimizer with which to compile model.
        metrics_list (list of str): Metrics to calculate during training.
        do_batch_norm (bool): Whether to perform batch normalization after each
                              hidden layer.
        do_dropout (float/None): Dropout fraction to use.
        activation_type (str): Type of activation function to apply to hidden
                               layer outputs.
        initializer (Keras initializer): Keras initializer to use for dense layers.

    Returns:
        model (Keras model): Compiled Keras model.
    """
    # Initialize model
    model = Sequential()
    N_layers = len(nodes_per_layer)

    for ilayer in range(N_layers):
        nodes = nodes_per_layer[ilayer]
        last_layer = ilayer == (N_layers - 1)
        # Handles each kind of layer (input, output, hidden) appropriately
        if ilayer == 0:
            model.add(Dense(nodes,
                            input_dim=X.shape[1],
                            kernel_initializer=initializer))
        elif ilayer == N_layers - 1:
            assert nodes == 1, 'Output layer should have 1 node.'
            model.add(Dense(nodes,
                            activation='sigmoid',
                            kernel_initializer=initializer))
        else:
            model.add(Dense(nodes, kernel_initializer=initializer))
        # Optional batch norm and dropout
        if not last_layer:
            if do_dropout is not None:
                assert do_dropout < 1.0 and do_dropout >= 0.0, \
                       'Dropout must be fraction between 0.0 and 1.0.'
                model.add(Dropout(do_dropout))
            if do_batch_norm:
                model.add(BatchNormalization())
            # Add activation Function
            model.add(Activation(activation_type))
    # Compile
    model.compile(loss=loss_type,
                  optimizer=optimizer,
                  metrics=metrics_list)
    return model


def logistic_regression_keras(X,
                              loss_type='binary_crossentropy',
                              optimizer=k.optimizers.Adam(lr=0.001),
                              metrics_list=['accuracy'],
                              initializer=k.initializers.RandomNormal(mean=0.0, stddev=0.05)):
    """Logistic regression model built in Keras.

    Args:
        X (np.ndarray): Training data with shape [n_samples, n_features].
        loss_type (str): The loss function to minimize.
        optmizer (Keras optimizer): Keras optimizer with which to compile model.
        metrics_list (list of str): Metrics to calculate during training.
        initializer (Keras initializer): Keras initializer to use for dense layers.

    Returns:
        model (Keras model): Compiled Keras model.
    """
    # Build and compile model
    model = Sequential()
    model.add(Dense(1,
                    input_dim=X.shape[1],
                    kernel_initializer=initializer,
                    activation='sigmoid'))
    model.compile(loss=loss_type,
                  optimizer=optimizer,
                  metrics=metrics_list)
    return model


def reset_weights(model):
    """Reset the weights of a previously compiled keras model.

    Args:
        model (Keras model): A compiled keras model.

    Returns:
        None
    """
    session = k.backend.get_session()
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel.initializer.run(session=session)
    return None


def kfold_wrap(X, Y, model, args,
               k=10,
               batch=500,
               verbose=True,
               seed=1234):
    """Wrapper function that performs stratified k-fold cross-validation of the model.

    Args:
        X (np.ndarray): Array with shape [n_examples, n_features]
                        containing data examples.
        Y (np.ndarray): Array with size n_examples.
        model (Keras model): Keras model to use for cross-validation.
        args :
        k (int): Number of folds to use for cross-validation.
        batch (int): Number of examples to use per batch.
        verbose (bool): Verbosity level for training.
        seed (int): Number to use for random number seeding.

    Returns:
        cv_scores (np.ndarray): Numpy array of shape [k, 2] containing the training
                                and validation accuracies.
    """
    cv_scores = []
    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for ifold, (trn, tst) in enumerate(kfold.split(X, Y), start=1):
        reset_weights(model)
        model.fit(X[trn], Y[trn],
                  validation_data=[X[tst], Y[tst]],
                  epochs=args.epochs,
                  batch_size=batch,
                  verbose=1,
                  callbacks=[TensorBoard(log_dir=args.tensor_board)]
        )
        model_path = os.path.join(args.path_save_model,
                                  'model-fold-{:d}.h5'.format(ifold))
        model.save(model_path)
        scores_tst = model.evaluate(X[tst], Y[tst], verbose=0)
        scores_trn = model.evaluate(X[trn], Y[trn], verbose=0)
        assert model.metrics_names[1] == 'acc', \
               'Double check the metrics you are using. I expect "accuracy".'
        cv_scores.append([scores_trn[1]*100, scores_tst[1]*100])
        # Logging accuracies so we can monitor them while training.
        logging.info('Fold {:d}: Trn acc {:.2f}, Val acc {:.2f}\n\n'.format(
            ifold,
            np.array(cv_scores)[:,0].mean(),
            np.array(cv_scores)[:,1].mean()
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
    """Function to load data set.

    Args:
        path_bb_data (str): Path to Billboard data.
        path_not_bb_data (str): Path to not-Billboard data.

    Returns:
        (X, Y) (tuple): Data set. X has shape [n_samples, n_features], Y has
                        size [n_samples].
    """
    assert os.path.exists(path_bb_data) and os.path.exists(path_not_bb_data), \
           "The files you specified do not exist."
    X_1 = np.load(path_bb_data)
    X_0 = np.load(path_not_bb_data)
    X = np.concatenate((X_1, X_0))
    assert X.shape[0] == (X_1.shape[0] + X_0.shape[0]), \
           'np.concatenate is not doing the right thing.'
    Y = np.hstack((np.ones(X_1.shape[0]),
                   np.zeros(X_0.shape[0])))
    assert X.shape[0] == Y.size, \
           'Train and test sets should have same # of elements.'
    return X, Y


def log_data_summary(X, Y):
    """Sanity check that logs the number of each class.

    Args:
        X (np.ndarray): Data set
        Y (np.ndarray): Labels

    Returns:
        None
    """
    N_total = Y.size
    N_1 = Y.sum()
    N_0 = N_total - N_1
    logging.info("{:d} total examples. {:.1f} % of training set is 1.".format(
        N_total, 100 * float(N_1) / float(N_total)
    ))
    return None


def main(args):
    path_full_self = path.realpath(__file__)
    path_base_self = path.dirname(path_full_self)
    path_log = path.join(path_base_self,
                         '..',
                         'logs',
                         'models.log')
    logging.basicConfig(filename=path_log, level=logging.DEBUG)
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

    # Make sure storage location exists
    if not os.path.exists(args.path_save_model):
        os.mkdir(args.path_save_model)

    # Test models
    if args.do_logistic:
        model = logistic_regression_keras(X)
        path_checkpoint = os.path.join(args.path_save_model, 'best_model.h5')
        history = model.fit(X[trn], Y[trn],
                  epochs=args.epochs,
                  validation_data=[X[tst], Y[tst]],
                  batch_size=300,
                  verbose=1,
                  callbacks=[TensorBoard(log_dir=args.tensor_board),
                             EarlyStopping(monitor='val_acc',
                                           mode='max',
                                           verbose=1,
                                           patience=10000),
                             ModelCheckpoint(path_checkpoint,
                                             monitor='val_acc',
                                             mode='max',
                                             verbose=1, 
                                             save_best_only=True)]

        )
        # Keep paths tidy
        path_model = os.path.join(args.path_save_model, 'model-logistic.h5')
        path_trn_acc = os.path.join(args.path_save_model, 'train-acc-logistic.npy')
        path_val_acc = os.path.join(args.path_save_model, 'val-acc-logistic.npy')
        path_y_pred = os.path.join(args.path_save_model, 'logistic-Y-pred-for-ROC.npy')
        path_y_true = os.path.join(args.path_save_model, 'logistic-Y-for-ROC.npy')
        # Save model and data
        model.save(path_model)
        # Load best model
        model = load_model(path_checkpoint)
        y_pred = model.predict(X[tst]).flatten()
        train_acc = history.history['acc']
        val_acc = history.history['val_acc']
        np.save(path_trn_acc, train_acc)
        np.save(path_val_acc, val_acc)
        np.save(path_y_true, Y[tst])
        np.save(path_y_pred, y_pred)
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
        X = X[:, :200, :]
        model = LSTM_keras(X, Y, N=64)
        path_checkpoint = os.path.join(args.path_save_model, 'best_model.h5')
        history = model.fit(X[trn], Y[trn],
                  epochs=args.epochs,
                  validation_data=[X[tst], Y[tst]],
                  batch_size=300,
                  verbose=1,
                  callbacks=[TensorBoard(log_dir=args.tensor_board),
                             EarlyStopping(monitor='val_acc',
                                           mode='max',
                                           verbose=1,
                                           patience=100),
                             ModelCheckpoint(path_checkpoint,
                                             monitor='val_acc',
                                             mode='max',
                                             verbose=1, 
                                             save_best_only=True)]
        )
        # Keep paths tidy
        path_model = os.path.join(args.path_save_model, 'model-LSTM.h5')
        path_trn_acc = os.path.join(args.path_save_model, 'train-acc-LSTM.npy')
        path_val_acc = os.path.join(args.path_save_model, 'val-acc-LSTM.npy')
        path_y_pred = os.path.join(args.path_save_model, 'LSTM-Y-pred-for-ROC.npy')
        path_y_true = os.path.join(args.path_save_model, 'LSTM-Y-for-ROC.npy')
        # Save model and data
        model.save(path_model)
        # Load best model
        model = load_model(path_checkpoint)
        y_pred = model.predict(X[tst]).flatten()
        y_true = Y[tst]
        train_acc = history.history['acc']
        val_acc = history.history['val_acc']
        np.save(path_trn_acc, train_acc)
        np.save(path_val_acc, val_acc)
        np.save(path_y_pred,  y_pred)
        np.save(path_y_true,  y_true)
    if args.do_NN:
        if not args.explore_models:
            arch = [100, 50, 1]
            act = 'sigmoid'
            model = deep_logistic_keras(X[trn],
                nodes_per_layer=arch,
                do_dropout=args.fraction_dropout,
                activation_type=act
            )
            path_checkpoint = os.path.join(args.path_save_model, 'best_model.h5')
            history = model.fit(X[trn], Y[trn],
                      epochs=args.epochs,
                      validation_data=[X[tst], Y[tst]],
                      batch_size=300,
                      verbose=1,
                      callbacks=[TensorBoard(log_dir=args.tensor_board),
                                 EarlyStopping(monitor='val_acc',
                                               mode='max',
                                               verbose=1,
                                               patience=10000),
                                 ModelCheckpoint(path_checkpoint,
                                                 monitor='val_acc',
                                                 mode='max',
                                                 verbose=1, 
                                                 save_best_only=True)]
            )
            path_model = os.path.join(args.path_save_model,
                                      'model-NN.h5')
            model.save(path_model)
            logging.info('Saved model to {:s}'.format(path_model))
            # Load best model
            model = load_model(path_checkpoint)
            y_pred = model.predict(X[tst]).flatten()
            train_acc = history.history['acc']
            val_acc = history.history['val_acc']
            # Keep paths tidy
            path_train_acc = os.path.join(args.path_save_model,
                                          'train-acc-NN.npy')
            path_val_acc = os.path.join(args.path_save_model,
                                        'val-acc-NN.npy')
            path_y_pred = os.path.join(args.path_save_model,
                                       'NN-Y-pred-for-ROC.npy')
            path_y_true = os.path.join(args.path_save_model,
                                       'NN-Y-for-ROC.npy')
            # Save data
            np.save(path_train_acc, train_acc)
            np.save(path_val_acc, val_acc)
            np.save(path_y_true, Y[tst])
            np.save(path_y_pred, model.predict(X[tst]).flatten())
        else:
            # Build model
            for arch in [
                [100, 1],
                [50, 1]
            ]:
                for drop in [0.2]:
                    for act in ['sigmoid']:
                        model = deep_logistic_keras(X,
                                                    nodes_per_layer=arch,
                                                    do_dropout=drop,
                                                    activation_type=act
                        )
                        # Cross-validation
                        result = kfold_wrap(X, Y, model, args, k=5, seed=args.seed)
                        to_write = 'Arch [ ' + ', '.join(map(str, arch)) + \
                            '], Drop ' + str(drop) + ', Act ' + act + '\n'
                        path_results = os.path.join(args.path_save_model,
                                                    'kfold-results-NN.txt')
                        with open(path_results, 'a+') as f:
                            f.write(to_write)
                            f.write('Trn acc: {:.2f}, Trn stdev: {:.2f}, Val acc {:.2f}, Val stdev {:.2f}\n\n'.format(
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
                        required=True,
                        help='Path to preprocessed Billboard data.')
    parser.add_argument('--path_not_bb_data',
                        type=str,
                        required=True,
                        help='Path to preprocessed not-Billboard data.')
    parser.add_argument('--path_save_model',
                        type=str,
                        required=True,
                        help='Where to save the trained model and related files.')
    args = parser.parse_args()
    main(args)
