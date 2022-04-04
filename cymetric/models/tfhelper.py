""" 
A collection of various helper functions.
"""
import tensorflow as tf


def prepare_tf_basis(basis, dtype=tf.complex64):
    new_basis = {}
    for key in basis:
        new_basis[key] = tf.cast(basis[key], dtype=dtype)
    return new_basis


def train_model(fsmodel, data, optimizer=None, epochs=50, batch_sizes=[64, 50000], verbose=1, custom_metrics=[], callbacks=[]):
    training_history = {}
    alpha0 = fsmodel.alpha[0]
    learn_kaehler = fsmodel.learn_kaehler
    learn_transition = fsmodel.learn_transition
    learn_ricci = fsmodel.learn_ricci
    learn_ricci_val = fsmodel.learn_ricci_val
    if optimizer is None:
        optimizer = tf.keras.optimizers.Adam()
    for epoch in range(epochs):
        batch_size = batch_sizes[0]
        fsmodel.learn_kaehler = learn_kaehler
        fsmodel.learn_transition = learn_transition
        fsmodel.learn_ricci = learn_ricci
        fsmodel.learn_ricci_val = learn_ricci_val
        fsmodel.learn_volk = tf.cast(False, dtype=tf.bool)
        fsmodel.alpha[0] = alpha0
        fsmodel.compile(custom_metrics=custom_metrics, optimizer=optimizer)
        if verbose > 0:
            print("Epoch {:2d}/{:d}".format(epoch + 1, epochs))
        history = fsmodel.fit(data['X_train'], data['y_train'], epochs=1, batch_size=batch_size, verbose=verbose, callbacks=callbacks)
        for k in history.history.keys():
            if "volk" in k:
                continue
            if k not in training_history.keys():
                training_history[k] = history.history[k]
            else:
                training_history[k] += history.history[k]
        batch_size = min(batch_sizes[1], len(data['X_train']))
        fsmodel.learn_kaehler = tf.cast(False, dtype=tf.bool)
        fsmodel.learn_transition = tf.cast(False, dtype=tf.bool)
        fsmodel.learn_ricci = tf.cast(False, dtype=tf.bool)
        fsmodel.learn_ricci_val = tf.cast(False, dtype=tf.bool)
        fsmodel.learn_volk = tf.cast(True, dtype=tf.bool)
        fsmodel.alpha[0] = tf.Variable(0., dtype=tf.float32)
        fsmodel.compile(custom_metrics=custom_metrics, optimizer=optimizer)
        history = fsmodel.fit(data['X_train'], data['y_train'], epochs=1, batch_size=batch_size, verbose=verbose, callbacks=callbacks)
        for k in history.history.keys():
            if "volk" not in k:
                continue
            if k not in training_history.keys():
                training_history[k] = history.history[k]
            else:
                training_history[k] += history.history[k]
    return fsmodel, training_history
    