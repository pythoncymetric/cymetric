""" 
A collection of various helper functions.
"""
import tensorflow as tf


def prepare_tf_basis(basis, dtype=tf.complex64):
    new_basis = {}
    for key in basis:
        new_basis[key] = tf.cast(basis[key], dtype=dtype)
    return new_basis
