"""
Sigma loss function in tensorflow.
"""
import tensorflow as tf


def sigma_loss(kappa=1., nfold=3., flat=False):
    r"""MA loss.

    Args:
        kappa (float): inverse volume of the CY given by weights. Defaults to 1.
        nfold (float): dimension of the CY. Defaults to 3.
        flat (bool): True if metric is a flat tensor and has to be put into
            hermitian matrix first. Defaults to False

    Returns:
        function: MA loss function.
    """
    factorial = float(1.)
    nfold = tf.cast(nfold, dtype=tf.int32)
    kappa = tf.cast(kappa, dtype=tf.float32)
    det_factor = float(1.)

    def to_hermitian_vec(x):
        r"""Takes a tensor of length (-1,NFOLD**2) and transforms it
        into a (-1,NFOLD,NFOLD) hermitian matrix.

        Args:
            x (tensor[(-1,NFOLD**2), tf.float]): input tensor

        Returns:
            tensor[(-1,NFOLD,NFOLD), tf.float]: hermitian matrix
        """
        t1 = tf.reshape(tf.complex(x, tf.zeros_like(x)), (-1, nfold, nfold))
        up = tf.linalg.band_part(t1, 0, -1)
        low = tf.linalg.band_part(1j * t1, -1, 0)
        out = up + tf.transpose(up, perm=[0, 2, 1]) - tf.linalg.band_part(t1, 0, 0)
        return out + low + tf.transpose(low, perm=[0, 2, 1], conjugate=True)

    def sigma_integrand_loss_flat(y_true, y_pred):
        r"""Monge-Ampere integrand loss.

        l = |1 - det(g)/ (Omega \wedge \bar{Omega})|

        Args:
            y_true (tensor[(bsize, x), tf.float]): some tensor  
                        with last value being (Omega \wedge \bar{Omega})
            y_pred (tensor[(bsize, 9), tf.float]): NN prediction

        Returns:
            tensor[(bsize, 1), tf.float]: loss for each sample in batch
        """
        g = to_hermitian_vec(y_pred)
        # older tensorflow versions require shape(y_pred) == shape(y_true)
        # then just give it some tensor where omega is the last value.
        omega_squared = y_true[:, -1]
        det = tf.math.real(tf.linalg.det(g))*factorial/det_factor
        return tf.abs(tf.ones(tf.shape(omega_squared), dtype=tf.float32) -
                      det/omega_squared/kappa)

    def sigma_integrand_loss(y_true, y_pred):
        r"""Monge-Ampere integrand loss.

        l = |1 - det(g)/ (Omega \wedge \bar{Omega})|

        Args:
            y_true (tensor[(bsize, x), tf.float]): some tensor  
                        with last value being (Omega \wedge \bar{Omega})
            y_pred (tensor[(bsize, 3, 3), tf.float]): NN prediction

        Returns:
            tensor[(bsize, 1), tf.float]: loss for each sample in batch
        """
        omega_squared = y_true[:, -1]
        det = tf.math.real(tf.linalg.det(y_pred))*factorial/det_factor
        return tf.abs(tf.ones(tf.shape(omega_squared), dtype=tf.float32) -
                      det/omega_squared/kappa)

    if flat:
        return sigma_integrand_loss_flat
    else:
        return sigma_integrand_loss
