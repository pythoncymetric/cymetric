""" 
A collection of tensorflow callbacks.
"""
import tensorflow as tf
import numpy as np
#import sys
from cymetric.models.measures import ricci_measure, sigma_measure, \
    kaehler_measure_loss, transition_measure_loss, ricci_scalar_fn
tfk = tf.keras

sigma_measure_tf = tf.function(func=sigma_measure)
kaehler_measure_tf = tf.function(func=kaehler_measure_loss)
transition_measure_tf = tf.function(func=transition_measure_loss)
ricci_measure_tf = tf.function(func=ricci_measure)
ricci_scalar_tf = tf.function(func=ricci_scalar_fn)

class AlphaCallback(tfk.callbacks.Callback):
    """Callback that allows to manipulate the alpha factors."""
    def __init__(self, scheduler):
        """A callback that manipulates the alpha factors.

        Args:
            scheduler (function): A function that returns a list of five
                tf.Variables and takes (int, dict, self.model.alpha) as args.
        """
        super(AlphaCallback, self).__init__()
        self.manipulater = scheduler

    def on_epoch_end(self, epoch, logs=None):
        r"""Manipulates alpha values according to function `scheduler`.

        Args:
            epoch (int): epoch
            logs (dict, optional): history.history. Defaults to None.
        """
        self.model.alpha = self.manipulater(epoch, logs, self.model.alpha)


class KaehlerCallback(tfk.callbacks.Callback):
    """Callback that tracks the weighted Kaehler measure."""
    def __init__(self, validation_data, nth=1, bSize=1000):
        r"""A callback which computes the kaehler measure for
        the validation data after every epoch end.

        See also: :py:func:`cymetric.models.measures.kaehler_measure_loss`.

        Args:
            validation_data (tuple(X_val, y_val)): Validation data.
            nth (int, optional): Run every n-th epoch. Defaults to 1.
            bSize (int, optional): Batch size. Defaults to 1000.
        """
        super(KaehlerCallback, self).__init__()
        self.X_val, self.y_val = validation_data
        self.weights = tf.cast(self.y_val[:, -2], tf.float32)
        self.omega = tf.cast(self.y_val[:, -1], tf.float32)
        self.nth = nth
        self.bSize = bSize

    def on_epoch_end(self, epoch, logs=None):
        r"""Computes kaehler measure.

        Args:
            epoch (int): epoch
            logs (dict, optional): history.history. Defaults to None.
        """
        # might have to batch this
        if epoch % self.nth == 0:
            n_p = len(self.X_val)
            # kaehler loss measure already takes the mean
            kaehler_losses = []
            for i in range(int(n_p/self.bSize)):
                s = i*self.bSize
                if i != int(n_p/self.bSize)-1:
                    e = (i+1)*self.bSize
                    kaehler_losses += [kaehler_measure_tf(
                        self.model, self.X_val[s:e])]
                else:
                    e = n_p
                    kaehler_losses += [kaehler_measure_tf(
                        self.model, self.X_val[s:e])]
                    # rescale last entry to give correct weight for mean
                    kaehler_losses[-1] *= (e-s)/self.bSize
            cb_res = np.mean(kaehler_losses).tolist()
            logs['kaehler_val'] = cb_res
            if cb_res <= 1e-3:
                print(' - Kaehler measure val:    {:.4e}'.format(cb_res))
            else:
                print(' - Kaehler measure val:    {:.4f}'.format(cb_res))

    def on_train_begin(self, logs=None):
        r"""Compute Kaehler measure before training as baseline.

        Args:
            logs (dict, optional): History. Defaults to None.
        """
        self.on_epoch_end(-1, logs=logs)


class RicciCallback(tfk.callbacks.Callback):
    """Callback that tracks the Ricci measure."""
    def __init__(self, validation_data, pullbacks, verbose=0, bSize=1000,
            nth=1, hlevel = 0):
        r"""A callback which computes the ricci measure for
        the validation data after every epoch end.

        See also: :py:func:`cymetric.models.measures.ricci_measure`, 
            :py:func:`cymetric.models.measures.ricci_scalar_fn`.

        .. math::

            ||R|| \equiv \frac{\text{Vol}_K^{\frac{1}{\text{nfold}}}}{\text{Vol}_{\text{CY}}}
                \int_X d\text{Vol}_K |R|

        Args:
            validation_data (tuple(X_val, y_val)): validation data
            pullbacks (tensor[(n_p, nfold, n_coord)]): pullback tensors
            verbose (int, optional): verbosity if >0 prints some info.
                Defaults to 0.
            bSize (int, optional): Batch size. Defaults to 1000.
            nth (int, optional): Run every n-th epoch. Defaults to 1.
            hlevel (int, optional): if > 0 adds increasingly more statistics.
                Defaults to 0.
        """
        super(RicciCallback, self).__init__()
        self.X_val, self.y_val = validation_data
        self.weights = tf.cast(self.y_val[:, -2], tf.float32)
        self.vol_cy = tf.math.reduce_mean(self.weights, axis=-1)
        self.omega = tf.cast(self.y_val[:, -1], tf.float32)
        self.pullbacks = tf.cast(pullbacks, tf.complex64)
        self.verbose = verbose
        self.hlevel = hlevel
        self.nth = nth
        self.bSize = bSize

    def on_epoch_end(self, epoch, logs=None):
        r"""Computes ricci measure.

        Args:
            epoch (int): epoch
            logs (dict, optional): history.history. Defaults to None.
        """
        if epoch % self.nth == 0:
            n_p = len(self.X_val)
            nfold = tf.cast(self.model.nfold, dtype=tf.float32)
            ricci_scalars = np.zeros(n_p)
            dets = np.zeros(n_p)
            for i in range(int(n_p/self.bSize)):
                s = i*self.bSize
                e = (i+1)*self.bSize if i != int(n_p/self.bSize)-1 else n_p
                ricci_scalars[s:e], dets[s:e] = ricci_scalar_tf(self.model,
                    self.X_val[s:e], pullbacks=self.pullbacks[s:e],
                    verbose=self.verbose, rdet=True)
            ricci_scalars = tf.math.abs(ricci_scalars)
            det_over_omega = dets / self.omega
            ricci_scalars = tf.cast(ricci_scalars, dtype=tf.float32)
            det_over_omega = tf.cast(det_over_omega, dtype=tf.float32)
            vol_k = tf.math.reduce_mean(det_over_omega * self.weights, axis=-1)
            ricci = (vol_k**(1/nfold) / self.vol_cy) * tf.math.reduce_mean(
                det_over_omega * ricci_scalars * self.weights, axis=-1)
            cb_res = ricci.numpy().tolist()
            logs['ricci_val'] = cb_res
            if self.hlevel > 0:
                logs['ricci_val_mean'] = float(np.mean(ricci_scalars))
                if self.hlevel > 1:
                    logs['ricci_val_median'] = float(np.median(ricci_scalars))
                    logs['ricci_val_var'] = float(np.var(ricci_scalars))
                    logs['ricci_val_std'] = float(np.std(ricci_scalars))
                    if self.hlevel > 2:
                        logs['ricci_val_dets'] = float(np.sum(dets<0)/len(dets))
            if cb_res <= 1e-3:
                print(' - Ricci measure val:      {:.4e}'.format(cb_res))
            else:
                print(' - Ricci measure val:      {:.4f}'.format(cb_res))

    def on_train_begin(self, logs=None):
        r"""Compute Ricci measure before training as baseline.

        Args:
            logs (dict, optional): History. Defaults to None.
        """
        self.on_epoch_end(-1, logs=logs)


class SigmaCallback(tfk.callbacks.Callback):
    """Callback that tracks the sigma measure."""
    def __init__(self, validation_data):
        r"""A callback which computes the sigma measure for
        the validation data after every epoch end.

        See also: :py:func:`cymetric.models.measures.sigma_measure`.

        .. math::

            \sigma_k \equiv \frac{1}{\text{Vol}_{\text{CY}}}
                \int_X d\text{Vol}_{\text{CY}} |1 - 
                \frac{\det(g)/\text{Vol}_K}{\Omega \wedge \bar\Omega / \text{CY}}|


        Args:
            validation_data (tuple(X_val, y_val)): validation data
        """
        super(SigmaCallback, self).__init__()
        self.X_val, self.y_val = validation_data
        self.X_val = tf.cast(self.X_val, tf.float32)
        self.y_val = tf.cast(self.y_val, tf.float32)
        #self.weights = tf.cast(self.y_val[:, -2], tf.float32)
        #self.omega = tf.cast(self.y_val[:, -1], tf.float32)

    def on_epoch_end(self, epoch, logs=None):
        r"""Computes sigma measure.

        Args:
            epoch (int): epoch
            logs (dict, optional): history.history. Defaults to None.
        """
        sigma = sigma_measure_tf(self.model, self.X_val, self.y_val)

        cb_res = sigma.numpy().tolist()
        logs['sigma_val'] = cb_res
        if cb_res <= 1e-3:
            print(' - Sigma measure val:      {:.4e}'.format(cb_res))
        else:
            print(' - Sigma measure val:      {:.4f}'.format(cb_res))

    def on_train_begin(self, logs=None):
        r"""Compute sigma measure before training as baseline.

        Args:
            logs (dict, optional): History. Defaults to None.
        """
        self.on_epoch_end(-1, logs=logs)


class TransitionCallback(tfk.callbacks.Callback):
    """Callback that tracks the transition loss weighted over the CY."""
    def __init__(self, validation_data):
        r"""A callback which computes the transition measure for
        the validation data after every epoch end.

        Args:
            validation_data (tuple(X_val, y_val)): validation data
        """
        super(TransitionCallback, self).__init__()
        self.X_val, self.y_val = validation_data
        self.X_val = tf.cast(self.X_val, tf.float32)
        
    def on_epoch_end(self, epoch, logs=None):
        r"""Computes transition measure.

        Args:
            epoch (int): epoch
            logs (dict, optional): history.history. Defaults to None.
        """
        transition = transition_measure_tf(self.model, self.X_val)

        cb_res = transition.numpy().tolist()
        logs['transition_val'] = cb_res
        if cb_res <= 1e-3:
            print(' - Transition measure val: {:.4e}'.format(cb_res))
        else:
            print(' - Transition measure val: {:.4f}'.format(cb_res))
    
    def on_train_begin(self, logs=None):
        r"""Compute transition measure before training as baseline.

        Args:
            logs (dict, optional): History. Defaults to None.
        """
        self.on_epoch_end(-1, logs=logs)


class VolkCallback(tfk.callbacks.Callback):
    r"""Callback that computes the volume from the metric.
    Has to be equal to 1 in order to stay at the same point in Kaehler moduli
    space.
    """
    def __init__(self, validation_data, nfold=3):
        r"""A callback which computes Volk of the validation data
        after every epoch end.

        .. math::

            \text{Vol}_K = \int_X \omega^3

        Args:
            validation_data (tuple(X_val, y_val)): validation data
            nfold (int, optional): degree of CY. Defaults to 3.
        """
        super(VolkCallback, self).__init__()
        self.X_val, self.y_val = validation_data
        self.weights = tf.cast(self.y_val[:, -2], dtype=tf.float32)
        self.omega = tf.cast(self.y_val[:, -1], dtype=tf.float32)
        self.nfold = tf.cast(nfold, dtype=tf.float32)
        #NOTE: Check that convention is consistent with rest of code.
        #self.factor = tf.exp(tf.math.lgamma(self.nfold+1))/(2.**nfold)
        self.factor = float(1.)

    def on_epoch_end(self, epoch, logs=None):
        r"""Tracks Volk during the training process.

        Args:
            epoch (int): epoch
            logs (dict, optional): history.history. Defaults to None.
        """
        prediction = self.model(self.X_val)
        volk = self.compute_volk(prediction, self.weights,
                                 self.omega, self.factor)

        cb_res = volk.numpy().tolist()
        logs['volk_val'] = cb_res
        if cb_res <= 1e-3:
            print(' - Volk val:               {:.4e}'.format(cb_res))
        else:
            print(' - Volk val:               {:.4f}'.format(cb_res))

    def on_train_begin(self, logs=None):
        r"""Compute Volk loss before training as baseline.

        Args:
            logs (dict, optional): History. Defaults to None.
        """
        self.on_epoch_end(-1, logs=logs)

    @tf.function
    def compute_volk(self, pred, weights, omega, factor):
        r"""Vol k integrated over all points.

        .. math::

            \text{Vol}_K = \int_X \omega^3 
                = \frac{1}{N} \sum_p \frac{\det(g)}{\Omega \wedge \bar\Omega} w

        Note:
            This is different than the Volk-loss. It should integrate to 1.

        Args:
            pred (tf.tensor([n_p, nfold, nfold], tf.complex)): 
                Metric prediction.
            weights (tf.tensor([n_p], tf.float)): Integration weights.
            omega (tf.tensor([n_p], tf.float)): 
                :math:`\Omega \wedge \bar\Omega`.
            factor (tf.float): Additional prefactors due to conventions.

        Returns:
            tf.tensor([n_p], tf.float): Vol k.
        """
        det = tf.math.real(tf.linalg.det(pred)) * factor
        volk_pred = tf.math.reduce_mean(det * weights / omega, axis=-1)
        return volk_pred
