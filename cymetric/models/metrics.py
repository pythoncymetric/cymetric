"""
A bunch of custom metrics for the custom model.
Need to be declared separately otherwise .fit 
throws an error, since they only take the loss values
and not y_pred, y_true as arguments.
"""

import tensorflow as tf
tfk = tf.keras


class SigmaLoss(tfk.metrics.Metric):
    def __init__(self, name='sigma_loss', **kwargs):
        super(SigmaLoss, self).__init__(name=name, **kwargs)
        self.sigma_loss = self.add_weight(name='sl', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, values, sample_weight=None):
        """
        Args:
            values: tupe (data['X_val'], data['y_val'])
            sample_weight: sample weights for the validation set (Default: None)

        Returns:

        """
        loss = values['sigma_loss']
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            loss = tf.multiply(loss, sample_weight)
        new_value = (tf.reduce_mean(loss, axis=-1) -
                     self.sigma_loss)/(self.count+1)
        self.sigma_loss.assign_add(new_value)
        self.count.assign_add(1)

    def result(self):
        return self.sigma_loss

    def reset_state(self):
        self.sigma_loss.assign(0)
        self.count.assign(0)


class KaehlerLoss(tfk.metrics.Metric):
    def __init__(self, name='kaehler_loss', **kwargs):
        super(KaehlerLoss, self).__init__(name=name, **kwargs)
        self.kaehler_loss = self.add_weight(name='kl', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, values, sample_weight=None):
        """
        Args:
            values: tupe (data['X_val'], data['y_val'])
            sample_weight: sample weights for the validation set (Default: None)

        Returns:

        """
        loss = values['kaehler_loss']
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            loss = tf.multiply(loss, sample_weight)
        new_value = (tf.reduce_mean(loss, axis=-1) - self.kaehler_loss)/(self.count+1)
        self.kaehler_loss.assign_add(new_value)
        self.count.assign_add(1)

    def result(self):
        return self.kaehler_loss

    def reset_state(self):
        self.kaehler_loss.assign(0)
        self.count.assign(0)


class TransitionLoss(tfk.metrics.Metric):
    def __init__(self, name='transition_loss', **kwargs):
        super(TransitionLoss, self).__init__(name=name, **kwargs)
        self.transition_loss = self.add_weight(name='tl', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, values, sample_weight=None):
        """
        Args:
            values: tupe (data['X_val'], data['y_val'])
            sample_weight: sample weights for the validation set (Default: None)

        Returns:

        """
        loss = values['transition_loss']
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            loss = tf.multiply(loss, sample_weight)
        new_value = (tf.reduce_mean(loss, axis=-1) - self.transition_loss)/(self.count+1)
        self.transition_loss.assign_add(new_value)
        self.count.assign_add(1)

    def result(self):
        return self.transition_loss

    def reset_state(self):
        self.transition_loss.assign(0)
        self.count.assign(0)


class RicciLoss(tfk.metrics.Metric):
    def __init__(self, name='ricci_loss', **kwargs):
        super(RicciLoss, self).__init__(name=name, **kwargs)
        self.ricci_loss = self.add_weight(name='rl', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, values, sample_weight=None):
        """
        Args:
            values: tupe (data['X_val'], data['y_val'])
            sample_weight: sample weights for the validation set (Default: None)

        Returns:

        """
        loss = values['ricci_loss']
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            loss = tf.multiply(loss, sample_weight)
        new_value = (tf.reduce_mean(loss, axis=-1) - self.ricci_loss)/(self.count+1)
        self.ricci_loss.assign_add(new_value)
        self.count.assign_add(1)

    def result(self):
        return self.ricci_loss

    def reset_state(self):
        self.ricci_loss.assign(0)
        self.count.assign(0)


class VolkLoss(tfk.metrics.Metric):
    def __init__(self, name='volk_loss', **kwargs):
        super(VolkLoss, self).__init__(name=name, **kwargs)
        self.volk_loss = self.add_weight(name='vk', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, values, sample_weight=None):
        """
        Args:
            values: tupe (data['X_val'], data['y_val'])
            sample_weight: sample weights for the validation set (Default: None)

        Returns:

        """
        loss = values['volk_loss']
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            loss = tf.multiply(loss, sample_weight)
        new_value = (tf.reduce_mean(loss, axis=-1) - self.volk_loss)/(self.count+1)
        self.volk_loss.assign_add(new_value)
        self.count.assign_add(1)

    def result(self):
        return self.volk_loss

    def reset_state(self):
        self.volk_loss.assign(0)
        self.count.assign(0)


class TotalLoss(tfk.metrics.Metric):
    def __init__(self, name='loss', **kwargs):
        super(TotalLoss, self).__init__(name=name, **kwargs)
        self.total_loss = self.add_weight(name='tl', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, values, sample_weight=None):
        """
        Args:
            values: tupe (data['X_val'], data['y_val'])
            sample_weight: sample weights for the validation set (Default: None)

        Returns:

        """
        loss = values['loss']
        # total loss already gets rescaled by sample weight
        # if sample_weight is not None:
        #    sample_weight = tf.cast(sample_weight, self.dtype)
        #    loss = tf.multiply(loss, sample_weight)
        new_value = (tf.reduce_mean(loss, axis=-1) - self.total_loss)/(self.count+1)
        self.total_loss.assign_add(new_value)
        self.count.assign_add(1)

    def result(self):
        return self.total_loss

    def reset_state(self):
        self.total_loss.assign(0)
        self.count.assign(0)
