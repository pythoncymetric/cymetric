""" 
A selection of custom tensorflow models for learning 
Calabi-Yau metrics using neural networks.
"""
import tensorflow as tf
from cymetric.models.losses import sigma_loss
from cymetric.models.fubinistudy import FSModel
from cymetric.pointgen.nphelper import get_all_patch_degrees, compute_all_w_of_x, get_levicivita_tensor
import numpy as np
tfk = tf.keras


class FreeModel(FSModel):
    r"""FreeModel from which all other models inherit.

    The training and validation steps are implemented in this class. All
    other computational routines are inherited from:
    cymetric.models.fubinistudy.FSModel
    
    Example:
        Assume that `BASIS` and `data` have been generated with a point 
        generator.

        >>> import tensorflow as tf
        >>> import numpy as np
        >>> from cymetric.models.tfmodels import FreeModel
        >>> from cymetric.models.tfhelper import prepare_tf_basis
        >>> tfk = tf.keras
        >>> data = np.load('dataset.npz')
        >>> BASIS = prepare_tf_basis(np.load('basis.pickle', allow_pickle=True))
    
        set up the nn and FreeModel

        >>> nfold = 3
        >>> ncoords = data['X_train'].shape[1]
        >>> nn = tfk.Sequential(
        ...     [   
        ...         tfk.layers.Input(shape=(ncoords)),
        ...         tfk.layers.Dense(64, activation="gelu"),
        ...         tfk.layers.Dense(nfold**2),
        ...     ]
        ... )
        >>> model = FreeModel(nn, BASIS)

        next we can compile and train

        >>> from cymetric.models.metrics import TotalLoss
        >>> metrics = [TotalLoss()]
        >>> opt = tfk.optimizers.Adam()
        >>> model.compile(custom_metrics = metrics, optimizer = opt)
        >>> model.fit(data['X_train'], data['y_train'], epochs=1)

        For other custom metrics and callbacks to be tracked, check
        :py:mod:`cymetric.models.metrics` and
        :py:mod:`cymetric.models.callbacks`.
    """
    def __init__(self, tfmodel, BASIS, alpha=None, **kwargs):
        r"""FreeModel is a tensorflow model predicting CY metrics. 
        
        The output is
            
            .. math:: g_{\text{out}} = g_{\text{NN}}
        
        a hermitian (nfold, nfold) tensor with each float directly predicted
        from the neural network.

        NOTE:
            * The model by default does not train against the ricci loss.
                
                To enable ricci training, set `self.learn_ricci = True`,
                **before** the tracing process. For validation data 
                `self.learn_ricci_val = True`,
                can be modified separately.

            * The models loss contributions are

                1. sigma_loss
                2. kaehler loss
                3. transition loss
                4. ricci loss (disabled)
                5. volk loss

            * The different losses are weighted with alpha.

            * The (FB-) norms for each loss are specified with the keyword-arg

                >>> model = FreeModel(nn, BASIS, norm = [1. for _ in range(5)])

            * Set kappa to the kappa value of your training data.

                >>> kappa = np.mean(data['y_train'][:,-2])

        Args:
            tfmodel (tfk.model): the underlying neural network.
            BASIS (dict): a dictionary containing all monomials and other
                relevant information from cymetric.pointgen.pointgen.
            alpha ([5//NLOSS], float): Weighting of each loss contribution.
                Defaults to None, which corresponds to equal weights.
        """
        super(FreeModel, self).__init__(BASIS=BASIS, **kwargs)
        self.model = tfmodel
        self.NLOSS = 5
        # variable or constant or just tensor?
        if alpha is not None:
            self.alpha = [tf.Variable(a, dtype=tf.float32) for a in alpha]
        else:
            self.alpha = [tf.Variable(1., dtype=tf.float32) for _ in range(self.NLOSS)]
        self.learn_kaehler = tf.cast(True, dtype=tf.bool)
        self.learn_transition = tf.cast(True, dtype=tf.bool)
        self.learn_ricci = tf.cast(False, dtype=tf.bool)
        self.learn_ricci_val = tf.cast(False, dtype=tf.bool)
        self.learn_volk = tf.cast(True, dtype=tf.bool)

        self.custom_metrics = None
        self.kappa = tf.cast(tf.math.real(BASIS['KAPPA']), dtype=tf.float32)
        self.gclipping = float(5.0)
        # add to compile?
        self.sigma_loss = sigma_loss(self.kappa, tf.cast(self.nfold, dtype=tf.float32))

    def call(self, input_tensor, training=True, j_elim=None):
        r"""Prediction of the NN.

        .. math:: g_{\text{out}} = g_{\text{NN}}

        The additional arguments are included for inheritance reasons.

        Args:
            input_tensor (tf.tensor([bSize, 2*ncoords], tf.float32)): Points.
            training (bool, optional): Defaults to True.
            j_elim (tf.tensor([bSize, nHyper], tf.int64), optional): 
                Coordinates(s) to be eliminated in the pullbacks.
                Not used in this model. Defaults to None.

        Returns:
            tf.tensor([bSize, nfold, nfold], tf.complex):
                Prediction at each point.
        """
        # nn prediction
        return self.to_hermitian(self.model(input_tensor, training=training))

    def compile(self, custom_metrics=None, **kwargs):
        r"""Compiles the model.

        kwargs takes any argument of regular `tf.model.compile()`

        Example:
            >>> model = FreeModel(nn, BASIS)
            >>> from cymetric.models.metrics import TotalLoss
            >>> metrics = [TotalLoss()]
            >>> opt = tfk.optimizers.Adam()
            >>> model.compile(custom_metrics = metrics, optimizer = opt)        

        Args:
            custom_metrics (list, optional): List of custom metrics.
                See also :py:mod:`cymetric.models.metrics`. If None, no metrics
                are tracked during training. Defaults to None.
        """
        super(FreeModel, self).compile(**kwargs)
        self.custom_metrics = custom_metrics

    @property
    def metrics(self):
        r"""Returns the models metrics including custom metrics.

        Returns:
            list: metrics
        """
        metrics = []
        if self._is_compiled:
            if self.compiled_loss is not None:
                metrics += self.compiled_loss.metrics
            if self.compiled_metrics is not None:
                metrics += self.compiled_metrics.metrics
            if self.custom_metrics is not None:
                metrics += self.custom_metrics

        for layer in self._flatten_layers():
            metrics.extend(layer._metrics)
        return metrics

    def train_step(self, data):
        r"""Train step of a single batch in model.fit().

        NOTE:
            1. The first epoch will take additional time, due to tracing.
            
            2. Warnings are plentiful. Disable on your own risk with 

                >>> tf.get_logger().setLevel('ERROR')
            
            3. The conditionals need to be set before tracing. 
            
            4. We employ under the hood gradient clipping.

        Args:
            data (tuple): test_data (x,y, sample_weight)

        Returns:
            dict: metrics
        """
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data

        with tf.GradientTape(persistent=False) as tape:
            trainable_vars = self.model.trainable_variables
            tape.watch(trainable_vars)
            # add other loss contributions.
            y_pred = self(x)
            if self.learn_kaehler:
                cijk_loss = self.compute_kaehler_loss(x)
            else:
                cijk_loss = tf.zeros_like(x[:, 0])
                # cijk_loss = tf.zeros(y.shape[-1], dtype=tf.float32)

            if self.learn_transition:
                t_loss = self.compute_transition_loss(x)
            else:
                t_loss = tf.zeros_like(cijk_loss)

            if self.learn_ricci:
                r_loss = self.compute_ricci_loss(x)
            else:
                r_loss = tf.zeros_like(cijk_loss)
                
            if self.learn_volk:
                # is scalar and not batch vector
                volk_loss = self.compute_volk_loss(x, y, y_pred)
            else:
                volk_loss = tf.zeros_like(cijk_loss)

            omega = tf.expand_dims(y[:, -1], -1)
            sigma_loss_cont = self.sigma_loss(omega, y_pred)**self.n[0]
            total_loss = self.alpha[0]*sigma_loss_cont +\
                self.alpha[1]*cijk_loss +\
                self.alpha[2]*t_loss +\
                self.alpha[3]*r_loss +\
                self.alpha[4]*volk_loss
            # weight the loss.
            if sample_weight is not None:
                total_loss *= sample_weight
        # Compute gradients
        gradients = tape.gradient(total_loss, trainable_vars)
        # remove nans and gradient clipping from transition loss.
        gradients = [tf.where(tf.math.is_nan(g), 1e-8, g) for g in gradients]
        gradients, _ = tf.clip_by_global_norm(gradients, self.gclipping)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Return metrics. NOTE: This interacts badly with any regular MSE
        # compiled loss. Make it so that only custom metrics are updated?
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        if self.custom_metrics is not None:
            loss_dict = {}
            loss_dict['loss'] = total_loss
            loss_dict['sigma_loss'] = sigma_loss_cont
            loss_dict['kaehler_loss'] = cijk_loss
            loss_dict['transition_loss'] = t_loss
            loss_dict['ricci_loss'] = r_loss
            loss_dict['volk_loss'] = volk_loss
            # add other loss?
            for m in self.custom_metrics:
                m.update_state(loss_dict, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        r"""Same as train_step without the outer gradient tape.
        Does *not* update the NN weights.

        NOTE:
            1. Computes the exaxt same losses as train_step
            
            2. Ricci loss val can be separately enabled with
                
                >>> model.learn_ricci_val = True
            
            3. Requires additional tracing.

        Args:
            data (tuple): test_data (x,y, sample_weight)

        Returns:
            dict: metrics
        """
        # unpack data
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            sample_weight = None
            x, y = data
        y_pred = self(x)
        # add loss contributions
        if self.learn_kaehler:
            cijk_loss = self.compute_kaehler_loss(x)
        else:
            cijk_loss = tf.zeros_like(x[:, 0])
            # cijk_loss = tf.zeros(y.shape[-1], dtype=tf.float32)
        if self.learn_transition:
            t_loss = self.compute_transition_loss(x)
        else:
            t_loss = tf.zeros_like(cijk_loss)
        if self.learn_ricci_val or self.learn_ricci:
            r_loss = self.compute_ricci_loss(x)
        else:
            r_loss = tf.zeros_like(cijk_loss)
        if self.learn_volk:
            volk_loss = self.compute_volk_loss(x, y, y_pred)
        else:
            volk_loss = tf.zeros_like(cijk_loss)
        
        omega = tf.expand_dims(y[:, -1], -1)
        sigma_loss_cont = self.sigma_loss(omega, y_pred)**self.n[0]
        total_loss = self.alpha[0]*sigma_loss_cont +\
            self.alpha[1]*cijk_loss +\
            self.alpha[2]*t_loss +\
            self.alpha[3]*r_loss +\
            self.alpha[4]*volk_loss
        # weight the loss.
        if sample_weight is not None:
            total_loss *= sample_weight
        # Return metrics.
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        if self.custom_metrics is not None:
            loss_dict = {}
            loss_dict['loss'] = total_loss
            loss_dict['sigma_loss'] = sigma_loss_cont
            loss_dict['kaehler_loss'] = cijk_loss
            loss_dict['transition_loss'] = t_loss
            loss_dict['ricci_loss'] = r_loss
            loss_dict['volk_loss'] = volk_loss
            # add other loss?
            for m in self.custom_metrics:
                m.update_state(loss_dict, sample_weight)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def to_hermitian(self, x):
        r"""Returns a hermitian tensor.
        
        Takes a tensor of length (-1,nfold**2) and transforms it
        into a (-1,nfold,nfold) hermitian matrix.

        Args:
            x (tensor[(-1,nfold**2), tf.float]): input tensor

        Returns:
            tensor[(-1,nfold,nfold), tf.float]: hermitian matrix
        """
        t1 = tf.reshape(tf.complex(x, tf.zeros_like(x)),
                        (-1, self.nfold, self.nfold))
        up = tf.linalg.band_part(t1, 0, -1)
        low = tf.linalg.band_part(1j * t1, -1, 0)
        out = up + tf.transpose(up, perm=[0, 2, 1]) - \
            tf.linalg.band_part(t1, 0, 0)
        return out + low + tf.transpose(low, perm=[0, 2, 1], conjugate=True)

    @tf.function
    def compute_volk_loss(self, input_tensor, wo, pred=None):
        r"""Computes volk loss.

        NOTE:
            This is an integral over the batch. Thus batch dependent.

        .. math::

            \mathcal{L}_{\text{vol}_k} = |\int_B g_{\text{FS}} -
                \int_B g_{\text{out}}|_n

        Args:
            input_tensor (tf.tensor([bSize, 2*ncoords], tf.float32)): Points.
            weights (tf.tensor([bSize], tf.float32)): Integration weights.
            pred (tf.tensor([bSize, nfold, nfold], tf.complex64), optional):
                Prediction from `self(input_tensor)`.
                If None will be calculated. Defaults to None.

        Returns:
            tf.tensor([bSize], tf.float32): Volk loss.
        """
        if pred is None:
            pred = self(input_tensor)
        
        aux_weights = tf.cast(wo[:, 0] / wo[:, 1], dtype=tf.complex64)
        aux_weights = tf.repeat(tf.expand_dims(aux_weights, axis=0), repeats=[len(self.BASIS['KMODULI'])], axis=0)
        # pred = tf.repeat(tf.expand_dims(pred, axis=0), repeats=[len(self.BASIS['KMODULI'])], axis=0)
        # ks = tf.eye(len(self.BASIS['KMODULI']), dtype=tf.complex64)
        # ks = tf.repeat(tf.expand_dims(self.fubini_study_pb(input_tensor), axis=0), repeats=[len(self.BASIS['KMODULI'])], axis=0)
        # input_tensor = tf.repeat(tf.expand_dims(input_tensor, axis=0), repeats=[len(self.BASIS['KMODULI'])], axis=0)
        # print(input_tensor.shape, pred.shape, ks.shape)
        # actual_slopes = tf.vectorized_map(self._calculate_slope, [input_tensor, pred, ks])
        ks = tf.eye(len(self.BASIS['KMODULI']), dtype=tf.complex64)

        def body(input_tensor, pred, ks, actual_slopes):
            f_a = self.fubini_study_pb(input_tensor, ts=ks[len(actual_slopes)])
            res = tf.expand_dims(self._calculate_slope([pred, f_a]), axis=0)
            actual_slopes = tf.concat([actual_slopes, res], axis=0)
            return input_tensor, pred, ks, actual_slopes

        def condition(input_tensor, pred, ks, actual_slopes):
            return len(actual_slopes) < len(self.BASIS['KMODULI'])

        f_a = self.fubini_study_pb(input_tensor, ts=ks[0])
        actual_slopes = tf.expand_dims(self._calculate_slope([pred, f_a]), axis=0)
        if len(self.BASIS['KMODULI']) > 1:
            _, _, _, actual_slopes = tf.while_loop(condition, body, [input_tensor, pred, ks, actual_slopes], shape_invariants=[input_tensor.get_shape(), pred.get_shape(), ks.get_shape(), tf.TensorShape([None, actual_slopes.shape[-1]])])
        actual_slopes = tf.reduce_mean(aux_weights * actual_slopes, axis=-1)
        loss = tf.reduce_mean(tf.math.abs(actual_slopes - self.slopes)**self.n[4])
        
        # return tf.repeat(tf.expand_dims(loss, axis=0), repeats=[input_tensor.shape[0]], axis=0)
        return tf.repeat(tf.expand_dims(loss, axis=0), repeats=[len(wo)], axis=0)

    def save(self, filepath, **kwargs):
        r"""Saves the underlying neural network to filepath.

        NOTE: 
            Currently does not save the whole custom model.

        Args:
            filepath (str): filepath
        """
        # TODO: save graph? What about Optimizer?
        # https://www.tensorflow.org/guide/keras/save_and_serialize#custom_objects
        self.model.save(filepath=filepath, **kwargs)


class MultFSModel(FreeModel):
    r"""MultFSModel inherits from :py:class:`FreeModel`.

    Example:
        Is identical to :py:class:`FreeModel`. Replace the model accordingly.
    """
    def __init__(self, *args, **kwargs):
        r"""MultFSModel is a tensorflow model predicting CY metrics.
        
        The output of this model has the following Ansatz
        
        .. math:: g_{\text{out}} = g_{\text{FS}} (1 + g_{\text{NN}})
        
        with elementwise multiplication and returns a hermitian (nfold, nfold)
        tensor.
        """
        super(MultFSModel, self).__init__(*args, **kwargs)

    def call(self, input_tensor, training=True, j_elim=None):
        r"""Prediction of the model.

        .. math:: 
        
            g_{\text{out}; ij} = g_{\text{FS}; ij} (1_{ij} + g_{\text{NN}; ij})

        Args:
            input_tensor (tf.tensor([bSize, 2*ncoords], tf.float32)): Points.
            training (bool, optional): Not used. Defaults to True.
            j_elim (tf.tensor([bSize, nHyper], tf.int64), optional): 
                Coordinates(s) to be eliminated in the pullbacks.
                If None will take max(dQ/dz). Defaults to None.

        Returns:
            tf.tensor([bSize, nfold, nfold], tf.complex):
                Prediction at each point.
        """
        # nn prediction
        nn_cont = self.to_hermitian(self.model(input_tensor, training=training))
        # fs metric
        fs_cont = self.fubini_study_pb(input_tensor, j_elim=j_elim)
        # return g_fs ( 1+ g_NN)
        return fs_cont + tf.math.multiply(fs_cont, nn_cont)


class MatrixFSModel(FreeModel):
    r"""MatrixFSModel inherits from :py:class:`FreeModel`.

    Example:
        Is identical to :py:class:`FreeModel`. Replace the model accordingly.
    """
    def __init__(self, *args, **kwargs):
        r"""MatrixFSModel is a tensorflow model predicting CY metrics.
        
        The output of this model has the following Ansatz
        
        .. math:: g_{\text{out}} = g_{\text{FS}} (1 + g_{\text{NN}})
        
        with matrix multiplication and returns a hermitian (nfold, nfold)
        tensor.
        """
        super(MatrixFSModel, self).__init__(*args, **kwargs)

    def call(self, input_tensor, training=True, j_elim=None):
        r"""Prediction of the model.

        .. math:: 
        
            g_{\text{out}; ik} = g_{\text{FS}; ij} (1_{jk} + g_{\text{NN}; jk})

        Args:
            input_tensor (tf.tensor([bSize, 2*ncoords], tf.float32)): Points.
            training (bool, optional): Not used. Defaults to True.
            j_elim (tf.tensor([bSize, nHyper], tf.int64), optional): 
                Coordinates(s) to be eliminated in the pullbacks.
                If None will take max(dQ/dz). Defaults to None.

        Returns:
            tf.tensor([bSize, nfold, nfold], tf.complex):
                Prediction at each point.
        """
        nn_cont = self.to_hermitian(self.model(input_tensor, training=training))
        fs_cont = self.fubini_study_pb(input_tensor, j_elim=j_elim)
        return fs_cont + tf.linalg.matmul(fs_cont, nn_cont)


class AddFSModel(FreeModel):
    r"""AddFSModel inherits from :py:class:`FreeModel`.

    Example:
        Is identical to :py:class:`FreeModel`. Replace the model accordingly.
    """
    def __init__(self, *args, **kwargs):
        r"""AddFSModel is a tensorflow model predicting CY metrics.
        
        The output of this model has the following Ansatz
        
        .. math:: g_{\text{out}} = g_{\text{FS}} + g_{\text{NN}}
        
        and returns a hermitian (nfold, nfold)tensor.
        """
        super(AddFSModel, self).__init__(*args, **kwargs)

    def call(self, input_tensor, training=True, j_elim=None):
        r"""Prediction of the model.

        .. math:: g_{\text{out}; ij} = g_{\text{FS}; ij}  + g_{\text{NN}; ij}

        Args:
            input_tensor (tf.tensor([bSize, 2*ncoords], tf.float32)): Points.
            training (bool, optional): Not used. Defaults to True.
            j_elim (tf.tensor([bSize, nHyper], tf.int64), optional): 
                Coordinates(s) to be eliminated in the pullbacks.
                If None will take max(dQ/dz). Defaults to None.

        Returns:
            tf.tensor([bSize, nfold, nfold], tf.complex64):
                Prediction at each point.
        """
        nn_cont = self.to_hermitian(self.model(input_tensor, training=training))
        fs_cont = self.fubini_study_pb(input_tensor, j_elim=j_elim)
        return fs_cont + nn_cont


class PhiFSModel(FreeModel):
    r"""PhiFSModel inherits from :py:class:`FreeModel`.

    The PhiModel learns the scalar potential correction to some Kaehler metric
    to make it the Ricci-flat metric. The Kaehler metric is taken to be the 
    Fubini-Study metric.

    Example:
        Is similar to :py:class:`FreeModel`. Replace the nn accordingly.

        >>> nn = tfk.Sequential(
        ...     [   
        ...         tfk.layers.Input(shape=(ncoords)),
        ...         tfk.layers.Dense(64, activation="gelu"),
        ...         tfk.layers.Dense(1),
        ...     ]
        ... )
        >>> model = PhiFSModel(nn, BASIS)

    You have to use this model if you want to remain in the same Kaehler class
    specified by the Kaehler moduli.
    """
    def __init__(self, *args, **kwargs):
        r"""PhiFSModel is a tensorflow model predicting CY metrics.
        
        The output of this model has the following Ansatz
        
        .. math:: 
        
            g_{\text{out}} = g_{\text{FS}} + 
                \partial \bar{\partial} \phi_{\text{NN}}
        
        and returns a hermitian (nfold, nfold) tensor. The model is by
        defintion Kaehler and thus this loss contribution is by default
        disabled. For similar reasons the Volk loss is also disabled if
        the last layer does not contain a bias. Otherwise it is required
        for successful tracing.
        """
        super(PhiFSModel, self).__init__(*args, **kwargs)
        # automatic in Phi network
        self.learn_kaehler = tf.cast(False, dtype=tf.bool)

    def call(self, input_tensor, training=True, j_elim=None):
        r"""Prediction of the model.

        .. math::

            g_{\text{out}; ij} = g_{\text{FS}; ij} + \
                partial_i \bar{\partial}_j \phi_{\text{NN}}

        Args:
            input_tensor (tf.tensor([bSize, 2*ncoords], tf.float32)): Points.
            training (bool, optional): Not used. Defaults to True.
            j_elim (tf.tensor([bSize, nHyper], tf.int64), optional):
                Coordinates(s) to be eliminated in the pullbacks.
                If None will take max(dQ/dz). Defaults to None.

        Returns:
            tf.tensor([bSize, nfold, nfold], tf.complex64):
                Prediction at each point.
        """
        # nn prediction
        with tf.GradientTape(persistent=False) as tape1:
            tape1.watch(input_tensor)
            with tf.GradientTape(persistent=False) as tape2:
                tape2.watch(input_tensor)
                # Need to disable training here, because batch norm
                # and dropout mix the batches, such that batch_jacobian
                # is no longer reliable.
                phi = self.model(input_tensor, training=False)
            d_phi = tape2.gradient(phi, input_tensor)
        dd_phi = tape1.batch_jacobian(d_phi, input_tensor)
        dx_dx_phi, dx_dy_phi, dy_dx_phi, dy_dy_phi = \
            0.25*dd_phi[:, :self.ncoords, :self.ncoords], \
            0.25*dd_phi[:, :self.ncoords, self.ncoords:], \
            0.25*dd_phi[:, self.ncoords:, :self.ncoords], \
            0.25*dd_phi[:, self.ncoords:, self.ncoords:]
        dd_phi = tf.complex(dx_dx_phi + dy_dy_phi, dx_dy_phi - dy_dx_phi)
        pbs = self.pullbacks(input_tensor, j_elim=j_elim)
        dd_phi = tf.einsum('xai,xij,xbj->xab', pbs, dd_phi, tf.math.conj(pbs))

        # fs metric
        fs_cont = self.fubini_study_pb(input_tensor, pb=pbs, j_elim=j_elim)
        # return g_fs + \del\bar\del\phi
        return tf.math.add(fs_cont, dd_phi)
        
    def compute_transition_loss(self, points):
        r"""Computes transition loss at each point. In the case of the Phi model, we demand that \phi(\lambda^q_i z_i)=\phi(z_i)

        Args:
            points (tf.tensor([bSize, 2*ncoords], tf.float32)): Points.

        Returns:
            tf.tensor([bSize], tf.float32): Transition loss at each point.
        """
        inv_one_mask = self._get_inv_one_mask(points)
        patch_indices = tf.where(~inv_one_mask)[:, 1]
        patch_indices = tf.reshape(patch_indices, (-1, self.nProjective))
        current_patch_mask = self._indices_to_mask(patch_indices)
        fixed = self._find_max_dQ_coords(points)
        cpoints = tf.complex(points[:, :self.ncoords], points[:, self.ncoords:])
        if self.nhyper == 1:
            other_patches = tf.gather(self.fixed_patches, fixed)
        else:
            combined = tf.concat((fixed, patch_indices), axis=-1)
            other_patches = self._generate_patches_vec(combined)
        
        other_patches = tf.reshape(other_patches, (-1, self.nProjective))
        other_patch_mask = self._indices_to_mask(other_patches)
        # NOTE: This will include same to same patch transitions
        exp_points = tf.repeat(cpoints, self.nTransitions, axis=-2)
        patch_points = self._get_patch_coordinates(exp_points, tf.cast(other_patch_mask, dtype=tf.bool))
        real_patch_points = tf.concat((tf.math.real(patch_points), tf.math.imag(patch_points)), axis=-1)
        gj = self.model(real_patch_points, training=True)
        gi = tf.repeat(self.model(points), self.nTransitions, axis=0)
        all_t_loss = tf.math.abs(gi-gj)
        all_t_loss = tf.reshape(all_t_loss, (-1, self.nTransitions))
        all_t_loss = tf.math.reduce_sum(all_t_loss**self.n[2], axis=-1)
        return all_t_loss/(self.nTransitions*self.nfold**2)

    def get_kahler_potential(self, points):
        r"""Computes the Kahler potential.

        Args:
            points (tf.tensor([bSize, 2*ncoords], tf.float32)): Points.

        Returns:
            tf.tensor([bSize], tf.float32): Kahler potential.
        """
        if self.nProjective > 1:
            # we go through each ambient space factor and create the Kahler potential.
            cpoints = tf.complex(
                points[:, :self.degrees[0]],
                points[:, self.ncoords:self.ncoords+self.degrees[0]])
            k_fs = self._fubini_study_n_potentials(cpoints, t=self.BASIS['KMODULI'][0])
            for i in range(1, self.nProjective):
                s = tf.reduce_sum(self.degrees[:i])
                e = s + self.degrees[i]
                cpoints = tf.complex(points[:, s:e],
                                     points[:, self.ncoords+s:self.ncoords+e])
                k_fs_tmp = self._fubini_study_n_potentials(cpoints, t=self.BASIS['KMODULI'][i])
                k_fs += k_fs_tmp
        else:
            cpoints = tf.complex(
                points[:, :self.ncoords],
                points[:, self.ncoords:2*self.ncoords])
            k_fs = self._fubini_study_n_potentials(cpoints, t=self.BASIS['KMODULI'][0])

        k_fs += tf.reshape(self.model(points), [-1])
        return k_fs


class ToricModel(FreeModel):
    r"""ToricModel is the base class of toric CYs and inherits from
    :py:class:`FreeModel`.

    Example:
        Is similar to :py:class:`FreeModel` but requires additional toric_data.
        This one can be generated with :py:mod:`cymetric.sage.sagelib`.

        >>> #generate toric_data with sage_lib
        >>> import pickle
        >>> toric_data = pickle.load('toric_data.pickle')
        >>> model = ToricModel(nn, BASIS, toric_data=toric_data)

    ToricModel does **not** train the underlying neural network. Instead, it 
    always predicts a generalization of the kaehler metric for toric CYs.
    """
    def __init__(self, *args, **kwargs):
        r"""ToricModel is the equivalent to
        :py:class:~`cymetric.models.fubinistudy.FSModel`.

        It will not learn the Ricci-flat metric, but can be used as a baseline
        to compare the neural network against.

        NOTE:
            1. Requires nevertheless a nn in its (kw)args.

            2. Requires `toric_data = toric_data` in its kwargs.
        """
        if 'toric_data' in kwargs.keys():
            self.toric_data = kwargs['toric_data']
            del kwargs['toric_data']
        self.nfold = self.toric_data['dim_cy']
        self.sections = [tf.cast(m, dtype=tf.complex64) for m in self.toric_data['exps_sections']]
        self.patch_masks = np.array(self.toric_data['patch_masks'], dtype=bool)
        self.glsm_charges = np.array(self.toric_data["glsm_charges"])
        self.nPatches = len(self.patch_masks)
        self.nProjective = len(self.toric_data["glsm_charges"])
        super(ToricModel, self).__init__(*args, **kwargs)
        self.kmoduli = self.BASIS['KMODULI']
        self.lc = tf.convert_to_tensor(get_levicivita_tensor(self.nfold), dtype=tf.complex64)
        self.slopes = self._target_slopes()

    def call(self, input_tensor, training=True, j_elim=None):
        r"""Computes the equivalent of the pullbacked 
        Fubini-Study metric at each point in input_tensor.

        .. math:: J = t^\alpha J_\alpha

        Args:
            input_tensor (tf.tensor([bSize, 2*ncoords], tf.float32)): Points.
            training (bool, optional): Defaults to True.
            j_elim (tf.tensor([bSize, nHyper], tf.int64), optional): 
                Coordinates(s) to be eliminated in the pullbacks.
                If None will take max(dQ/dz). Defaults to None.

        Returns:
            tf.tensor([bSize, nfold, nfold], tf.complex):
                Prediction at each point.
        """
        # FS prediction
        return self.fubini_study_pb(input_tensor, j_elim=j_elim)

    def fubini_study_pb(self, points, pb=None, j_elim=None, ts=None):
        r"""Computes the pullbacked Fubini-Study metric.

        NOTE:
            The pb argument overwrites j_elim.

        .. math::

            g_{ij} = \frac{1}{\pi} J_i^a \bar{J}_j^b \partial_a 
                \bar{\partial}_b \ln |\vec{z}|^2


        Args:
            points (tf.tensor([bSize, 2*ncoords], tf.float32)): Points.
            pb (tf.tensor([bSize, nfold, ncoords], tf.float32)):
                Pullback at each point. Overwrite j_elim. Defaults to None.
            j_elim (tf.tensor([bSize], tf.int64)): index to be eliminated. 
                Coordinates(s) to be eliminated in the pullbacks.
                If None will take max(dQ/dz). Defaults to None.
            ts (tf.tensor([len(kmoduli)], tf.complex64)):
                Kahler parameters. Defaults to the ones specified at time of point generation

        Returns:
            tf.tensor([bSize, nfold, nfold], tf.complex64):
                FS-metric at each point.
        """
        if ts is None:
            ts = self.BASIS['KMODULI']
        # NOTE: Cannot use super since for toric models we have only one toric space, but more than one Kahler modulus
        pullbacks = self.pullbacks(points, j_elim=j_elim) if pb is None else pb
        cpoints = tf.complex(points[:, :self.ncoords], points[:, self.ncoords:])

        Js = self._fubini_study_n_metrics(cpoints, n=0, t=ts[0])
        if len(self.kmoduli) != 1:
            for i in range(1, len(self.kmoduli)):
                Js += self._fubini_study_n_metrics(cpoints, n=i, t=ts[i])

        gFSpb = tf.einsum('xai,xij,xbj->xab', pullbacks, Js, tf.math.conj(pullbacks))
        return gFSpb

    @tf.function
    def _fubini_study_n_metrics(self, points, n=None, t=tf.complex(1., 0.)):
        r"""Computes the Fubini-Study equivalent on the ambient space for each
        Kaehler modulus.

        .. math:: g_\alpha = \partial_i \bar\partial_j \ln \rho_\alpha

        Args:
            points (tf.tensor([bSize, ncoords], tf.complex64)): Points.
            n (int, optional): n^th Kahler potential term. Defaults to None.
            t (tf.complex, optional): Volume factor. Defaults to 1+0j.

        Returns:
            tf.tensor([bSize, ncoords, ncoords], tf.complex64): 
                Metric contribution at each point for t_n.
        """
        alpha = 0 if n is None else n 
        degrees = self.sections[alpha]
        ms = tf.math.pow(points[:, tf.newaxis, :], degrees[tf.newaxis, :, :])
        ms = tf.math.reduce_prod(ms, axis=int(-1))
        mss = ms * tf.math.conj(ms)     
        kappa_alphas = tf.reduce_sum(mss, int(-1))
        zizj = points[:, :, tf.newaxis] * tf.math.conj(points[:, tf.newaxis, :])
        J_alphas = float(1.) / zizj
        J_alphas = tf.einsum('x,xab->xab', float(1.) / (kappa_alphas**int(2)), J_alphas)
        coeffs = tf.einsum('xa,xb,ai,aj->xij', mss, mss, degrees, degrees) - tf.einsum('xa,xb,ai,bj->xij', mss, mss, degrees, degrees)
        return J_alphas * coeffs * t/tf.constant(np.pi, dtype=tf.complex64)

    def _generate_helpers(self):
        """Additional helper functions."""
        self.nTransitions = int(np.max(np.sum(~self.patch_masks, axis=-2)))
        self.fixed_patches = self._generate_all_patches()
        patch_degrees = get_all_patch_degrees(self.glsm_charges, self.patch_masks)
        w_of_x, del_w_of_x, del_w_of_z = compute_all_w_of_x(patch_degrees, self.patch_masks)
        self.patch_degrees = tf.cast(patch_degrees, dtype=tf.complex64)
        self.transition_coefficients = tf.cast(w_of_x, dtype=tf.complex64)
        self.transition_degrees = tf.cast(del_w_of_z, dtype=tf.complex64)
        self.patch_masks = tf.cast(self.patch_masks, dtype=tf.bool)
        # Not needed; cause transition loss is different
        # self.degrees = None <- also only needed for rescaling and patches in FS
        self.proj_matrix = None
        self._proj_indices = None
        return None

    def _generate_all_patches(self):
        """Torics only have on hypersurface, thus we can generate all patches"""
        # fixed patches will be of shape (ncoords, npatches, nTransitions)
        fixed_patches = np.repeat(np.arange(self.nPatches), self.nTransitions)
        fixed_patches = np.tile(fixed_patches, self.ncoords)
        fixed_patches = fixed_patches.reshape(
            (self.ncoords, self.nPatches, self.nTransitions))
        for i in range(self.ncoords):
            # keep each coordinate fixed and add all patches, where its zero
            all_patches = ~self.patch_masks[:, i]
            all_indices = np.where(all_patches)[0]
            fixed_patches[i, all_indices, 0:len(all_indices)] = all_indices * \
                np.ones((len(all_indices), len(all_indices)), dtype=np.int)
        return tf.cast(fixed_patches, dtype=tf.int64)

    @tf.function
    def _get_patch_coordinates(self, points, patch_index):
        r"""Goes to a patch specified by patch_index which contains the patch
        index for self.patch_degrees and return the coordinates in this patch.
        """
        # NOTE: this is different than for regular FS models it takes the patch index as argument, not a mask
        degrees = tf.gather(self.patch_degrees, patch_index[:, 0])
        scaled_points = points[:, tf.newaxis, :]
        scaled_points = tf.math.pow(scaled_points, degrees)
        return tf.reduce_prod(scaled_points, axis=-1)

    @tf.function
    def _mask_to_patch_index(self, mask):
        """Computes the patch index in self.patch_mask of a given patch mask."""
        # NOTE: this computes the patch index, not the indices of the patch coordinates.
        mask_to_index = tf.math.equal(mask[:, tf.newaxis, :], self.patch_masks)
        mask_to_index = tf.reduce_all(mask_to_index, axis=-1)
        indices = tf.where(mask_to_index)
        return indices[:, 1:]

    @tf.function
    def compute_transition_loss(self, points):
        r"""Computes transition loss at each point.

        This function is essentially the same as for `FSModel`. It only differs
        in the patch selection. TODO: Unify this approach?

        Args:
            points (tf.tensor([bSize, 2*ncoords], tf.float)): Points.

        Returns:
            tf.tensor([bSize], tf.complex): transition loss at each point.
        """
        inv_one_mask = self._get_inv_one_mask(points)
        current_patch_mask = ~inv_one_mask
        current_patch_index = self._mask_to_patch_index(current_patch_mask)
        cpoints = tf.complex(points[:, :self.ncoords],
                             points[:, self.ncoords:])
        fixed = self._find_max_dQ_coords(points)
        other_patches = tf.gather_nd(
            self.fixed_patches,
            tf.concat([fixed, current_patch_index], axis=-1))
        other_patch_mask = tf.gather(self.patch_masks, other_patches)
        other_patch_mask = tf.reshape(other_patch_mask, (-1, self.ncoords))
        # NOTE: This will include same to same patch transitions
        exp_points = tf.repeat(cpoints, self.nTransitions, axis=-2)
        patch_points = self._get_patch_coordinates(exp_points, tf.reshape(other_patches, (-1, 1)))
        fixed = tf.reshape(tf.tile(fixed, [1, self.nTransitions]), (-1, self.nhyper))
        real_points = tf.concat((tf.math.real(patch_points), tf.math.imag(patch_points)), axis=-1)
        gj = self(real_points, training=True, j_elim=fixed)
        gi = tf.repeat(self(points), self.nTransitions, axis=0)
        current_patch_mask = tf.repeat(
            current_patch_mask, self.nTransitions, axis=0)
        Tij = self.get_transition_matrix(
            patch_points, other_patch_mask, current_patch_mask, fixed)
        all_t_loss = tf.math.abs(self.transition_loss_matrices(gj, gi, Tij))
        all_t_loss = tf.math.reduce_sum(all_t_loss, axis=[1, 2])
        # This should now be nTransitions 
        all_t_loss = tf.reshape(all_t_loss, (-1, self.nTransitions))
        all_t_loss = tf.math.reduce_sum(all_t_loss, axis=-1)
        return all_t_loss/(self.nTransitions*self.nfold**2)

    @tf.function
    def get_transition_matrix(self, points, i_mask, j_mask, fixed):
        r"""Computes transition matrix between patch i and j 
        for each point in points where fixed is the coordinate,
        which is being eliminated.

        See also: :py:meth:`cymetric.models.FSModel.get_transition_matrix`.

        This function is more simplified than the original one as we 
        compute a basis for all :math:`\partial w_i / \partial z_j` before hand.

        Args:
            points (tf.tensor([bSize, 2*ncoords], tf.float32)): Points.
            i_mask (tf.tensor([bSize, ncoords], tf.bool)): Mask of pi-indices.
            j_mask (tf.tensor([bSize, ncoords], tf.bool)): Mask of pi-indices.
            fixed (tf.tensor([bSize, 1], tf.int64)): Elimination indices.

        Returns:
            tf.tensor([bSize, nfold, nfold], tf.complex64): T_ij on the CY.
        """
        same_patch = tf.where(tf.math.reduce_all(i_mask == j_mask, axis=-1))
        diff_patch = tf.where(~tf.math.reduce_all(i_mask == j_mask, axis=-1))
        same_patch = same_patch[:, 0]
        diff_patch = diff_patch[:, 0]
        n_p = tf.math.reduce_sum(tf.ones_like(fixed[:, 0]))
        n_p_red = tf.math.reduce_sum(tf.ones_like(diff_patch))

        # reduce non trivial
        i_mask_red = tf.gather(i_mask, diff_patch)
        j_mask_red = tf.gather(j_mask, diff_patch)
        fixed_red = tf.gather(fixed, diff_patch)
        points_red = tf.gather(points, diff_patch)
        
        # recompute patch indices
        i_patch_indices = self._mask_to_patch_index(i_mask_red)
        j_patch_indices = self._mask_to_patch_index(j_mask_red)

        # fill tij
        tij_indices = tf.concat([fixed_red, i_patch_indices, j_patch_indices], axis=-1)
        tij_degrees = tf.gather_nd(self.transition_degrees, tij_indices)
        tij_coeff = tf.gather_nd(self.transition_coefficients, tij_indices)
        tij_red = tf.math.pow(points_red[:, tf.newaxis, tf.newaxis, :], tij_degrees)
        tij_red = tf.multiply(tij_coeff, tf.reduce_prod(tij_red, axis=-1))
        tij_red = tf.transpose(tij_red, perm=[0, 2, 1])

        # fill tij
        tij_eye = tf.eye(self.nfold, batch_shape=[n_p-n_p_red], dtype=tf.complex64)
        tij_all = tf.zeros((n_p, self.nfold, self.nfold), dtype=tf.complex64)
        tij_all = tf.tensor_scatter_nd_update(tij_all, tf.reshape(diff_patch, (-1, 1)), tij_red)
        tij_all = tf.tensor_scatter_nd_update(tij_all, tf.reshape(same_patch, (-1, 1)), tij_eye)
        return tij_all


class PhiFSModelToric(ToricModel):
    r"""PhiFSModelToric inherits from :py:class:`ToricModel`.

    The PhiModel learns the scalar potential correction to some Kaehler metric
    to make it the Ricci-flat metric. The Kaehler metric is taken to be a toric 
    equivalent of the Fubini-Study metric. See also :py:class:`PhiFSModel`.

    Example:
        Is similar to :py:class:`FreeModel`. Replace the nn accordingly.

        >>> nn = tfk.Sequential(
        ...     [   
        ...         tfk.layers.Input(shape=(ncoords)),
        ...         tfk.layers.Dense(64, activation="gelu"),
        ...         tfk.layers.Dense(1, use_bias=False),
        ...     ]
        ... )
        >>> model = PhiFSModelToric(nn, BASIS, toric_data = toric_data)

    You have to use this model if you want to remain in the same Kaehler class
    specified by the Kaehler moduli.
    """
    def __init__(self, *args, **kwargs):
        r"""PhiFSModelToric is a tensorflow model predicting CY metrics.
        
        The output of this model has the following Ansatz
        
        .. math:: 
            
            g_{\text{out}} = g_{\text{FS'}} +
                \partial \bar{\partial} \phi_{\text{NN}}
        
        and returns a hermitian (nfold, nfold) tensor. The model is by
        defintion Kaehler and thus this loss contribution is by default
        disabled.
        """
        super(PhiFSModelToric, self).__init__(*args, **kwargs)
        self.learn_kaehler = tf.cast(False, dtype=tf.bool)

    def call(self, input_tensor, training=True, j_elim=None):
        r"""Prediction of the model.

        .. math:: 
            g_{\text{out}; ij} = g_{\text{FS'}; ij} +
                \partial_i \bar{\partial}_j \phi_{\text{NN}}

        Args:
            input_tensor (tf.tensor([bSize, 2*ncoords], tf.float32)): Points.
            training (bool, optional): Not used. Defaults to True.
            j_elim (tf.tensor([bSize, nHyper], tf.int64), optional): 
                Coordinates(s) to be eliminated in the pullbacks.
                If None will take max(dQ/dz). Defaults to None.

        Returns:
            tf.tensor([bSize, nfold, nfold], tf.complex64):
                Prediction at each point.
        """
        # nn prediction
        with tf.GradientTape(persistent=False) as tape1:
            tape1.watch(input_tensor)
            with tf.GradientTape(persistent=False) as tape2:
                tape2.watch(input_tensor)
                # see comment at other Phi model why training disabled.
                phi = self.model(input_tensor, training=False)
            d_phi = tape2.gradient(phi, input_tensor)
        dd_phi = tape1.batch_jacobian(d_phi, input_tensor)
        dx_dx_phi, dx_dy_phi, dy_dx_phi, dy_dy_phi = \
            0.25*dd_phi[:, :self.ncoords, :self.ncoords], \
            0.25*dd_phi[:, :self.ncoords, self.ncoords:], \
            0.25*dd_phi[:, self.ncoords:, :self.ncoords], \
            0.25*dd_phi[:, self.ncoords:, self.ncoords:]
        dd_phi = tf.complex(dx_dx_phi + dy_dy_phi, dx_dy_phi - dy_dx_phi)
        pbs = self.pullbacks(input_tensor, j_elim=j_elim)
        dd_phi = tf.einsum('xai,xij,xbj->xab', pbs, dd_phi, tf.math.conj(pbs))
        
        # fs metric
        fs_cont = self.fubini_study_pb(input_tensor, pb=pbs, j_elim=j_elim)
        # return g_fs + \del\bar\del\phi
        return tf.math.add(fs_cont, dd_phi)

    def compute_transition_loss(self, points, num_random_scalings=10):
        r"""Computes transition loss at each point. In the case of the Phi model, we demand that \phi(\lambda^q_i z_i)=\phi(z_i)

        Args:
            points (tf.tensor([bSize, 2*ncoords], tf.float32)): Points.
            num_random_scalings (int): If None, uses scalings for each patch to set one coordinate to one. 
                                       If a number, uses this many random scalings for \lambda

        Returns:
            tf.tensor([bSize], tf.float32): Transition loss at each point.
        """
        if num_random_scalings is None:
            return super(PhiFSModelToric, self).compute_transition_loss(points)
        
        cpoints = tf.complex(points[:, :self.ncoords], points[:, self.ncoords:])
        num_pns = self.glsm_charges.shape[0]
        
        # we scale the lambdas_rand to have abs value in [0.1, 0.9]
        scale_factor_rand = tf.cast(tf.random.uniform(minval=0.1, maxval=0.9, shape=(num_random_scalings, num_pns), dtype=tf.float32), dtype=tf.complex64)
        scale_factor_rand = tf.repeat(tf.expand_dims(scale_factor_rand, -1), repeats=self.ncoords, axis=-1)
        
        # real and imaginary part of random lambdas (we draw a different one for each ambient P^n)
        lambdas_rand = tf.random.uniform(minval=-1, maxval=1, shape=(num_random_scalings, num_pns, 2), dtype=tf.float32)
        lambdas_rand = tf.complex(lambdas_rand[:,:,0], lambdas_rand[:,:,1])
        lambdas_rand = tf.repeat(tf.expand_dims(lambdas_rand, -1), repeats=self.ncoords, axis=-1)
        lambdas_rand = scale_factor_rand * lambdas_rand/(lambdas_rand * tf.math.conj(lambdas_rand))**(.5)  # rescale \lambdas
        lambdas_rand = lambdas_rand ** self.glsm_charges
        lambdas_rand = tf.reduce_prod(lambdas_rand, 1)

        scaled_points = tf.einsum('xi,ai->xai', cpoints, lambdas_rand)
        scaled_points = tf.reshape(scaled_points, (-1, self.ncoords))  # this has now shape (batch_size*num_random_scalings, ncoords)
        
        real_patch_points = tf.concat((tf.math.real(scaled_points), tf.math.imag(scaled_points)), axis=-1)
        gj = self.model(real_patch_points, training=True)
        gi = tf.repeat(self.model(points), num_random_scalings, axis=0)
        all_t_loss = tf.math.abs(gi-gj)
        all_t_loss = tf.reshape(all_t_loss, (-1, num_random_scalings))
        all_t_loss = tf.math.reduce_sum(all_t_loss**self.n[2], axis=-1)
        return all_t_loss / num_random_scalings    
 
    @tf.function
    def _fubini_study_n_potentials(self, points, n=None, t=tf.complex(1., 0.)):
        r"""Computes the Fubini-Study equivalent on the ambient space for each
        Kaehler modulus.

        .. math:: g_\alpha = \partial_i \bar\partial_j \ln \rho_\alpha

        Args:
            points (tf.tensor([bSize, ncoords], tf.complex64)): Points.
            t (tf.complex, optional): Volume factor. Defaults to 1+0j.

        Returns:
            tf.tensor([bSize, ncoords, ncoords], tf.complex64):
                Metric contribution at each point for t_n.
        """
        alpha = 0 if n is None else n
        degrees = self.sections[alpha]
        ms = tf.math.pow(points[:, tf.newaxis, :], degrees[tf.newaxis, :, :])
        ms = tf.math.reduce_prod(ms, axis=int(-1))
        mss = ms * tf.math.conj(ms)
        kappa_alphas = tf.reduce_sum(mss, int(-1))
        return tf.cast(tf.math.real(t/np.pi), dtype=tf.float32) * tf.cast(tf.math.real(tf.math.log(kappa_alphas)), tf.float32)

    def get_kahler_potential(self, points):
        r"""Returns toric equivalent of the FS Kahler potential for each point.

        .. math::

            J = t^\alpha J_\alpha \quad \text{ with: }
                J_\alpha = \frac{i}{2\pi} \partial \bar\partial \ln \rho_\alpha

        :math:`\rho_\alpha` is a basis of sections.

        Args:
            points (tf.tensor([bSize, 2*ncoords], tf.float32)): Points.
            pb (tf.tensor([bSize, nfold, ncoords], tf.float32)):
                Pullback at each point. Overwrite j_elim. Defaults to None.
            j_elim (tf.tensor([bSize, nHyper], tf.int64), optional):
                Coordinates(s) to be eliminated in the pullbacks.
                If None will take max(dQ/dz). Defaults to None.

        Returns:
            tf.tensor([bSize, nfold, nfold], tf.complex64):
                Kaehler metric at each point.
        """
        cpoints = tf.cast(tf.complex(points[:, :self.ncoords], points[:, self.ncoords:]), dtype=tf.complex64)

        k_fs = self._fubini_study_n_potentials(cpoints, t=self.kmoduli[0])
        if len(self.kmoduli) != 1:
            for i in range(1, len(self.kmoduli)):
                k_fs += self._fubini_study_n_potentials(cpoints, i, t=self.kmoduli[i])

        k_fs += tf.reshape(self.model(points), [-1])
        return k_fs


class MatrixFSModelToric(ToricModel):
    r"""MatrixFSModelToric inherits from :py:class:`ToricModel`.

    See also: :py:class:`MatrixFSModel` and :py:class:`FreeModel`
    """
    def __init__(self, *args, **kwargs):
        r"""MatrixFSModelToric is a tensorflow model predicting CY metrics.
        
        The output of this model has the following Ansatz
        
        .. math:: g_{\text{out}} = g_{\text{FS'}} (1 + g_{\text{NN}})
        
        with matrix multiplication and returns a hermitian (nfold, nfold)
        tensor.
        """
        super(MatrixFSModelToric, self).__init__(*args, **kwargs)

    def call(self, input_tensor, training=True, j_elim=None):
        r"""Prediction of the model.

        .. math:: 
        
            g_{\text{out}; ik} = g_{\text{FS}; ij} (1_{jk} + g_{\text{NN}; jk})

        Args:
            input_tensor (tf.tensor([bSize, 2*ncoords], tf.float32)): Points.
            training (bool, optional): Not used. Defaults to True.
            j_elim (tf.tensor([bSize, nHyper], tf.int64), optional): 
                Coordinates(s) to be eliminated in the pullbacks.
                If None will take max(dQ/dz). Defaults to None.

        Returns:
            tf.tensor([bSize, nfold, nfold], tf.complex):
                Prediction at each point.
        """
        nn_cont = self.to_hermitian(self.model(input_tensor, training=training))
        fs_cont = self.fubini_study_pb(input_tensor, j_elim=j_elim)
        return fs_cont + tf.linalg.matmul(fs_cont, nn_cont)
