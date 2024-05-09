""" 
Pullbacked fubini study metric implemented as a tfk.model.
"""
import tensorflow as tf
import itertools as it
from cymetric.pointgen.nphelper import generate_monomials, get_levicivita_tensor
import numpy as np
tfk = tf.keras


class FSModel(tfk.Model):
    r"""FSModel implements all underlying tensorflow routines for pullbacks 
    and computing various loss contributions.

    It is *not* intended for actual training and does not have an explicit 
    training step included. It should be used to write your own custom models
    for training of CICYs. Toric hypersurfaces require some extra routines,
    which are implemented here: `cymetric.models.tfmodels.ToricModel`
    """
    def __init__(self, BASIS, norm=None):
        r"""A tensorflow implementation of the pulled back Fubini-Study metric.

        Args:
            BASIS (dict): a dictionary containing all monomials and other
                relevant information from e.g.
                `cymetric.pointgen.pointgen.PointGenerator`
            norm ([5//NLOSS], optional): degree of norm for various losses.
                Defaults to 1 for all but Kaehler norm (2).
        """
        super(FSModel, self).__init__()
        self.BASIS = BASIS
        self.ncoords = len(self.BASIS['DQDZB0'])
        self.nProjective = len(self.BASIS['AMBIENT'])
        self.nfold = int(tf.math.real(self.BASIS['NFOLD']))
        if norm is None:
            self.n = [tf.cast(1., dtype=tf.float32) for _ in range(5)]
            # Default: we want to punish violation of kählerity stronger
            self.n[1] = tf.cast(2., dtype=tf.float32)
        else:
            self.n = [tf.cast(n, dtype=tf.float32) for n in norm]
        # projective vars
        self.degrees = tf.cast(tf.ones_like(self.BASIS['AMBIENT']) + self.BASIS['AMBIENT'], dtype=tf.int32)
        self.pi = tf.constant(tf.cast(np.pi, dtype=tf.complex64))
        self.nhyper = int(tf.cast(BASIS['NHYPER'], dtype=tf.int64))
        self._generate_helpers()
        
    def _generate_helpers(self):
        r"""Bunch of helper functions to run during initialization"""
        self.lc = tf.convert_to_tensor(get_levicivita_tensor(self.nfold), dtype=tf.complex64)
        self.proj_matrix = self._generate_proj_matrix()
        self.nTransitions = self._patch_transitions()
        if self.nhyper == 1:
            self.fixed_patches = self._generate_all_patches()
        self._proj_indices = self._generate_proj_indices()
        self.slopes = self._target_slopes()

    def _generate_proj_matrix(self):
        r"""TensorFlow does not allow for nice slicing. Here we create
        `proj_matrix` which stores information about the ambient spaces, so that
        we can slice via matrix products. See usage in: `self.fubini_study_pb`.
        """
        proj_matrix = {}
        for i in range(self.nProjective):
            matrix = np.zeros((self.degrees[i], self.ncoords),
                              dtype=np.complex64)
            s = np.sum(self.degrees[:i])
            e = np.sum(self.degrees[:i+1])
            matrix[:, s:e] = np.eye(self.degrees[i], dtype=np.complex64)
            proj_matrix[str(i)] = tf.cast(matrix, dtype=tf.complex64)
        return proj_matrix

    def _generate_proj_indices(self):
        r"""Makes a tensor with corresponding projective index for each variable
        from the ambient space.
        """
        flat_list = []
        for i, p in enumerate(self.degrees):
            for _ in range(p):
                flat_list += [i]
        return tf.cast(flat_list, dtype=tf.int64)

    def _generate_all_patches(self):
        r"""We generate all possible patches for CICYs. Note for CICYs with
        more than one hypersurface patches are generated on spot.
        """
        fixed_patches = []
        for i in range(self.ncoords):
            all_patches = np.array(
                list(it.product(*[[j for j in range(sum(self.degrees[:k]), sum(self.degrees[:k+1])) if j != i] for k in range(len(self.degrees))], repeat=1)))
            if len(all_patches) == self.nTransitions:
                fixed_patches += [all_patches]
            else:
                # need to pad if there are less than nTransitions.
                all_patches = np.tile(all_patches, (int(self.nTransitions/len(all_patches)) + 1, 1))
                fixed_patches += [all_patches[0:self.nTransitions]]
        fixed_patches = np.array(fixed_patches)
        return tf.cast(fixed_patches, dtype=tf.int64)

    def _patch_transitions(self):
        r"""Computes the maximum number of patch transitions with same fixed
        variables. This is often not the same number for all patches. In case
        there are less transitions we padd with same to same patches."""
        nTransitions = 0
        for t in generate_monomials(self.nProjective, self.nhyper):
            tmp_deg = [d-t[j] for j, d in enumerate(self.degrees)]
            n = tf.math.reduce_prod(tmp_deg)
            if n > nTransitions:
                nTransitions = n
        # if tf.int None vs unknown shape issue in for loop
        # over counting by one which is same to same transition
        return int(nTransitions)

    def _target_slopes(self):
        ks = tf.eye(len(self.BASIS['KMODULI']), dtype=tf.complex64)
        
        if self.nfold == 1:
            slope = tf.einsum('a, xa->x', self.BASIS['INTNUMS'], ks)

        elif self.nfold == 2:
            slope = tf.einsum('ab, a, xb->x', self.BASIS['INTNUMS'], self.BASIS['KMODULI'], ks)

        elif self.nfold == 3:
            slope = tf.einsum('abc, a, b, xc->x', self.BASIS['INTNUMS'], self.BASIS['KMODULI'], self.BASIS['KMODULI'], ks)
        
        elif self.nfold == 4:
            slope = tf.einsum('abcd, a, b, c, xd->x', self.BASIS['INTNUMS'], self.BASIS['KMODULI'], self.BASIS['KMODULI'], self.BASIS['KMODULI'], ks)
        
        elif self.nfold == 5:
            slope = tf.einsum('abcd, a, b, c, d, xe->x', self.BASIS['INTNUMS'], self.BASIS['KMODULI'], self.BASIS['KMODULI'], self.BASIS['KMODULI'], self.BASIS['KMODULI'], ks)

        else:
            self.logger.error('Only implemented for nfold <= 5. Run the tensor contraction yourself :).')
            raise NotImplementedError
        
        return slope

    @tf.function
    def _calculate_slope(self, args):
        r"""Computes the slopes \mu(F_i) = \int J \wedge J \wegde F_i at the point in Kahler moduli space t_a = 1 for all a
        and for F_i = O_X(0, 0,... , 1, 0, ..., 0), i.e. the flux integers are k_i^a = \delta_{i,a}"""
        pred, f_a = args[0], args[1]
        if self.nfold == 1:
            slope = tf.einsum('xab->x',
                              f_a)
        elif self.nfold == 2:
            slope = tf.einsum('xab,xcd,ac,bd->x',
                              pred, f_a, self.lc, self.lc)
        elif self.nfold == 3:
            slope = tf.einsum('xab,xcd,xef,ace,bdf->x',
                              pred, pred, f_a, self.lc, self.lc)
        elif self.nfold == 4:
            slope = tf.einsum('xab,xcd,xef,xgh,aceg,bdfh->x',
                              pred, pred, pred, f_a, self.lc, self.lc)
        elif self.nfold == 5:
            slope = tf.einsum('xab,xcd,xef,xgh,xij,acegi,bdfhj->x',
                              pred, pred, pred, pred, f_a, self.lc, self.lc)
        else:
            self.logger.error('Only implemented for nfold <= 5. Run the tensor contraction yourself :).')
            raise NotImplementedError
        
        slope = tf.cast(1./tf.exp(tf.math.lgamma(tf.cast(tf.math.real(self.BASIS['NFOLD']), dtype=tf.float32) + 1)), dtype=tf.complex64) * slope
        return slope

    def call(self, input_tensor, training=True, j_elim=None):
        r"""Call method. Computes the pullbacked 
        Fubini-Study metric at each point in input_tensor.

        Args:
            input_tensor (tf.tensor([bSize, 2*ncoords], tf.float)): Points.
            training (bool, optional): Switch between training and eval mode. Not used at the moment
            j_elim (tf.array([bSize], tf.int64)): index to be eliminated.
                Coordinates(s) to be eliminated in the pullbacks.
                If None will take max(dQ/dz). Defaults to None.

        Returns:
            tf.tensor([bSize, nfold, nfold], tf.complex): 
                Pullbacked FS-metric at each point.
        """
        return self.fubini_study_pb(input_tensor, j_elim=j_elim)

    @tf.function
    def compute_kaehler_loss(self, x):
        r"""Computes Kähler loss.

        .. math::
            \cal{L}_{\text{dJ}} = \sum_{ijk} ||Re(c_{ijk})||_n + 
                    ||Im(c_{ijk})||_n \\
                \text{with: } c_{ijk} = g_{i\bar{j},k} - g_{k\bar{j},i}

        Args:
            x (tf.tensor([bSize, 2*ncoords], tf.float)): Points.

        Returns:
            tf.tensor([bSize, 1], tf.float): \sum_ijk abs(cijk)**n
        """
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(x)
            # set training to False for batch_jacobian to work
            y_pred = self(x, training=False)
            pb = self.pullbacks(x)
            gij_re, gij_im = tf.math.real(y_pred), tf.math.imag(y_pred)
        gijk_re = tf.cast(t1.batch_jacobian(gij_re, x), dtype=tf.complex64)
        gijk_im = tf.cast(t1.batch_jacobian(gij_im, x), dtype=tf.complex64)
        cijk = 0.5*(gijk_re[:, :, :, :self.ncoords] +
                    gijk_im[:, :, :, self.ncoords:] +
                    1.j*gijk_im[:, :, :, :self.ncoords] -
                    1.j*gijk_re[:, :, :, self.ncoords:])
        cijk_pb = tf.einsum('xija,xka->xijk', cijk, pb)
        cijk_pb = cijk_pb - tf.transpose(cijk_pb, [0, 3, 2, 1])
        cijk_loss = tf.math.reduce_sum(tf.abs(cijk_pb)**self.n[1], [1, 2, 3])
        return cijk_loss

    @tf.function
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
        # TODO: Naming conventions here and in pointgen are different.
        if self.nProjective > 1:
            # we go through each ambient space factor and create fs.
            cpoints = tf.complex(
                points[:, :self.degrees[0]],
                points[:, self.ncoords:self.ncoords+self.degrees[0]])
            fs = self._fubini_study_n_metrics(cpoints, n=self.degrees[0], t=ts[0])
            fs = tf.einsum('xij,ia,bj->xab', fs, self.proj_matrix['0'], tf.transpose(self.proj_matrix['0']))
            for i in range(1, self.nProjective):
                s = tf.reduce_sum(self.degrees[:i])
                e = s + self.degrees[i]
                cpoints = tf.complex(points[:, s:e],
                                     points[:, self.ncoords+s:self.ncoords+e])
                fs_tmp = self._fubini_study_n_metrics(
                    cpoints, n=self.degrees[i], t=ts[i])
                fs_tmp = tf.einsum('xij,ia,bj->xab',
                                   fs_tmp, self.proj_matrix[str(i)],
                                   tf.transpose(self.proj_matrix[str(i)]))
                fs += fs_tmp
        else:
            cpoints = tf.complex(
                points[:, :self.ncoords],
                points[:, self.ncoords:2*self.ncoords])
            fs = self._fubini_study_n_metrics(cpoints,
                                              t=ts[0])

        if pb is None:
            pb = self.pullbacks(points, j_elim=j_elim)
        fs_pb = tf.einsum('xai,xij,xbj->xab', pb, fs, tf.math.conj(pb))
        return fs_pb

    @tf.function
    def _find_max_dQ_coords(self, points):
        r"""Finds in each hypersurface the coordinates for which |dQ/dzj|
        is largest.

        NOTE:
            If a coordinate is the largest for more than one hypersurface, it
            will only be selected for the first and subsequently the second 
            largest will be taken, etc..

        Args:
            points (tf.tensor([bSize, 2*ncoords], tf.float32)): Points.

        Returns:
            tf.tensor([bSize, nhyper], tf.int64): max(dQ/dz) index per hyper.
        """
        # creates coordinate mask with patch coordinates
        cpoints = tf.complex(points[:, :self.ncoords], points[:, self.ncoords:])
        available_mask = tf.cast(self._get_inv_one_mask(points), dtype=tf.complex64)

        indices = []
        for i in range(self.nhyper):
            dQdz = self._compute_dQdz(cpoints, i)
            if i == 0:
                indices = tf.argmax(tf.math.abs(dQdz*available_mask), axis=-1)
                indices = tf.reshape(indices, (-1, 1))
            else:
                max_dq = tf.argmax(tf.math.abs(dQdz*available_mask), axis=-1)
                indices = tf.concat([indices, tf.reshape(max_dq, (-1, 1))], axis=-1)
            available_mask -= tf.one_hot(
                indices[:, i], self.ncoords, dtype=tf.complex64)
        return indices

    @tf.function
    def pullbacks(self, points, j_elim=None):
        r"""Computes the pullback tensor at each point.

        NOTE:
            Scatter-nd uses a while loop when creating the graph.

        .. math::

            J^i_a = \frac{dz_i}{dx_a}

        where x_a are the nfold good coordinates after eliminating j_elim.

        Args:
            points (tf.tensor([bSize, 2*ncoords], tf.float32)): Points.
            j_elim (tf.tensor([bSize, nHyper], tf.int64), optional): 
                Coordinates(s) to be eliminated in the pullbacks.
                If None will take max(dQ/dz). Defaults to None.

        Returns:
            tf.tensor([bSize, nfold, ncoords], tf.complex64): Pullback at each
                point.
        """
        inv_one_mask = self._get_inv_one_mask(points)
        cpoints = tf.complex(points[:, :self.ncoords],
                             points[:, self.ncoords:])
        if j_elim is None:
            dQdz_indices = self._find_max_dQ_coords(points)
        else:
            dQdz_indices = j_elim
        full_mask = tf.cast(inv_one_mask, dtype=tf.float32)
        for i in range(self.nhyper):
            dQdz_mask = -1.*tf.one_hot(dQdz_indices[:, i], self.ncoords)
            full_mask = tf.math.add(full_mask, dQdz_mask)
        n_p = tf.cast(tf.reduce_sum(tf.ones_like(full_mask[:, 0])), dtype=tf.int64)
        full_mask = tf.cast(full_mask, dtype=tf.bool)
        x_z_indices = tf.where(full_mask)
        good_indices = x_z_indices[:, 1:2]
        pullbacks = tf.zeros((n_p, self.nfold, self.ncoords),
                             dtype=tf.complex64)
        y_indices = tf.repeat(
            tf.expand_dims(tf.cast(tf.range(self.nfold), dtype=tf.int64), 0),
            n_p, axis=0)
        y_indices = tf.reshape(y_indices, (-1, 1))
        diag_indices = tf.concat((x_z_indices[:, 0:1], y_indices, good_indices),
                                 axis=-1)
        pullbacks = tf.tensor_scatter_nd_update(
            pullbacks, diag_indices,
            tf.ones(self.nfold*n_p, dtype=tf.complex64)
        )
        fixed_indices = tf.reshape(dQdz_indices, (-1, 1))
        for i in range(self.nhyper):
            # compute p_i\alpha eq (5.24)
            pia_polys = tf.gather_nd(self.BASIS['DQDZB'+str(i)], good_indices)
            pia_factors = tf.gather_nd(self.BASIS['DQDZF'+str(i)], good_indices)
            pia = tf.expand_dims(tf.repeat(cpoints, self.nfold, axis=0), 1)
            pia = tf.math.pow(pia, pia_polys)
            pia = tf.reduce_prod(pia, axis=-1)
            pia = tf.reduce_sum(tf.multiply(pia_factors, pia), axis=-1)
            pia = tf.reshape(pia, (-1, 1, self.nfold))
            if i == 0:
                dz_hyper = pia
            else:
                dz_hyper = tf.concat((dz_hyper, pia), axis=1)
            # compute p_ifixed
            pif_polys = tf.gather_nd(self.BASIS['DQDZB'+str(i)], fixed_indices)
            pif_factors = tf.gather_nd(self.BASIS['DQDZF'+str(i)],
                                       fixed_indices)
            pif = tf.expand_dims(tf.repeat(cpoints, self.nhyper, axis=0), 1)
            pif = tf.math.pow(pif, pif_polys)
            pif = tf.reduce_prod(pif, axis=-1)
            pif = tf.reduce_sum(tf.multiply(pif_factors, pif), axis=-1)
            pif = tf.reshape(pif, (-1, 1, self.nhyper))
            if i == 0:
                B = pif
            else:
                B = tf.concat((B, pif), axis=1)
        all_dzdz = tf.einsum('xij,xjk->xki', tf.linalg.inv(B), tf.complex(-1., 0.) * dz_hyper)

        # fill at the right position
        for i in range(self.nhyper):
            fixed_indices = tf.reshape(
                tf.repeat(dQdz_indices[:, i], self.nfold), (-1, 1))
            zjzi_indices = tf.concat(
                (x_z_indices[:, 0:1], y_indices, fixed_indices), axis=-1)
            zjzi_values = tf.reshape(all_dzdz[:,:,i], [self.nfold*n_p])
            pullbacks = tf.tensor_scatter_nd_update(
                pullbacks, zjzi_indices, zjzi_values)
        return pullbacks

    @tf.function
    def _get_inv_one_mask(self, points):
        r"""Computes mask with True when z_i != 1+0.j."""
        cpoints = tf.complex(points[:, :self.ncoords], points[:, self.ncoords:])
        return tf.math.logical_not(tf.experimental.numpy.isclose(cpoints, 1.))
        # one_mask = tf.math.logical_or(
        #     tf.math.less(points[:, 0:self.ncoords], self.epsilon_low),
        #     tf.math.greater(points[:, 0:self.ncoords], self.epsilon_high))
        # zero_mask = tf.math.greater(
        #     tf.math.abs(points[:, self.ncoords:]), 1.-self.epsilon_low)
        # inv_mask = tf.math.logical_and(
        #     tf.math.logical_not(one_mask), tf.math.logical_not(zero_mask))
        # return tf.math.logical_not(inv_mask)

    @tf.function
    def _indices_to_mask(self, indices):
        r"""Takes indices ([bSize,nTrue], int) and creates a faux coordinates
        mask. NOTE: the output is *not* of boolean type.
        """
        mask = tf.one_hot(indices, depth=self.ncoords)
        mask = tf.math.reduce_sum(mask, axis=1)
        return mask

    @tf.function
    def _generate_patches(self, args):
        r"""Generates possible patch transitions for the patches sepcified in
        args. Note it uses tf.split which won't allow for tf.vectorized_map, 
        because of different signature during graph building.
        """
        # TODO: Clean up all the tf.int64; some are needed because tf mixes its default int types for range and indexing
        fixed = args[0:self.nhyper]
        original = args[self.nhyper:]
        inv_fixed_mask = ~tf.cast(tf.reduce_sum(
            tf.one_hot(fixed, self.ncoords), axis=0), tf.bool)
        fixed_proj = tf.one_hot(
            tf.gather(self._proj_indices, fixed),
            self.nProjective, dtype=tf.int64)
        fixed_proj = tf.reduce_sum(fixed_proj, axis=0)
        splits = tf.cast(self.degrees, dtype=tf.int64) - fixed_proj
        all_coords = tf.boolean_mask(
            tf.cast(tf.range(self.ncoords), dtype=tf.int64),
            inv_fixed_mask)
        products = tf.split(all_coords, splits)
        all_patches = tf.stack(tf.meshgrid(*products, indexing='ij'), axis=-1)
        all_patches = tf.reshape(all_patches, (-1, self.nProjective))
        npatches = tf.reduce_sum(tf.ones_like(all_patches[:, 0]))
        if npatches != self.nTransitions:
            same = tf.tile(original, [self.nTransitions-npatches])
            same = tf.reshape(same, (-1, self.nProjective))
            same = tf.cast(same, dtype=tf.int64)
            return tf.concat([all_patches, same], axis=0)
        return all_patches

    @tf.function
    def _generate_patches_vec(self, combined):
        # NOTE: vectorized_map makes issues cause `_generate_patches`
        # has a different signature depending on its input.
        # The problem arises when using split which changes shape/dimension
        # for different input.
        # Thus after initial tracing the shapes might not fit anymore.
        # However given that it transforms to while loop
        # anyway, we can also just use map_fn without any performance gains.
        # return tf.vectorized_map(self._generate_patches, combined)
        return tf.map_fn(self._generate_patches, combined)

    @tf.function
    def _fubini_study_n_potentials(self, points, t=tf.complex(1., 0.)):
        r"""Computes the Fubini-Study Kahler potential on a single projective
        ambient space factor specified by n.

        Args:
            points (tf.tensor([bSize, ncoords], tf.complex64)): Coordinates of
                the n-th projective spce.
           t (tf.complex, optional): Volume factor. Defaults to 1+0j.

        Returns:
            tf.tensor([bsize], tf.float32):
                FS-metric in the ambient space coordinates.
        """
        point_square = tf.math.reduce_sum(tf.math.abs(points)**2, axis=-1)
        return tf.cast(tf.math.real(t/self.pi), tf.float32) * tf.cast(tf.math.real(tf.math.log(point_square)), tf.float32)

    @tf.function
    def _fubini_study_n_metrics(self, points, n=None, t=tf.complex(1., 0.)):
        r"""Computes the Fubini-Study metric on a single projective
        ambient space factor specified by n.

        Args:
            points (tf.tensor([bSize, ncoords], tf.complex64)): Coordinates of
                the n-th projective spce.
            n (int, optional): Degree of P**n. Defaults to None(=self.ncoords).
            t (tf.complex, optional): Volume factor. Defaults to 1+0j.

        Returns:
            tf.tensor([bsize, ncoords, ncoords], tf.complex64): 
                FS-metric in the ambient space coordinates.
        """
        if n is None:
            n = self.ncoords
        point_square = tf.math.reduce_sum(tf.math.abs(points)**2, axis=-1)
        point_square = tf.cast(point_square, dtype=tf.complex64)
        point_diag = tf.einsum('x,ij->xij', point_square,
                               tf.cast(tf.eye(n), dtype=tf.complex64))
        outer = tf.einsum('xi,xj->xij', tf.math.conj(points), points)
        outer = tf.cast(outer, dtype=tf.complex64)
        gFS = tf.einsum('xij,x->xij', (point_diag - outer), point_square**-2)
        return gFS*t/self.pi

    @tf.function
    def _find_good_coord_mask(self, points):
        r"""Creates coordinate mask with x_a = True.

        NOTE:
            Legacy code. Currently not used anywhere. Remove?

        Args:
            points (tf.tensor([bSize, 2*ncoords], tf.float32)): Points.

        Returns:
            tf.tensor([bSize, nfold, ncoords], tf.bool): Good coord mask.
        """
        # creates coordinate mask with patch coordinates
        inv_one_mask = self._get_inv_one_mask(points)
        cpoints = tf.complex(points[:, :self.ncoords],
                             points[:, self.ncoords:])
        dQdz = self._compute_dQdz(cpoints)
        dQdz = dQdz*tf.cast(inv_one_mask, dtype=tf.complex64)
        indices = tf.argmax(tf.math.abs(dQdz), axis=-1)
        dQdz_mask = -1.*tf.one_hot(indices, self.ncoords)
        full_mask = tf.math.add(
            tf.cast(inv_one_mask, dtype=tf.float32), dQdz_mask)
        return tf.cast(full_mask, dtype=tf.bool)

    @tf.function
    def _compute_dQdz(self, points, k):
        r"""Computes dQdz at each point.

        Args:
            points (tf.tensor([bSize, ncoords], tf.complex)):
                         vector of coordinates
            k (int): k-th hypersurface

        Returns:
            tf.tensor([bSize, ncoords], tf.complex): dQdz at each point.
        """
        p_exp = tf.expand_dims(tf.expand_dims(points, 1), 1)
        dQdz = tf.math.pow(p_exp, self.BASIS['DQDZB'+str(k)])
        dQdz = tf.math.reduce_prod(dQdz, axis=-1)
        dQdz = tf.math.multiply(self.BASIS['DQDZF'+str(k)], dQdz)
        dQdz = tf.reduce_sum(dQdz, axis=-1)
        return dQdz

    @tf.function
    def _get_patch_coordinates(self, points, patch_mask):
        r"""Transforms the coordinates, such that they are in the patch
        given in patch_mask.
        """
        norm = tf.boolean_mask(points, patch_mask)
        norm = tf.reshape(norm, (-1, self.nProjective))
        # TODO: think about how to avoid loop and concat.
        full_norm = 1.
        for i in range(self.nProjective):
            degrees = tf.ones(self.degrees[i], dtype=tf.complex64)
            tmp_norm = tf.einsum('i,x->xi', degrees, norm[:, i])
            if i == 0:
                full_norm = tmp_norm
            else:
                full_norm = tf.concat((full_norm, tmp_norm), axis=-1)
        return points / full_norm

    @tf.function
    def compute_transition_loss(self, points):
        r"""Computes transition loss at each point.

        .. math::

            \mathcal{L} = \frac{1}{d} \sum_{k,j} 
                ||g^k - T_{jk} \cdot g^j T^\dagger_{jk}||_n

        Args:
            points (tf.tensor([bSize, 2*ncoords], tf.float32)): Points.

        Returns:
            tf.tensor([bSize], tf.float32): Transition loss at each point.
        """
        inv_one_mask = self._get_inv_one_mask(points)
        patch_indices = tf.where(~inv_one_mask)[:, 1]
        patch_indices = tf.reshape(patch_indices, (-1, self.nProjective))
        current_patch_mask = self._indices_to_mask(patch_indices)
        cpoints = tf.complex(points[:, :self.ncoords],
                             points[:, self.ncoords:])
        fixed = self._find_max_dQ_coords(points)
        if self.nhyper == 1:
            other_patches = tf.gather(self.fixed_patches, fixed)
        else:
            combined = tf.concat((fixed, patch_indices), axis=-1)
            other_patches = self._generate_patches_vec(combined)
        other_patches = tf.reshape(other_patches, (-1, self.nProjective))
        other_patch_mask = self._indices_to_mask(other_patches)
        # NOTE: This will include same to same patch transitions
        exp_points = tf.repeat(cpoints, self.nTransitions, axis=-2)
        patch_points = self._get_patch_coordinates(
            exp_points,
            tf.cast(other_patch_mask, dtype=tf.bool))
        fixed = tf.reshape(
            tf.tile(fixed, [1, self.nTransitions]), (-1, self.nhyper))
        real_patch_points = tf.concat(
            (tf.math.real(patch_points), tf.math.imag(patch_points)),
            axis=-1)
        gj = self(real_patch_points, training=True, j_elim=fixed)
        # NOTE: We will compute this twice.
        # TODO: disentangle this to save one computation?
        gi = tf.repeat(self(points), self.nTransitions, axis=0)
        current_patch_mask = tf.repeat(
            current_patch_mask, self.nTransitions, axis=0)
        Tij = self.get_transition_matrix(
            patch_points, other_patch_mask, current_patch_mask, fixed)
        all_t_loss = tf.math.abs(self.transition_loss_matrices(gj, gi, Tij))
        all_t_loss = tf.math.reduce_sum(all_t_loss**self.n[2], axis=[1, 2])
        # This should now be nTransitions 
        all_t_loss = tf.reshape(all_t_loss, (-1, self.nTransitions))
        all_t_loss = tf.math.reduce_sum(all_t_loss, axis=-1)
        return all_t_loss/(self.nTransitions*self.nfold**2)

    @tf.function
    def get_transition_matrix(self, points, i_mask, j_mask, fixed):
        r"""Computes transition matrix between patch i and j 
        for each point in points where fixed is the coordinate,
        which is being eliminated.

        Example (by hand):
            Consider the bicubic with:
            
            .. math:: 
            
                P_1^2 [a_0 : a_1 : a_2] \text{ and } P_2^2 [b_0 : b_1 : b_2]. 
            
            Assume we eliminate :math:`b_2` and keep it fixed. Then we 
            consider two patches.
            Patch 1 where :math:`a_0 = b_0 = 1` with new coordinates
            :math:`(x_1, x_2, x_3) = (a_1/a_0, a_2/a_0, b_1/b_0)`
            Patch 2 where :math:`a_1=b_1=1` with new coordinates
            :math:`(w_1, w_2, w_3) = (a_0/a_1, a_2/a_1, b_0/b_1)`
            such that we can reexpress w in terms of x:
            :math:`w_1(x)=1/x_1,\; w_2(x)=x_2/x_1,\; w_3(x)=1/x_3`
            from which follows:

            .. math::

                T_{11} &= \frac{\partial w_1}{\partial x_1} = 
                    -1/x_1^2 = -a_0^2/a_1^2 \\
                T_{12} &= \frac{\partial w_2}{\partial x_1} = 
                    -x_2/x_1^2 = -a_2 a_0/a_1^2 \\
                T_{13} &= \frac{\partial w_3}{\partial x_1} = 0 \\
                T_{21} &= \frac{\partial w_1}{\partial x_2} = 0 \\
                       & \dots

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
        p2 = tf.reshape(tf.where(j_mask_red)[:, 1], (-1, self.nProjective))

        # g1
        g1_mask = tf.reduce_sum(tf.one_hot(fixed_red, self.ncoords), axis=-2)
        g1_mask = g1_mask + i_mask_red
        g1_mask = ~tf.cast(g1_mask, dtype=tf.bool)
        g1_i = tf.where(g1_mask)
        g1_i = tf.reshape(g1_i[:, 1], (-1, self.nfold))

        # g2
        g2_mask = tf.reduce_sum(tf.one_hot(fixed_red, self.ncoords), axis=-2)
        g2_mask = g2_mask + j_mask_red
        g2_mask = ~tf.cast(g2_mask, dtype=tf.bool)
        g2_i = tf.where(g2_mask)
        g2_i = tf.reshape(g2_i[:, 1], (-1, self.nfold))

        # find proj indices
        proj_indices = tf.reshape(
            tf.tile(self._proj_indices, [n_p_red]),
            (-1, self.ncoords))
        g1_proj = tf.boolean_mask(proj_indices, g1_mask)
        g1_proj = tf.reshape(g1_proj, (-1, self.nfold))

        ratios = tf.reshape(
            tf.boolean_mask(points_red, i_mask_red) / tf.boolean_mask(points_red, j_mask_red),
            (-1, self.nProjective))
        tij_red = tf.zeros((n_p_red, self.nfold, self.nfold), 
                           dtype=tf.complex64)
        # fill the mixed ratio elements
        for j in range(self.nProjective):
            t_pos = tf.einsum('xi,xj->xij',
                              tf.cast(g1_i == p2[:, j:j+1], dtype=tf.int32),
                              tf.cast(g1_proj == j, dtype=tf.int32))
            t_indices = tf.where(tf.cast(t_pos, dtype=tf.bool))
            num_indices = tf.gather_nd(
                g2_i, tf.concat((t_indices[:, 0:1], t_indices[:, 2:3]), axis=-1))
            num_indices = tf.concat(
                (t_indices[:, 0:1], tf.reshape(num_indices, (-1, 1))), axis=-1)
            num_tpos = tf.gather_nd(points_red, num_indices)
            ratio_indices = num_indices[:, 0]  # match the x-axis indices
            ratio_tpos = tf.gather(ratios[:, j], ratio_indices)
            denom_indices = p2[:, j:j+1]
            denom_indices = tf.concat(
                (tf.reshape(tf.range(n_p_red), (-1, 1)), denom_indices), axis=-1)
            denom_tpos = tf.gather_nd(points_red, denom_indices)
            denom_tpos = tf.gather(denom_tpos, ratio_indices)
            t_values = -1.*num_tpos*ratio_tpos/denom_tpos
            # update tij
            tij_red = tf.tensor_scatter_nd_update(
                tij_red, t_indices, t_values)
        # fill the single ratio elements
        c_pos = tf.where(tf.reshape(g1_i, (-1, 1, self.nfold)) == tf.reshape(g2_i, (-1, self.nfold, 1)))
        c_indices = tf.gather_nd(g1_proj, c_pos[:, 0:2])
        c_indices = tf.concat(
            (c_pos[:, 0:1], tf.reshape(c_indices, (-1, 1))), axis=-1)
        c_values = tf.gather_nd(ratios, c_indices)
        # need to switch cols, either here or before
        c_pos = tf.concat((c_pos[:, 0:1], c_pos[:, 2:3], c_pos[:, 1:2]), axis=-1)
        tij_red = tf.tensor_scatter_nd_update(tij_red, c_pos, c_values)
        # fill tij
        tij_eye = tf.eye(self.nfold, batch_shape=[n_p-n_p_red], dtype=tf.complex64)
        tij_all = tf.zeros((n_p, self.nfold, self.nfold), dtype=tf.complex64)
        tij_all = tf.tensor_scatter_nd_update(
            tij_all, tf.reshape(diff_patch, (-1, 1)), tij_red)
        tij_all = tf.tensor_scatter_nd_update(
            tij_all, tf.reshape(same_patch, (-1, 1)), tij_eye)
        return tij_all

    @tf.function
    def transition_loss_matrices(self, gj, gi, Tij):
        r"""Computes transition loss matrix between metric
        in patches i and j with transition matrix Tij.

        Args:
            gj (tf.tensor([bSize, nfold, nfold], tf.complex64)):
                Metric in patch j.
            gi (tf.tensor([bSize, nfold, nfold], tf.complex64)):
                Metric in patch i.
            Tij (tf.tensor([bSize, nfold, nfold], tf.complex64)):
                Transition matrix from patch i to patch j.

        Returns:
            tf.tensor([bSize, nfold, nfold], tf.complex64): 
                .. math::`g_j - T^{ij} g_i T^{ij,\dagger}`
        """
        return gj - tf.einsum('xij,xjk,xkl->xil', Tij, gi,
                              tf.transpose(Tij, perm=[0, 2, 1], conjugate=True))

    @tf.function
    def compute_ricci_scalar(self, points, pb=None):
        r"""Computes the Ricci scalar for each point.

        .. math::

            R = g^{ij} J_i^a \bar{J}_j^b \partial_a \bar{\partial}_b 
                \log \det g

        Args:
            points (tf.tensor([bSize, 2*ncoords], tf.float)): Points.
            pb (tf.tensor([bSize, nfold, ncoords], tf.float), optional):
                Pullback tensor at each point. Defaults to None.

        Returns:
            tf.tensor([bSize], tf.float): R|_p.
        """
        x_vars = points
        # take derivatives
        with tf.GradientTape(persistent=False) as tape1:
            tape1.watch(x_vars)
            with tf.GradientTape(persistent=False) as tape2:
                tape2.watch(x_vars)
                # training = false for batch_jacobian
                prediction = self(x_vars, training=False)
                det = tf.math.real(tf.linalg.det(prediction))
                # * factorial / (2**nfold)
                log = tf.math.log(det)
            di_dg = tape2.gradient(log, x_vars)
        didj_dg = tf.cast(tape1.batch_jacobian(di_dg, x_vars),
                          dtype=tf.complex64)
        # add derivatives together to complex tensor
        ricci_ij = didj_dg[:, 0:self.ncoords, 0:self.ncoords]
        ricci_ij += 1j*didj_dg[:, 0:self.ncoords, self.ncoords:]
        ricci_ij -= 1j*didj_dg[:, self.ncoords:, 0:self.ncoords]
        ricci_ij += didj_dg[:, self.ncoords:, self.ncoords:]
        ricci_ij *= 0.25
        pred_inv = tf.linalg.inv(prediction)
        if pb is None:
            pullbacks = self.pullbacks(points)
        else:
            pullbacks = pb
        ricci_scalar = tf.einsum('xba,xai,xij,xbj->x', pred_inv, pullbacks,
                                 ricci_ij, tf.math.conj(pullbacks))
        ricci_scalar = tf.math.real(ricci_scalar)
        return ricci_scalar

    @tf.function
    def compute_ricci_loss(self, points, pb=None):
        r"""Computes the absolute value of the Ricci scalar for each point. Since negative
        Ricci scalars are bad, we take a loss of \|1-e^-ricci\|^p. This will exponentially
        punish negative Ricci scalars, and it vanishes for Ricci scalar 0

        .. seealso:: method :py:meth:`.compute_ricci_scalar`.

        Args:
            points (tf.tensor([bSize, 2*ncoords], tf.float)): Points.
            pb (tf.tensor([bSize, nfold, ncoords], tf.float), optional):
                Pullback tensor at each point. Defaults to None.

        Returns:
            tf.tensor([bSize], tf.float): \|R\|_n.
        """
        ricci_scalar = self.compute_ricci_scalar(points, pb)
        
        return tf.math.abs(1-tf.math.exp(-ricci_scalar))
