""" 
Implementation of the Donaldson algorithm using numpy.
"""
import numpy as np
from joblib import Parallel, delayed
from cymetric.pointgen.nphelper import generate_monomials
from cymetric.models.fubinistudy import FSModel
import os as os
from scipy.special import factorial
import sympy as sp
from sympy.geometry.util import idiff
import logging
import sys as sys
import tensorflow as tf
tfk = tf.keras

logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s')
logger = logging.getLogger('Donaldson')


class Donaldson:

    def __init__(self, pointgen, k=None, pw=[], log=3):

        self.pointgen = pointgen
        if log == 1:
            level = logging.DEBUG
        elif log == 2:
            level = logging.INFO
        else:
            level = logging.WARNING
        logger.setLevel(level=level)
        if k is not None:
            self._init_monomials(k)
            self.hbalanced = self.compute_hbalanced(k, pw)
        else:
            self.k = [0]
            self.sections = None
            self.jacobians = None
            self.hessians = None
        self.dzdzdz_basis = None

    def __call__(self, points):
        return self.g_pull_backs(points)

    def _init_monomials(self, k):
        self.k = [k for _ in range(len(self.pointgen.ambient))]
        self._generate_sections(self.k)
        self._generate_jacobians()
        self._generate_hessians()

    def set_hbalanced(self, hb, k):
        self.hbalanced = hb
        self._init_monomials(k)

    def compute_hbalanced(self, k, point_weights=[], max_iterations=10,
                          n_proc=-1, n_chunks=100):
        r"""Donaldson algorithm to compute hbalanced.

        Args:
            k (int): degree k of the line bundle
            point_weights (list, optional): point weights. Defaults to [].
            max_iterations (int, optional): # of inverses being taken. Defaults to 15.
            n_proc (int, optional): # of cores being used. Defaults to -1.
            n_chunks (int, optional): # chunks in toperator. Defaults to 100.

        Returns:
            ndarray[np.complex]: hbalanced matrix
        """
        # TODO: Make it so we can use mixed degrees for products?
        if self.jacobians is None or k != np.mean(self.k):
            self._init_monomials(k)
        if len(point_weights) == 0:
            # generate number of points which we can chunk nicely later
            n_pw = int(self._needed_points(self.nsections)/n_chunks-1)*n_chunks
            logger.info('Generating {} point weights.'.format(n_pw))
            point_weights = self.pointgen.generate_point_weights(n_pw)
            logger.info('Point weights generated.')
        else:
            # chunk it
            n_pw = int(len(point_weights)/(n_chunks-1))*n_chunks
            point_weights = point_weights[:n_pw]
            if n_pw < self._needed_points(self.nsections):
                logger.warning('Too little point weights {} < {} \
                                (needed for numerical stability).'.format(
                    n_pw, self._needed_points(self.nsections)))
        volume_cy = (1/n_pw) * np.sum(point_weights['weight'])
        h_balanced_new = self._initial_hbalanced(self.nsections)
        logger.info(
            'Applying T-operator for {} iterations'.format(max_iterations))
        # apply t_operator
        for i in range(max_iterations):
            # TODO: vectorize this
            top = np.sum(Parallel(n_jobs=n_proc)(delayed(self.t_operator_vec)(chunks, h_balanced_new)
                                                 for chunks in point_weights.reshape((n_chunks, -1))), axis=0)
            h_balanced = (self.nsections / (n_pw * volume_cy)) * top
            h_balanced = np.linalg.inv(h_balanced)
            h_balanced = np.transpose(h_balanced)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('{}-th iteration with relative change {}.'.format(i,
                                                                               np.sum(np.abs(h_balanced-h_balanced_new))/np.sum(np.abs(h_balanced))))
                logger.debug('isHermitian: {}. isDense: {}.'.format(
                    np.allclose(h_balanced, np.conjugate(h_balanced.T)),
                    np.sum(np.abs(h_balanced))/(self.nsections**2)))
                logger.debug('is invertible: {}.'.format(np.all(np.isclose(np.einsum('ij,jk', h_balanced,
                             np.linalg.inv(h_balanced)), np.eye(self.nsections, dtype=np.complex), atol=1e-10))))
            h_balanced_new = np.copy(h_balanced)
        return h_balanced_new

    def _initial_hbalanced(self, nsections, permut=True, nattempts=10, atol=1e-12):
        # use permutation around diagonal
        if not permut:
            return np.eye(nsections, dtype=np.complex128)

        # 10 attempts? don't want to get stuck in infinite loop
        for _ in range(nattempts):
            h = np.random.randn(nsections, nsections) + \
                1j*np.random.randn(nsections, nsections)
            h = np.triu(h, 1) + np.conjugate(np.triu(h, 1)).T + \
                np.eye(nsections)*np.random.randn(nsections)
            # check if invertible
            h = np.eye(len(h), dtype=np.complex128)+0.1*h
            if np.all(np.isclose(np.einsum('ij,jk', h, np.linalg.inv(h)),
                                 np.eye(nsections, dtype=np.complex), atol=atol)):
                return h.astype(np.complex128)

        logger.warning('Unable to find initial \
                        invertible matrix in {} attempts. \
                        Returning Identity.'.format(nattempts))
        return np.eye(nsections, dtype=np.complex128)

    def _needed_points(self, nsections):
        #2.22 in 1910.08605
        return 10 * nsections**2 + 50000

    def _generate_sections(self, k):
        # TODO: This function is a mess. numpyze it.
        self.sections = None
        ambient_polys = [0 for i in range(len(k))]
        for i in range(len(k)):
            # create all monomials of degree k in ambient space factors
            ambient_polys[i] = list(generate_monomials(
                self.pointgen.degrees[i], k[i]))
        # create all combinations for product of projective spaces
        monomial_basis = [x for x in ambient_polys[0]]
        for i in range(1, len(k)):
            lenB = len(monomial_basis)
            monomial_basis = monomial_basis*len(ambient_polys[i])
            for l in range(len(ambient_polys[i])):
                for j in range(lenB):
                    monomial_basis[l*lenB+j] = monomial_basis[l *
                                                              lenB+j]+ambient_polys[i][l]
        sections = np.array(monomial_basis, dtype=np.int32)
        # reduce sections; pick (arbitrary) first monomial in point gen
        reduced = np.unique(
            np.where(sections - self.pointgen.monomials[0] < -0.1)[0])
        self.sections = sections[reduced]
        self.nsections = len(self.sections)
        # Sanity check; M is (py)CICY object
        # if self.nsections != np.round(self.M.line_co_euler(k)):
        #    logger.warning('Reduced basis {} is not fully reduced {}.'.format(
        #                                self.nsections, self.M.line_co_euler(k)))

    # vectorized
    def g_pull_backs(self, points, h=None):
        if h is None:
            h = self.hbalanced
        pbs = self.pointgen.pullbacks(points)
        g_kaehler = self.kaehler_metrics(h, points)
        return np.einsum('xai,xij,xbj->xab', pbs, g_kaehler, np.conjugate(pbs))

    # at single point
    def g_pull_back(self, h, point):
        jac = self.pointgen.pullback_tensor(point)
        g_kaehler = self.kaehler_metric(h, point)
        return np.einsum('ai,ij,bj', jac, g_kaehler, np.conjugate(jac))

    def kaehler_metrics(self, h, points):
        s_ps = self.eval_sections_vec(points)
        partial_sps = self.eval_jacobians_vec(points)
        k_0 = np.real(1 / np.einsum('ij,xi,xj->x',
                      h, s_ps, np.conjugate(s_ps)))
        k_1 = np.einsum('ab,xai,xb->xi', h, partial_sps, np.conjugate(s_ps))
        k_1_bar = np.conjugate(k_1)
        k_2 = np.einsum('ab,xai,xbj->xij', h, partial_sps,
                        np.conjugate(partial_sps))
        return (np.einsum('x,xij->xij', k_0, k_2) -
                np.einsum('x,xi,xj->xij', k_0 ** 2, k_1, k_1_bar))/(np.mean(self.k) * np.pi)

    def load_hbalanced(self, fname, k):
        self.hbalanced = np.load(fname, allow_pickle=True)
        self._init_monomials(k)

    def save_dzdzdz_basis(self, dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        fname = os.path.join(dirname, 'dbasis.npz')
        np.savez_compressed(fname,
                            DZDZDZB=self.dzdzdz_basis,
                            DZDZDZF=self.dzdzdz_factor,
                            )

    def load_dzdzdz_basis(self, fname):
        dbasis = np.load(fname, allow_pickle=True)
        self.dzdzdz_basis = dbasis['DZDZDZB']
        self.dzdzdz_factor = dbasis['DZDZDZF']

    def save_hbalanced(self, dirname):
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        fname = os.path.join(
            dirname, 'k'+str(int(np.mean(self.k)))+'hbalanced.npz')
        np.savez_compressed(fname,
                            hb=self.hbalanced,
                            k=self.k,
                            sections=self.sections,
                            jacobians=self.jacobians,
                            j_factors=self.j_factors
                            )

    def load_hbalanced_dict(self, fname):
        df = np.load(fname, allow_pickle=True)
        self.hbalanced = df['hb']
        self.k = df['k']
        self.sections = df['sections']
        self.jacobians = df['jacobians']
        self.j_factors = df['j_factors']

    def eval_sections_vec(self, points):
        return np.multiply.reduce(np.power(np.expand_dims(points, 1),
                                           self.sections), axis=-1)

    def eval_sections(self, point):
        return np.multiply.reduce(np.power(point, self.sections), axis=-1)

    def t_operator_vec(self, point_weights, hn):
        s_ps = self.eval_sections_vec(point_weights['point'])
        inner_products = np.real(
            np.einsum('ij, xi, xj -> x', hn, s_ps, np.conjugate(s_ps)))
        all_products = np.einsum('xij,x -> ij', np.einsum('xi,xj->xij', s_ps, np.conjugate(s_ps)),
                                 point_weights['weight'] / inner_products)
        #t_op = np.add.reduce(all_products)
        return all_products

    def t_operator_single(self, pw, hn):
        s_p = self.eval_sections(pw['point'])
        inner_product = np.real(
            np.einsum('ij, i, j', hn, s_p, np.conjugate(s_p)))
        return np.einsum('i,j', s_p, np.conjugate(s_p))*pw['weight'] / inner_product

    def eval_jacobians(self, point):
        return np.multiply.reduce(np.power(point,
                                           self.jacobians), axis=-1) * self.j_factors[0]

    def eval_jacobians_vec(self, points):
        return np.multiply.reduce(np.power(np.expand_dims(np.expand_dims(points, 1), 1),
                                           self.jacobians), axis=-1) * self.j_factors

    def eval_hessians(self, point):
        return np.multiply.reduce(np.power(point,
                                           self.hessians), axis=-1) * self.h_factors[0]

    def eval_hessians_vec(self, points):
        # TODO: I think points needs to be expanded along more dims.
        return np.multiply.reduce(np.power(np.expand_dims(points, 1),
                                           self.hessians), axis=-1) * self.h_factors

    def _generate_jacobians(self):
        # check which ones are good
        self.jacobians = np.expand_dims(self.sections, 1) - \
            np.eye(self.pointgen.n_coords, dtype=np.int)
        self.j_factors = np.expand_dims(self.jacobians.diagonal(0, 1, 2)+1, 0)
        mask = np.any(self.jacobians < 0, axis=-1)
        self.jacobians[mask] = np.zeros(
            (np.sum(mask), self.pointgen.n_coords), dtype=np.int)
        self.j_factors[0, mask] = np.zeros(np.sum(mask), dtype=np.int)
        # TODO: remove zeros for faster performance?
        # but then we can no longer vectorize with numpy

    def _generate_hessians(self):
        self.hessians = np.expand_dims(self.jacobians, 2) - \
            np.eye(self.pointgen.n_coords, dtype=np.int)
        self.h_factors = (self.hessians.diagonal(0, 2, 3)+1) * \
            np.expand_dims(self.j_factors[0], axis=-1)
        self.h_factors = np.expand_dims(self.h_factors, 0)
        mask = np.any(self.hessians < 0, axis=-1)
        self.hessians[mask] = np.zeros(
            (np.sum(mask), self.pointgen.n_coords), dtype=np.int)
        self.h_factors[0, mask] = np.zeros(np.sum(mask), dtype=np.int)
        # TODO: remove zeros for faster performance?
        # but then we can no longer vectorize with numpy

    def sigma_measure(self, h=None, k=None, point_weights=[]):

        if k is not None:
            self._init_monomials(k)
        if h is None:
            h = self.hbalanced
        # confirm that k and hbalanced match
        assert len(self.sections) == len(
            h), "dimensions of k and h don't match"

        if len(point_weights) == 0:
            n_t = 10000
            point_weights = self.pointgen.generate_point_weights(
                n_t, omega=True)
        else:
            n_t = len(point_weights)
            if n_t < 10000:
                logger.warning('It is recommended to use 10000 points.')
        logger.info('Computing sigma measure for {} points.'.format(n_t))

        volume_cy = np.mean(point_weights['weight'])
        omega_wedge_omega = np.real(
            point_weights['omega'] * np.conj(point_weights['omega']))
        det = np.linalg.det(self(point_weights['point']))
        det = np.real(det)*factorial(self.pointgen.nfold) / \
            (2**self.pointgen.nfold)
        vol_k = np.mean(det*point_weights['weight']/omega_wedge_omega)
        ratio = volume_cy/np.real(vol_k)
        logger.info(
            'CY-volume: {}, K-vol: {}, ratio: {}.'.format(volume_cy, vol_k, ratio))

        sigma_integrand = np.abs(
            np.ones(n_t) - ratio * det/omega_wedge_omega) * point_weights['weight']
        sigma = np.mean(sigma_integrand) / volume_cy
        logger.info('Sigma measure: {}.'.format(sigma))
        return sigma

    def ricci_measure(self, h=None, k=None, point_weights=[]):
        # maybe make a prepare function? to not repeat this everytime
        # have to generate sections and such
        if k is not None:
            self._init_monomials(k)
        if h is None:
            h = self.hbalanced
        if self.dzdzdz_basis is None:
            self._generate_dzdzdz_basis()
        assert len(self.sections) == len(
            h), "dimensions of k and h don't match"

        if len(point_weights) == 0:
            n_t = 10000
            point_weights = self.pointgen.generate_point_weights(n_t)
        else:
            n_t = len(point_weights)
            if n_t < 10000:
                logger.warning('It is recommended to use 10000 points.')
        logger.info('Computing Ricci measure for {} points.'.format(n_t))

        volume_cy = np.mean(point_weights['weight'])
        omega = np.array([self.pointgen.Omega(p)
                         for p in point_weights['point']])
        omega_wedge_omega = np.real(omega * np.conj(omega))
        det = np.linalg.det(self(point_weights['point']))
        det = np.real(det)*factorial(self.pointgen.nfold) / \
            (2**self.pointgen.nfold)
        vol_k = np.mean(det*point_weights['weight']/omega_wedge_omega)
        ratio = volume_cy/np.real(vol_k)
        logger.info(
            'CY-volume: {}, K-vol: {}, ratio: {}.'.format(volume_cy, vol_k, ratio))

        ricci = np.array(Parallel(n_jobs=-1, backend='multiprocessing',
                                  batch_size=500)(delayed(self.ricci_trace)(h, p) for p in point_weights['point']))
        ricci_measure = (vol_k**(1/self.pointgen.nfold)) * np.mean(np.abs(ricci) *
                                                                   point_weights['weight'] * det/omega_wedge_omega)/volume_cy
        logger.info('Ricci measure: {}. Mean abs(R): {}.'.format(ricci_measure,
                                                                 np.mean(np.abs(ricci))))
        return ricci_measure

    # TODO: Rewrite in terms of proper basis.
    def _generate_dzdzdz_basis(self):
        # take one more implicit derivative
        self.dzdzdz_basis = [[[0 for _ in range(self.pointgen.n_coords)]
                              for _ in range(self.pointgen.n_coords)] for _ in range(self.pointgen.n_coords)]
        self.dzdzdz_factor = [[[0 for _ in range(self.pointgen.n_coords)]
                               for _ in range(self.pointgen.n_coords)] for _ in range(self.pointgen.n_coords)]

        self.iiderivatives = [[Parallel(n_jobs=-1, backend='multiprocessing')
                               (delayed(self.second_idiff)(i, j, k)
                                for i in range(self.pointgen.n_coords))
                               for j in range(self.pointgen.n_coords)]
                              for k in range(self.pointgen.n_coords)]
        for k in range(self.pointgen.n_coords):
            for j in range(self.pointgen.n_coords):
                for i in range(self.pointgen.n_coords):
                    if i != j and i != k:
                        self.dzdzdz_basis[k][j][i], self.dzdzdz_factor[k][j][i] = self.pointgen._frac_to_monomials(
                            self.iiderivatives[k][j][i])

    def second_idiff(self, i, j, k):
        return self._take_2nd_implicit_deriv(self.pointgen.poly, self.pointgen.x[i],
                                             self.pointgen.x[j], self.pointgen.x[k]) if i != j and i != k else 0

    def _take_2nd_implicit_deriv(self, eq, z1, z2, z3):
        # check if z2 and z3 are the same
        if z2 == z3:
            return idiff(eq, z1, z2, n=2)

        # solve manually
        dep = {z1}
        f = {s: sp.Function(s.name)(z2, z3) for s in eq.free_symbols
             if s != z2 and s != z3 and s in dep}

        dz1dz2 = sp.Function(z1.name)(z2, z3).diff(z2)
        dz1dz3 = sp.Function(z1.name)(z2, z3).diff(z3)
        dzij = sp.Function(z1.name)(z2, z3).diff(z2).diff(z3)
        eq = eq.subs(f)

        derivs = {}

        d2 = sp.solve(eq.diff(z2), dz1dz2)[0]
        d3 = sp.solve(eq.diff(z3), dz1dz3)[0]
        derivs[dz1dz2] = d2
        derivs[dz1dz3] = d3

        zij = sp.solve(eq.diff(z2).diff(z3), dzij)[0].subs(derivs)

        return zij.subs([(v, k) for k, v in f.items()])

    def compute_dzdzdz(self, point, zj, zi, zk):
        # compute dzj/(dzk dzi)
        numerator = np.sum(self.dzdzdz_factor[zk][zi][zj][0] *
                           np.multiply.reduce(np.power(point, self.dzdzdz_basis[zk][zi][zj][0]), axis=-1))
        denominator = np.sum(self.dzdzdz_factor[zk][zi][zj][1] *
                             np.multiply.reduce(np.power(point, self.dzdzdz_basis[zk][zi][zj][1]), axis=-1))
        return numerator/denominator

    def ricci_trace(self, h, point):
        # take trace wrt to pullback
        g = self.g_pull_back(h, point)
        g_inv = np.linalg.inv(g)
        ricci_tensor = self.ricci_tensor(h, point)
        return np.einsum('ba, ab', g_inv, ricci_tensor).real

    # TODO: Vectorize this
    def ricci_tensor(self, h, point):
        # We compute B. 75

        kaehler_terms = self.kaehler_terms(h, point)
        # a bit of unnecessary computation here
        gt = self.kaehler_metric(h, point)
        di_gt = self.partiali_g(kaehler_terms)
        dij_gt = self.partialij_g(kaehler_terms)
        J = self.pointgen.pullback_tensor(point)
        diJ = self.partial_pullback_tensor(point)

        # we need B.76
        di_g = np.einsum('iaj,jk,bk -> iab', diJ, gt, np.conjugate(J)) + \
            np.einsum('aj,ijk,bk -> iab', J, di_gt, np.conjugate(J))
        # we need B.77
        dij_g = np.einsum('ial,ljk,bk -> ijab', diJ, np.conjugate(di_gt), np.conjugate(J)) + \
            np.einsum('al,ijlk,bk -> ijab', J, dij_gt, np.conjugate(J)) + \
            np.einsum('ial,lk,jbk -> ijab', diJ, gt, np.conjugate(diJ)) + \
            np.einsum('al,ikl,jbk -> ijab', J, di_gt, np.conjugate(diJ))
        ginv = np.linalg.inv(self.g_pull_back(h, point))
        # B.75, take trace wrt to g or
        # take regular trace?
        ricci_tensor = np.einsum('ba,ijab -> ij', np.eye(3), (-1 * np.einsum('ab,ibc,cd,jed -> ijae', ginv, di_g, ginv, np.conjugate(di_g)) +
                                                              np.einsum('ab,ijbc -> ijac', ginv, dij_g)))
        # B.72 - we return
        ricci_tensor = np.einsum(
            'ai,bj,ij -> ab', J, np.conjugate(J), ricci_tensor)
        return ricci_tensor

    def kaehler_metric(self, h, point):
        # save the results? so we do not need to recompute everything for each point?
        s_p = self.eval_sections(point)
        partial_sp = self.eval_jacobians(point)
        k_0 = np.real(1 / np.einsum('ij,i,j', h, s_p, np.conjugate(s_p)))
        k_1 = np.einsum('ab,ai,b', h, partial_sp, np.conjugate(s_p))
        k_1_bar = np.conjugate(k_1)
        k_2 = np.einsum('ab,ai,bj', h, partial_sp, np.conjugate(partial_sp))
        return (k_0 * k_2 - (k_0 ** 2) * np.einsum('i,j', k_1, k_1_bar))/(np.mean(self.k) * np.pi)

    def kaehler_terms(self, h, point):
        # compute B.67-B.70, B.79-B.80, B.82-B.83
        s_p = self.eval_sections(point)
        partial_sp = self.eval_jacobians(point)
        double_partial_sp = self.eval_hessians(point)
        k_00 = np.real(1 / np.einsum('ij,i,j', h, s_p, np.conjugate(s_p)))
        k_10 = np.einsum('ab,ai,b', h, partial_sp, np.conjugate(s_p))
        k_20 = np.einsum('ab,aij,b', h, double_partial_sp, np.conjugate(s_p))
        k_11 = np.einsum('ab,ai,bj', h, partial_sp, np.conjugate(partial_sp))
        k_21 = np.einsum('ab,aik,bl -> ikl', h,
                         double_partial_sp, np.conjugate(partial_sp))
        k_22 = np.einsum('ab,aik,bjl -> ijkl', h,
                         double_partial_sp, np.conjugate(double_partial_sp))
        return [k_00, k_10, k_20, k_11, k_21, k_22]

    def partial_pullback_tensor(self, point):
        # J_aj^i = dz_i/(dx_a dz_j)
        zi = self.pointgen._find_max_dQ_coord(point)
        diag_i = np.where(self.pointgen._find_good_coordinate_mask(point))[0]
        J_aji = np.zeros((self.pointgen.n_coords, self.pointgen.nfold,
                          self.pointgen.n_coords), dtype=np.complex128)
        for j in range(self.pointgen.n_coords):
            for a in range(self.pointgen.nfold):
                if j != zi:
                    J_aji[j][a][zi] = self.compute_dzdzdz(
                        point, zi, diag_i[a], j)
        return J_aji

    def partiali_g(self, k):
        # B.78
        # g_ikl
        di_g = - k[0]**2 * (np.einsum('i,kl -> ikl', k[1], k[3]) +
                            np.einsum('k, il -> ikl', k[1], k[3]) +
                            np.einsum('l, ik -> ikl', np.conjugate(k[1]), k[2])) + \
            k[0] * k[4] + 2 * k[0]**3 * \
            np.einsum('i,k,l -> ikl', k[1], k[1], np.conjugate(k[1]))

        return di_g / (np.pi * np.mean(self.k))

    def partialij_g(self, k):
        # B.81
        # g_ijkl
        dij_g = k[0] * k[5] - k[0]**2 * (np.einsum('ij,kl -> ijkl', k[3], k[3]) +
                                         np.einsum('ik,jl -> ijkl', k[2], np.conjugate(k[2])) +
                                         np.einsum('kj,il -> ijkl', k[3], k[3]) +
                                         # the next four become hermitian together
                                         np.einsum('j,ikl -> ijkl', np.conjugate(k[1]), k[4]) +
                                         np.einsum('l,ikj -> ijkl', np.conjugate(k[1]), k[4]) +
                                         np.einsum('i,ljk -> ijkl', k[1], np.conjugate(k[4])) +
                                         np.einsum('k,jli -> ijkl',
                                                   k[1], np.conjugate(k[4]))
                                         ) + \
            2 * k[0]**3 * (np.einsum('i,j,kl -> ijkl', k[1], np.conjugate(k[1]), k[3]) +
                           np.einsum('ij,k,l -> ijkl', k[3], k[1], np.conjugate(k[1])) +
                           # same for next four
                           np.einsum('j,k,il -> ijkl', np.conjugate(k[1]), k[1], k[3]) +
                           np.einsum('i,kj,l -> ijkl', k[1], k[3], np.conjugate(k[1])) +
                           np.einsum('i,k,jl -> ijkl', k[1], k[1], np.conjugate(k[2])) +
                           np.einsum(
                'j,ik,l -> ijkl', np.conjugate(k[1]), k[2], np.conjugate(k[1]))
        ) - \
            6 * k[0]**4 * np.einsum('i,j,k,l -> ijkl', k[1],
                                    np.conjugate(k[1]), k[1], np.conjugate(k[1]))

        return dij_g / (np.pi * np.mean(self.k))


class HbalancedModel(FSModel):
    r"""MatrixFSModelToric inherits :py:class:`cymetric.fsmodel.FSModel`.

    Computes the metric with tensorflow gradienttapes from a hbalanced metric.
    Require hbalanced tensor from Donaldson algorithm.

    NOTE:
        - This one has not been tested extensively. Use with caution.

        - If one were to implement a training step, one could learn h_b directly.
    """
    def __init__(self, hb, k, sections, jacobians, j_factors,
                 BASIS, **kwargs):
        super(HbalancedModel, self).__init__(
            BASIS=BASIS, **kwargs)
        self.hbalanced = tf.cast(hb, dtype=tf.complex64)
        self.sections = tf.cast(sections, dtype=tf.complex64)
        self.jacobians = tf.cast(jacobians, dtype=tf.complex64)
        self.j_factors = tf.cast(j_factors, dtype=tf.complex64)
        self.k = tf.cast(k, dtype=tf.complex64)

    def call(self, input_tensor, training=True):
        return self.g_pull_backs(input_tensor)

    @tf.function
    def g_pull_backs(self, points):
        cpoints = tf.complex(points[:, :self.ncoords],
                             points[:, self.ncoords:])
        pbs = self.pullbacks(points)
        g_kaehler = self.kaehler_metrics(cpoints)
        return tf.einsum('xai,xij,xbj->xab', pbs, g_kaehler, tf.math.conj(pbs))

    @tf.function
    def kaehler_metrics(self, points):
        s_ps = self.eval_sections_vec(points)
        partial_sps = self.eval_jacobians_vec(points)
        k_0 = 1. / tf.einsum('ij,xi,xj->x', self.hbalanced,
                             s_ps, tf.math.conj(s_ps))
        k_1 = tf.einsum('ab,xai,xb->xi', self.hbalanced,
                        partial_sps, tf.math.conj(s_ps))
        k_1_bar = tf.math.conj(k_1)
        k_2 = tf.einsum('ab,xai,xbj->xij', self.hbalanced,
                        partial_sps, tf.math.conj(partial_sps))
        k_02 = tf.einsum('x, xij -> xij', k_0, k_2)
        k_011 = tf.einsum('x,xij->xij', tf.square(k_0),
                          tf.einsum('xi,xj->xij', k_1, k_1_bar))
        return (k_02 - k_011) / (self.k * self.pi)

    @tf.function
    def eval_jacobians_vec(self, points):
        return tf.math.reduce_prod(tf.math.pow(tf.expand_dims(
            tf.expand_dims(points, 1), 1),
            self.jacobians), axis=-1) * self.j_factors

    @tf.function
    def eval_sections_vec(self, points):
        return tf.math.reduce_prod(tf.math.pow(tf.expand_dims(points, 1),
                                               self.sections), axis=-1)

    def compute_ricci_measure(self, points, y, bSize=1000, verbose=1):
        # simple optimizable hack to compute batched
        # ricci mesaure
        weights = y[:, -2]
        omegas = y[:, -1]
        ricci_scalars = tf.zeros_like(points[:, 0])
        for i in range(len(points)//bSize):
            tmp_ricci = self.compute_ricci_scalar(
                points[i*bSize:(i+1)*bSize]
            )
            tmp_ricci = tf.math.abs(tmp_ricci)
            ricci_scalars = tf.tensor_scatter_nd_update(
                ricci_scalars,
                tf.reshape(tf.range(i*bSize, (i+1)*bSize), [-1, 1]),
                tmp_ricci
            )
        tmp_ricci = self.compute_ricci_scalar(
            points[(len(points)//bSize)*bSize:]
        )
        tmp_ricci = tf.math.abs(tmp_ricci)
        ricci_scalars = tf.tensor_scatter_nd_update(
            ricci_scalars,
            tf.reshape(tf.range((len(points)//bSize)
                       * bSize, len(points)), [-1, 1]),
            tmp_ricci
        )
        det = tf.math.real(tf.linalg.det(self(points)))
        nfold = tf.cast(self.nfold, dtype=tf.float32)
        factorial = tf.exp(tf.math.lgamma(nfold+1))
        det = det * factorial / (2**nfold)
        det_over_omega = det / omegas
        volume_cy = tf.math.reduce_mean(weights, axis=-1)
        vol_k = tf.math.reduce_mean(det_over_omega * weights, axis=-1)
        if verbose:
            tf.print('Mean abs(R) = ',
                     tf.math.reduce_mean(tf.abs(ricci_scalars)),
                     output_stream=sys.stdout)
        ricci_measure = (vol_k**(1/nfold) / volume_cy) * \
            tf.math.reduce_mean(
                det_over_omega * ricci_scalars * weights, axis=-1)
        return ricci_measure
