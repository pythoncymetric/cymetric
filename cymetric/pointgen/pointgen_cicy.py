"""
CICY PointGenerator. 

:Authors:
    Fabian Ruehle f.ruehle@northeastern.edu
"""

import numpy as np
import logging
import sympy as sp
from sympy.geometry.util import idiff
import scipy.optimize as opt
from joblib import Parallel, delayed
from cymetric.pointgen.pointgen import PointGenerator
from cymetric.pointgen.nphelper import generate_monomials, prepare_dataset, get_levicivita_tensor

logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s')
logger = logging.getLogger('CICYpointgen')


class CICYPointGenerator(PointGenerator):
    r"""CICYPointGenerator class.

    The numerics are entirely done in numpy; sympy is used for 
    computing the initial derivatives.

    Use this module if you want to generate points and data on a CY given by
    more than one hypersurface.
    
    Example:
        A simple example on a generic CY manifold of the family defined
        by the following configuration matrix:
        
        .. math::
            X \in [5|33]

        can be set up with

        >>> import numpy as np
        >>> from cymetric.pointgen.pointgen_cicy import CICYPointGenerator
        >>> from cymetric.pointgen.nphelper import generate_monomials
        >>> monomials = np.array(list(generate_monomials(6, 3)))
        >>> monomials_per_hyper = [monomials, monomials]
        >>> coeff = [np.random.randn(len(m)) for m in monomials_per_hyper]
        >>> kmoduli = np.ones(1)
        >>> ambient = np.array([5])
        >>> pg = CICYPointGenerator(monomials_per_hyper, coeff, 
        ...                         kmoduli, ambient)
        
        Once the CICYPointGenerator is initialized you can generate a training
        dataset with 

        >>> pg.prepare_dataset(number_of_points, dir_name) 

        and prepare the required tensorflow model data with 
        
        >>> pg.prepare_basis(dir_name)
    """

    def __init__(self, monomials, coefficients, kmoduli, ambient, vol_j_norm=None, verbose=2, backend='multiprocessing'):
        r"""The CICYPointGenerator uses the *joblib* module to parallelize 
        computations.

        Args:
            monomials (list(ndarray[(nMonomials, ncoord), np.int])): list of
                length nHyper with monomials for each defining equation.
            coefficients (list(ndarray[(nMonomials)])): list of coefficients
                in front of each monomial.
            kmoduli (ndarray[(nProj)]): The kaehler moduli.
            ambient (ndarray[(nProj), np.int]): the direct product of 
                projective spaces making up the ambient space.
            vol_j_norm (float, optional): Normalization of the volume of the
                Calabi-Yau X as computed from

                .. math:: \int_X J^n \; \text{ at } \; t_1=t_2=...=t_n = 1.

                Defaults to 1.
            verbose (int, optional): Controls logging. 1-Debug, 2-Info,
                else Warning. Defaults to 2.
            backend (str, optional): Backend for Parallel. Defaults to
                'multiprocessing'. 'loky' makes issues with pickle5.
        """
        if verbose == 1:
            level = logging.DEBUG
        elif verbose == 2:
            level = logging.INFO
        else:
            level = logging.WARNING
        logger.setLevel(level=level)
        self.monomials = [m.astype(np.int64) for m in monomials]
        self.coefficients = coefficients
        self.kmoduli = kmoduli
        self.ambient = ambient
        self.degrees = ambient + 1
        self.nhyper = len(monomials)
        self.nmonomials = []
        self.ncoords = monomials[0].shape[1]
        self.coord_to_ambient = np.concatenate(
            [[i for _ in range(p + 1)] for i, p in enumerate(ambient)])
        self.conf = []
        for m in monomials:
            self.nmonomials += [m.shape[0]]
            deg = []
            for i in range(len(ambient)):
                s = np.sum(ambient[:i]) + i
                e = np.sum(ambient[:i + 1]) + i + 1
                deg += [np.sum(m[0, s:e])]
            self.conf += [deg]
        self.p_conf = [[a] + c for a, c in zip(self.ambient, np.array(self.conf).transpose().tolist())]
        self.nfold = np.sum(self.ambient) - self.nhyper

        # sympy variables
        self.x = sp.var('x0:' + str(self.ncoords))
        self.poly = [sum(self.coefficients[i] * np.multiply.reduce(
            np.power(self.x, m), axis=-1)) for i, m in enumerate(monomials)]
        self.backend = backend
        # some more internal variables
        self._set_seed(2021)
        self.lc = get_levicivita_tensor(int(self.nfold))
        self._generate_all_bases()
        self.intersection_tensor = self._generate_intersection_tensor()
        self.vol_j_norm = self.get_volume_from_intersections(np.ones_like(self.kmoduli)) if vol_j_norm is None else vol_j_norm
        

    def _generate_all_bases(self):
        r"""This function calls a bunch of others
        which then generate various monomial basis
        needed for point generation, residue theorem
        and the pullback tensor.
        """
        self.all_ts = self._generate_all_freets()
        self.selected_t = self._find_degrees()
        self._generate_root_basis()
        self._generate_dQdz_basis()
        self.dzdz_generated = False
        self._generate_padded_basis()

    def _generate_root_basis(self):
        r"""Generates monomial basis for the polynomials in 
        the free parameters which are distributed according to
        self.selected_ts. This is roughly 
        
        .. math::
        
            Q(\sum_k p_j*t_jk + q_j)

        """
        self.root_vars = {}
        self.root_monomials = []
        self.root_factors = []
        self.tpoly = 0
        degrees = self.selected_t

        self.root_vars = {}
        self.root_vars['p'] = sp.var('p0:{}:{}'.format(
            self.ncoords, np.max(degrees) + 1))
        self.root_vars['ps'] = sp.Matrix(np.reshape(
            self.root_vars['p'], (self.ncoords, np.max(degrees) + 1)))
        self.root_vars['t'] = sp.var('t0:{}'.format(self.nhyper))
        self.root_vars['ts'] = sp.ones(
            int(self.ncoords), int(np.max(degrees) + 1))
        for i in range(len(self.ambient)):
            for j in range(np.max(degrees) + 1):
                if j > degrees[i]:
                    s = np.sum(self.ambient[:i]) + i
                    e = np.sum(self.ambient[:i + 1]) + i + 1
                    self.root_vars['ps'][s:e, j] = sp.zeros(*np.shape(self.root_vars['ps'][s:e, j]))
        j = 0
        for i in range(len(self.ambient)):
            for k in range(degrees[i]):
                s = np.sum(self.ambient[:i]) + i
                e = np.sum(self.ambient[:i + 1]) + i + 1
                self.root_vars['ts'][s:e, 1 + k] = self.root_vars['t'][j] * sp.ones(
                    *np.shape(self.root_vars['ts'][s:e, 1 + k]))
                j += 1
        self.tpoly = [pi.subs(
            [(self.x[i], sum(self.root_vars['ps'].row(i) * self.root_vars['ts'].row(i).T)) for i in range(self.ncoords)]
        ).as_poly() for pi in self.poly]
        poly_dict = [pi.as_dict() for pi in self.tpoly]
        all_vars = np.array(list(self.root_vars['p']) + list(self.root_vars['t']))
        self.root_monomials = [np.zeros((len(pi), len(all_vars)), dtype=np.int32) for pi in poly_dict]
        self.root_factors = [np.zeros(len(pi), dtype=np.complex128) for pi in poly_dict]
        for j in range(self.nhyper):
            mask = np.logical_or.reduce(all_vars == np.array(list(self.tpoly[j].free_symbols)).reshape(-1, 1))
            for i, entry in enumerate(poly_dict[j]):
                antry = np.array(entry)
                self.root_monomials[j][i, mask] = antry
                self.root_factors[j][i] = poly_dict[j][entry]

    def _root_polynomial(self, x, p):
        r"""Function to be optimized by scipy.opt.fsolve.
        Computes the difference from zero when plugging p+qt into
        the defining polynomials.

        NOTE:
            Takes real arguments in x and returns real output as [Re, Im] of
            the complex computations.

        Args:
            x (ndarray[nHyper, np.float64]): t-values
            p (ndarray[(ncoords, t-max-deg), complex]): Values 
                for points on the spheres p, q, ...

        Returns:
            ndarray[2*nhyper, np.float64]: Difference from zero.
        """
        c = x.view(np.complex128)
        p_e = np.concatenate((p.flatten(), c), axis=-1)
        poly = np.array(
            [np.sum(fact * np.multiply.reduce(np.power(p_e, poly), axis=-1))
             for poly, fact in zip(self.root_monomials, self.root_factors)])
        return poly.view(np.float64)

    def _root_prime(self, x, p):
        # NOTE: this does not actually work as fprime, because going from
        # complex to real to complex messes with the argument shapes and
        # derivatives need not be real.
        # TODO: Work out a hack to make the shapes work.
        c = x.view(np.complex128)
        p_e = np.concatenate((p.flatten(), c), axis=-1)
        poly = np.array(
            [[np.sum(fi * np.multiply.reduce(np.power(p_e, pi), axis=-1)) for pi, fi in zip(poly, fact)]
             for poly, fact in zip(self.root_jacobian, self.root_factors)])
        return poly.view(np.float64)

    def _point_from_sol_sympy(self, p, sol):
        r"""Substitutes the solution for the t-values from scipy.opt
        to generate points on the CICY.

        NOTE:
            Don't use this function as the symbolic substitution is quite 
            expensive. Work with `_point_from_sol()` instead.

        Args:
            p (ndarray[(ncoords, t-max-deg), complex]): Values 
                for points on the spheres p, q, ...
            sol (ndarray[(nhyper), complex]): Complex t-values.

        Returns:
            ndarray[(ncoords), complex]: point on the CICY.
        """
        p_matrix = self.root_vars['ps'].subs(
            tuple((pi, pj) for pi, pj in zip(self.root_vars['p'], p.flatten())))
        t_matrix = self.root_vars['ts'].subs(
            tuple((ti, tj) for ti, tj in zip(self.root_vars['t'], sol)))
        p_matrix = np.array(p_matrix).astype(np.complex128)
        t_matrix = np.array(t_matrix).astype(np.complex128)
        point = np.sum(p_matrix * t_matrix, axis=-1)
        return point

    def _generate_all_freets(self):
        r"""Finds all possible ways to arrange the free parameters.

        Returns:
            ndarray([n_comb, nProj], np.int): all possible arrangements
        """
        free_ts = list(generate_monomials(len(self.ambient), self.nhyper))
        free_ts = np.array(free_ts)
        # we remove all not allowed by degree of projective ambient space
        good = np.reshape(self.ambient, (1, -1)) - free_ts >= 0
        good = np.logical_and.reduce(good, axis=-1)
        # we remove all not allowed because a hypersurface does not get a free t
        for i in range(len(free_ts)):
            if good[i]:
                for hyper in self.conf:
                    involved = free_ts[i].astype(bool)
                    hyper = np.array(hyper)
                    if np.sum(hyper[involved]) == 0:
                        good[i] = False
                        break
        return free_ts[good]

    def _find_degrees(self):
        r"""Generates t-degrees in ambient space factors.
        Determines the shape for the expanded sphere points.
        """
        degrees = np.zeros(len(self.ambient), dtype=np.int32)
        for j in range(self.nhyper):
            d = np.argmax(self.conf[j])
            if degrees[d] == self.ambient[d]:
                # in case we already exhausted all degrees of freedom
                # shouldn't really be here other than for
                # some interesting p1 splits (redundant CICY description?)
                d = np.argmax(self.conf[j, d + 1:])
            degrees[d] += 1
        return degrees

    def _generate_tselected_points(self, n_p, nproc=-1, acc=1e-8, nattempts=1, fprime=None, batch_size=1000):
        r"""Generates complex points.

        NOTE:
            The code will generate `n_p` points, but discard all points not 
            satisfying `acc` on the CICY equations.
            Thus the number of returned points is usually less than n_p.

        Args:
            n_p (int): # of points
            nproc (int, optional): # of processes used. Defaults to -1.
            acc (float, optional): Required CY accuracy. Defaults to 1e-8.
            nattempts (int, optional): # of initial attempts. Defaults to 1.
            fprime (function, optional): fprime of CY conditions with float
                returns. Defaults to None.
            batch_size (int, optional): batch_size of Parallel. 
                Defaults to 1000.

        Returns:
            ndarray[(<np, ncoord), np.complex128]: Points on the CICY.
        """
        max_deg = np.max(self.selected_t)
        pn_pnts = np.zeros((n_p, self.ncoords, max_deg + 1),
                           dtype=np.complex128)
        for i in range(len(self.ambient)):
            for k in range(self.selected_t[i] + 1):
                s = np.sum(self.ambient[:i]) + i
                e = np.sum(self.ambient[:i + 1]) + i + 1
                pn_pnts[:, s:e, k] += self.generate_pn_points(n_p, self.ambient[i])
        points = Parallel(n_jobs=nproc, batch_size=batch_size, backend=self.backend)(
            delayed(self._get_point)(p, acc, nattempts, fprime) for p in pn_pnts)
        points = np.array(points)
        cy_cond = self.cy_condition(points)
        cy_mask = np.logical_and.reduce(np.abs(cy_cond) < acc, axis=-1)
        return points[cy_mask]

    def generate_points(self, n_p, nproc=-1, nattempts=1, acc=1e-8, fprime=None, batch_size=1000):
        r"""Generates n_p complex points from the t-selection in
        'self.selected_t' with accuracy 'acc'.

        Args:
            n_p (int): # of points.
            nproc (int, optional): # of jobs used. Defaults to -1. Then
                uses all available resources.
            nattempts (int, optional): # of attempts for each selection of 
                points generated on the spheres. Defaults to 1.
            acc (float, optional): Required CY accuracy. Defaults to 1e-8.
            fprime (function, optional): fprime of CY conditions with float
                returns. Defaults to None.
            batch_size (int, optional): batch_size of Parallel.
                Defaults to 1000.

        Returns:
            ndarray[(n_p, ncoord), np.complex128]: Points on the CICY.
        """
        points = np.ones((n_p, self.ncoords), dtype=np.complex128)
        logger.debug('Generating points for t-selections {}.'.format(
            self.selected_t))
        # NOTE: be careful if we give different lengths
        new_points = self._generate_tselected_points(n_p, nproc=nproc, acc=acc, nattempts=nattempts, fprime=fprime,
                                                     batch_size=batch_size)
        n_p_found = len(new_points)
        n_p_red = n_p
        logger.debug('found {} out of {} expected.'.format(n_p_found, n_p_red))
        points[0:n_p_found] = new_points
        if n_p_found < n_p_red:
            # sample more points
            ratio = n_p_red / n_p_found
            for _ in range(5):
                # hopefully only need one iteration, but might get unlucky
                missing = n_p_red - n_p_found
                n_p_more = int(ratio * missing + 50)
                logger.debug('generating {} more points.'.format(n_p_more))
                new_points = self._generate_tselected_points(n_p_more, nproc=nproc, acc=acc, nattempts=nattempts,
                                                             fprime=fprime, batch_size=batch_size)
                logger.debug('found {} out of {} expected.'.format(
                    len(new_points), missing))
                if n_p_found + len(new_points) > n_p_red:
                    points[n_p_found:] = new_points[0:missing]
                    break
                else:
                    points[n_p_found:n_p_found + len(new_points)] = new_points
                    n_p_found += len(new_points)
        npoints = self._rescale_points(points)
        return npoints

    def _get_point(self, p, acc=1e-8, nattempts=1, fprime=None):
        r"""Generates a single point on the CICY
        for given points on the ambient spheres.

        NOTE:
            It is not guaruanteed that fsolve converges to a point on the CICY.
            Thus double check the result.

        Args:
            p (ndarray[(ncoords, t-max-deg), np.complex128]): Values 
                for points on the spheres p, q, ...
            acc (float, optional): Accuracy for accepted solution.
                Defaults to 1e-8.
            nattempts (int, optional): Attempts to find a solution with given
                accuracy. Defaults to 1.
            fprime (function, optional): function for fprime in opt.fsolve.
                Defaults to None.

        Returns:
            ndarray[(nCoords), np.complex128]: (potential) Point on the CICY.
        """
        # TODO: add more arguments to fsolve
        best_sol = np.random.randn(2 * self.nhyper)
        best_acc = np.sum(np.abs(self._root_polynomial(best_sol, p)))
        for _ in range(nattempts):
            sol = opt.fsolve(
                self._root_polynomial, np.random.randn(2 * self.nhyper),
                fprime=fprime, args=(p))
            acc1 = np.sum(np.abs(self._root_polynomial(sol, p)))
            if acc1 < best_acc:
                best_acc = acc1
                best_sol = sol
                if acc1 < acc:
                    break
        return self._point_from_sol(p, best_sol.view(np.complex128))

    def _generate_dQdz_basis(self):
        r"""Generates a basis for dQ/dz_j.
        """
        self.dQdz_basis = [[] for _ in range(self.nhyper)]
        self.dQdz_factors = [[] for _ in range(self.nhyper)]
        for j in range(self.nhyper):
            for i, m in enumerate(np.eye(self.ncoords, dtype=np.int32)):
                basis = self.monomials[j] - m
                factors = self.monomials[j][:, i] * self.coefficients[j]
                good = np.ones(len(basis), dtype=bool)
                good[np.where(basis < 0)[0]] = False
                self.dQdz_basis[j] += [basis[good]]
                self.dQdz_factors[j] += [factors[good]]

    def _generate_dzdz_basis(self, nproc=-1):
        r"""Generates a basis for dz_i/dz_j which was needed for the
        pullback tensor. NOTE: This code is not actively needed anymore.
        """
        self.dzdz_generated = True
        self.dzdz_basis = [[[(np.zeros((1, self.ncoords), dtype=np.int32),
                              np.zeros((1, self.ncoords), dtype=np.int32))
                             for _ in range(self.ncoords)]
                            for _ in range(self.ncoords)] for _ in range(self.nhyper)]
        self.dzdz_factor = [[[([0], [0]) for _ in range(self.ncoords)]
                             for _ in range(self.ncoords)] for _ in range(self.nhyper)]
        self.iderivatives = [[Parallel(n_jobs=nproc, backend=self.backend)
                              (delayed(self._implicit_diff)(i, j, k) for i in range(self.ncoords))
                              for j in range(self.ncoords)] for k in range(self.nhyper)]
        for k in range(self.nhyper):
            for j in range(self.ncoords):
                for i in range(self.ncoords):
                    if i != j and self.conf[self.coord_to_ambient[i]][k] != 0 \
                            and self.conf[self.coord_to_ambient[j]][k] != 0:
                        self.dzdz_basis[k][j][i], self.dzdz_factor[k][j][i] = \
                            self._frac_to_monomials(self.iderivatives[k][j][i])

    def _implicit_diff(self, i, j, k):
        r"""Compute the implicit derivative of
            dzi/dzj
        given the sympy polynomial self.poly

        Args:
            i (int): i index
            j (int): j index
            k (int): hypersurface index

        Returns:
            sympy poly: implicit derivative
        """
        if i == j or self.conf[self.coord_to_ambient[i]][k] == 0 or \
                self.conf[self.coord_to_ambient[j]][k] == 0:
            return 0
        return idiff(self.poly[k], self.x[i], self.x[j])

    def _generate_padded_basis(self):
        r"""Generates padded basis for tf_models.
        """
        self.BASIS = {}
        for k in range(self.nhyper):
            # first dQdz
            shape = np.array([np.shape(mb) for mb in self.dQdz_basis[k]])
            DQDZB = np.zeros((len(shape), np.max(shape[:, 0]), len(shape)),
                             dtype=np.complex64)
            DQDZF = np.zeros((len(shape), np.max(shape[:, 0])),
                             dtype=np.complex64)
            for i, m in enumerate(zip(self.dQdz_basis[k],
                                      self.dQdz_factors[k])):
                DQDZB[i, 0:shape[i, 0]] += m[0]
                DQDZF[i, 0:shape[i, 0]] += m[1]
            self.BASIS['DQDZB' + str(k)] = np.copy(DQDZB)
            self.BASIS['DQDZF' + str(k)] = np.copy(DQDZF)
            self.BASIS['QB' + str(k)] = self.monomials[k]
            self.BASIS['QF' + str(k)] = self.coefficients[k]
            # next dzdz, TODO padd before remove enumerate?
            if self.dzdz_generated:
                shapes = np.array([[
                    [np.shape(t[0]), np.shape(t[1])] if i != j
                    else [[-1, -1], [-1, -1]] for i, t in enumerate(zi)]
                    for j, zi in enumerate(self.dzdz_basis[k])]
                )
                DZDZB_d = np.zeros((self.ncoords, self.ncoords, np.max(shapes[:, 0, 0]), self.ncoords), dtype=np.int64)
                DZDZB_n = np.zeros((self.ncoords, self.ncoords, np.max(shapes[:, 1, 0]), self.ncoords), dtype=np.int64)
                DZDZF_d = np.zeros((self.ncoords, self.ncoords, np.max(shapes[:, 0, 0])), dtype=np.complex64)
                DZDZF_n = np.zeros((self.ncoords, self.ncoords, np.max(shapes[:, 1, 0])), dtype=np.complex64)
                for i in range(self.ncoords):
                    for j in range(self.ncoords):
                        if i != j:
                            DZDZB_d[i, j, 0:shapes[i, j, 0, 0]] += self.dzdz_basis[k][i][j][0]
                            DZDZB_n[i, j, 0:shapes[i, j, 1, 0]] += self.dzdz_basis[k][i][j][1]
                            DZDZF_d[i, j, 0:shapes[i, j, 0, 0]] += self.dzdz_factor[k][i][j][0]
                            DZDZF_n[i, j, 0:shapes[i, j, 1, 0]] += self.dzdz_factor[k][i][j][1]
                self.BASIS['DZDZB_d' + str(k)] = np.copy(DZDZB_d)
                self.BASIS['DZDZB_n' + str(k)] = np.copy(DZDZB_n)
                self.BASIS['DZDZF_d' + str(k)] = np.copy(DZDZF_d)
                self.BASIS['DZDZF_n' + str(k)] = np.copy(DZDZF_n)

    def holomorphic_volume_form(self, points, j_elim=None):
        r"""We compute the holomorphic volume form
        at all points by solving the residue theorem:
    
        .. math::
    
            \Omega &= \int_\rho \frac{1}{Q} \wedge^n dz_i \\
                   &= \frac{1}{\frac{\partial Q}{\partial z_j}}\wedge^{n-1} dz_a
    
        where the index a runs over the local n-fold good coordinates.
    
        Args:
            points (ndarray[(n_p, ncoords), np.complex128]): Points.
            j_elim (ndarray[(n_p, nhyper), np.int64]): Index to be eliminated. 
                Defaults to None. If None eliminates max(dQdz).
    
        Returns:
            ndarray[(n_p), np.complex128]: Omega evaluated at each point
        """
        indices = self._find_max_dQ_coords(points) if j_elim is None else j_elim
        omega = np.zeros((len(points), self.nhyper, self.nhyper), dtype=np.complex128)
        for i in range(self.nhyper):
            for j in range(self.nhyper):
                tmp_omega = np.power(np.expand_dims(points, 1), self.BASIS['DQDZB' + str(i)][indices[:, j]])
                tmp_omega = np.multiply.reduce(tmp_omega, axis=-1)
                omega[:, i, j] = np.add.reduce(self.BASIS['DQDZF' + str(i)][indices[:, j]] * tmp_omega, axis=-1)
        # compute (dQ/dzj)**-1
        return 1 / np.linalg.det(omega)

    def _find_max_dQ_coords(self, points):
        r"""finds the coordinates for which |dQ/dzj| is largest.

        Args:
            points (ndarray[(n_p, ncoords), np.complex128]): Points.

        Returns:
            ndarray[(n_p, nhyper), np.int64]: maxdQdz indices
        """
        available_mask = ~np.isclose(points, complex(1, 0))
        max_coords = np.zeros((len(points), self.nhyper), dtype=np.int32)
        for i in range(self.nhyper):
            dQdz = np.abs(self._compute_dQdz(points, i))
            max_coords[:, i] = np.argmax(dQdz * available_mask, axis=-1)
            available_mask[np.arange(len(points)), max_coords[:, i]] = False
        return max_coords

    def _find_good_coordinate_mask(self, points):
        r"""Computes a mask for points with True
        in the position of the nfold good coordinates.

        Args:
            points (ndarray[(n_p, ncoords), np.complex128]): Points.

        Returns:
            ndarray[(n_p, ncoord), bool]: good coordinate mask
        """
        mask = ~np.isclose(points, complex(1, 0))
        indices = self._find_max_dQ_coords(points)
        for i in range(self.nhyper):
            mask[np.arange(len(points)), indices[:, i]] = False
        return mask

    def _compute_dQdz(self, points, k):
        r"""Computes dQdz at each point.

        Args:
            points (ndarray([n_p, ncoords], complex)): Points.
            k (int): hypersurface index

        Returns:
            ndarray([n_p, ncoords], complex): dQdz at each point.
        """
        p_exp = np.expand_dims(np.expand_dims(points, 1), 1)
        dQdz = np.power(p_exp, self.BASIS['DQDZB' + str(k)])
        dQdz = np.multiply.reduce(dQdz, axis=-1)
        dQdz = np.multiply(self.BASIS['DQDZF' + str(k)], dQdz)
        dQdz = np.add.reduce(dQdz, axis=-1)
        return dQdz

    def prepare_dataset(self, n_p, dirname, ltails=0., **kwargs):
        r"""Prepares training and validation data.

        keyword arguments can be any from 
        :py:func:`cymetric.pointgen.nphelper.prepare_dataset`.

        NOTE:
            By default we remove 0.05 of the points with vanishing 
            weights. In our experience the numerics can become quite messy
            at these points. Furthermore, they don't contribute much to the MC
            integration and can thus more or less safely be removed.

        Args:
            n_p (int): Number of points to generate.
            dirname (str): Directory name to save dataset in.
            ltails (float, optional): percentage discarded on the left tail
                of weight distribution. Defaults to 0.

        Returns:
            int: 0
        """
        return prepare_dataset(self, n_p, dirname, ltails=ltails, **kwargs)

    def cy_condition(self, points):
        r"""Computes the CY condition at each point.

        Args:
            points (ndarray[(n_p, ncoords), complex]): Points.

        Returns:
            ndarray([n_p, nhyper], complex): CY condition
        """
        cy_cond = np.zeros((len(points), self.nhyper), dtype=points.dtype)
        for i, (c, m) in enumerate(zip(self.coefficients, self.monomials)):
            cy_cond[:, i] = np.add.reduce(c * np.multiply.reduce(
                np.power(np.expand_dims(points, 1), m), axis=-1), axis=-1)
        return cy_cond
