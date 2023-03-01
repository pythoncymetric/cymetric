"""
Main PointGenerator module.

:Authors:
    Fabian Ruehle <fabian.ruehle@cern.ch> and 
    Robin Schneider <robin.schneider@physics.uu.se>
"""
import numpy as np
import logging
import sympy as sp
from cymetric.pointgen.nphelper import prepare_basis_pickle, prepare_dataset, get_levicivita_tensor
from sympy.geometry.util import idiff
from joblib import Parallel, delayed
import itertools

logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s')
logger = logging.getLogger('pointgen')


class PointGenerator:
    r"""The PointGenerator class.

    The numerics are entirely done in numpy; sympy is used for taking 
    (implicit) derivatives.

    Use this one if you want to generate points and data on a CY given by
    one hypersurface.
    
    All other PointGenerators inherit from this class.

    Example:
        We consider the Fermat quintic given by

        .. math::

            Q(z) = z_1^5 + z_2^5 + z_3^5 + z_4^5 + z_5^5

        and set it up with:

        >>> import numpy as np
        >>> from cymetric.pointgen.pointgen import PointGenerator
        >>> monomials = 5*np.eye(5, dtype=np.int)
        >>> coefficients = np.ones(5)
        >>> kmoduli = np.ones(1)
        >>> ambient = np.array([4])
        >>> pg = PointGenerator(monomials, coefficients, kmoduli, ambient)

        Once the PointGenerator is initialized you can generate a training
        dataset with 

        >>> pg.prepare_dataset(number_of_points, dir_name) 

        and prepare the required tensorflow model data with 
        
        >>> pg.prepare_basis(dir_name)
    """

    def __init__(self, monomials, coefficients, kmoduli, ambient, vol_j_norm=None, verbose=2, backend='multiprocessing'):
        r"""The PointGenerator uses the *joblib* module to parallelize 
        computations. 

        Args:
            monomials (ndarray[(nMonomials, ncoords), np.int]): monomials
            coefficients (ndarray[(nMonomials)]): coefficients in front of each
                monomial.
            kmoduli (ndarray[(nProj)]): the kaehler moduli.
            ambient (ndarray[(nProj), np.int]): the direct product of projective
                spaces making up the ambient space.
            vol_j_norm (float, optional): Normalization of the volume of the
                Calabi-Yau X as computed from

                .. math:: \int_X J^n \; \text{ at } \; t_1=t_2=...=t_n = 1.

                Defaults to None, in which case the normalization will be computed automatically from the intersection numbers.
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
        self.monomials = monomials.astype(np.int64)
        self.coefficients = coefficients
        self.kmoduli = kmoduli
        self.ambient = ambient.astype(np.int64)
        self.degrees = ambient + 1
        self.nhyper = 1
        self.nmonomials, self.ncoords = monomials.shape
        self.nfold = np.sum(self.ambient) - self.nhyper
        self.backend = backend
        self.p_conf = np.array([[a, d] for a, d in zip(self.ambient, self.degrees)])
        self.lc = get_levicivita_tensor(int(self.nfold))
        # sympy variables
        self.x = sp.var('x0:' + str(self.ncoords))
        self.poly = sum(self.coefficients * np.multiply.reduce(
            np.power(self.x, self.monomials), axis=-1))
        # more general
        # self.c = sp.var('c0:'+str(self.nmonomials))
        # self.gpoly = sum(self.c *
        #    np.multiply.reduce(np.power(self.x, self.monomials), axis=-1))

        old_monoms = self.monomials.copy()  # make dimensions consistent with CICY case
        self.monomials = np.array([self.monomials]) 
        self.intersection_tensor = self._generate_intersection_tensor()
        self.monomials = old_monoms  # undo change
        
        self.vol_j_norm = self.get_volume_from_intersections(np.ones_like(self.kmoduli)) if vol_j_norm is None else vol_j_norm

        # some more internal variables
        self._set_seed(2021)
        self._generate_all_bases()

    @staticmethod
    def _set_seed(seed):
        # sets the numpy seed for point gen
        np.random.seed(seed)

    def _generate_all_bases(self):
        r"""This function calls a bunch of others
        which then generate various monomial bases
        needed for point generation, residue theorem
        and the pullback tensor.
        """
        self.all_ts = np.eye(len(self.ambient), dtype=np.int64)
        self.selected_t = self.all_ts[np.argmax(self.ambient)]
        self._generate_root_basis()
        self._generate_dQdz_basis()
        # we disable dzdz derivatives, there is not much difference in 
        # pullback accuracy with the inverse vs implicit derivatives.
        self.dzdz_generated = False
        self._generate_padded_basis()

    def _generate_root_basis(self):
        r"""Generates monomial basis for the polynomials in 
        the free parameters which are distributed according to
        self.selected_t. This is roughly 
        
        .. math::
        
            Q(\sum_k p_j*t_jk + q_j)

        """
        self.root_vars = {}
        self.root_monomials = []
        self.root_factors = []
        self.tpoly = 0
        self.root_vars = {}
        self.root_vars['p'] = sp.var('p0:{}:{}'.format(
            self.ncoords, np.max(self.selected_t) + 1))
        self.root_vars['ps'] = sp.Matrix(np.reshape(
            self.root_vars['p'], (self.ncoords, np.max(self.selected_t) + 1)))
        self.root_vars['t'] = sp.var('t0:{}'.format(self.nhyper))
        self.root_vars['ts'] = sp.ones(
            int(self.ncoords), int(np.max(self.selected_t) + 1))
        for i in range(len(self.ambient)):
            for j in range(np.max(self.selected_t) + 1):
                if j > self.selected_t[i]:
                    s = np.sum(self.ambient[:i]) + i
                    e = np.sum(self.ambient[:i + 1]) + i + 1
                    self.root_vars['ps'][s:e, j] = \
                        sp.zeros(*np.shape(self.root_vars['ps'][s:e, j]))
        j = 0
        for i in range(len(self.ambient)):
            for k in range(self.selected_t[i]):
                s = np.sum(self.ambient[:i]) + i
                e = np.sum(self.ambient[:i + 1]) + i + 1
                self.root_vars['ts'][s:e, 1 + k] = self.root_vars['t'][j] * sp.ones(*np.shape(self.root_vars['ts'][s:e, 1 + k]))
                j += 1
        self.tpoly = self.poly.subs(
            [(self.x[i], sum(self.root_vars['ps'].row(i) * self.root_vars['ts'].row(i).T))
             for i in range(self.ncoords)]).as_poly()
        poly_dict = self.tpoly.as_dict()
        all_vars = np.array(list(self.root_vars['p']) + list(self.root_vars['t']))
        root_monomials = np.zeros((len(poly_dict), len(all_vars)), dtype=np.int32)
        root_factors = np.zeros(len(poly_dict), dtype=np.complex128)
        mask = np.logical_or.reduce(all_vars == np.array(list(self.tpoly.free_symbols)).reshape(-1, 1))
        for i, entry in enumerate(poly_dict):
            antry = np.array(entry)
            root_monomials[i, mask] = antry
            root_factors[i] = poly_dict[entry]
        # sort root_monomials to work with np.root
        t_mask = self.root_vars['t'] == all_vars
        t_index = np.where(t_mask)[0][0]
        # +1 because hypersurface
        max_degree = int(self.ambient[self.selected_t.astype(bool)]) + 1
        # +1 for degree zero
        for j in range(max_degree + 1):
            good = root_monomials[:, t_index] == max_degree - j
            tmp_monomials = root_monomials[good]
            self.root_monomials += [np.delete(tmp_monomials, t_index, axis=1)]
            self.root_factors += [root_factors[good]]

    def _generate_root_basis_Q(self):
        r"""Generates a monomial basis for the 1d poly in t
        coming from :math:`Q(p \cdot t + q)`. 

        NOTE: this one is legacy code.
        """
        p = sp.var('p0:{}'.format(self.ncoords))
        q = sp.var('q0:{}'.format(self.ncoords))
        t = sp.var('t')
        # change here to self.gpoly
        poly_p = self.poly.subs([(self.x[i], p[i] * t + q[i])
                                 for i in range(self.ncoords)]).as_poly()
        poly_dict = poly_p.as_dict()
        p_monomials = np.zeros((len(poly_dict), self.ncoords), dtype=np.int32)
        q_monomials = np.zeros((len(poly_dict), self.ncoords), dtype=np.int32)
        factors = np.zeros(len(poly_dict), dtype=np.complex128)
        for i, entry in enumerate(poly_dict):
            p_monomials[i, :] = entry[0:self.ncoords]
            q_monomials[i, :] = entry[self.ncoords:2 * self.ncoords]
            factors[i] = poly_dict[entry]
        self.root_monomials_Q = []
        self.root_factors_Q = []
        for i in range(self.ncoords + 1):
            sums = np.sum(p_monomials, axis=-1)
            indi = np.where(sums == self.ncoords - i)[0]
            self.root_monomials_Q += [(p_monomials[indi], q_monomials[indi])]
            self.root_factors_Q += [factors[indi]]

    def _generate_dQdz_basis(self):
        r"""Generates a monomial basis for dQ/dz_j."""
        self.dQdz_basis = []
        self.dQdz_factors = []
        for i, m in enumerate(np.eye(self.ncoords, dtype=np.int32)):
            basis = self.monomials - m
            factors = self.monomials[:, i] * self.coefficients
            good = np.ones(len(basis), dtype=bool)
            good[np.where(basis < 0)[0]] = False
            self.dQdz_basis += [basis[good]]
            self.dQdz_factors += [factors[good]]

    def _generate_dzdz_basis(self, nproc=-1):
        r"""Generates a monomial basis for dz_i/dz_j
        which was needed for the pullback tensor.
        """
        self.dzdz_basis = [[([], []) for _ in range(self.ncoords)]
                           for _ in range(self.ncoords)]
        self.dzdz_factor = [[([0], [0]) for _ in range(self.ncoords)]
                            for _ in range(self.ncoords)]
        # take implicit derivatives
        self.iderivatives = [Parallel(n_jobs=nproc, backend=self.backend)
                             (delayed(self._implicit_diff)(i, j)
                              for i in range(self.ncoords))
                             for j in range(self.ncoords)]
        for j in range(self.ncoords):
            for i in range(self.ncoords):
                if i != j:
                    self.dzdz_basis[j][i], self.dzdz_factor[j][i] = \
                        self._frac_to_monomials(self.iderivatives[j][i])

    def _generate_intersection_tensor(self):
        if self.nfold == 1:
            get_int = self._dr
        elif self.nfold == 2:
            get_int = self._drs
        elif self.nfold == 3:
            get_int = self._drst
        elif self.nfold == 4:
            get_int = self._drstu
        elif self.nfold > 4:
            raise NotImplementedError("Computation of intersection numbers is not supported for {}-folds".format(self.nfold))

        comb = itertools.combinations_with_replacement(range(len(self.kmoduli)), int(self.nfold))
        d = np.zeros([len(self.kmoduli)] * int(self.nfold), dtype=int)
        for x in comb:
            d_int = get_int(*x)
            entries = itertools.permutations(x, int(self.nfold))
            # there will be some redundant elements, but they will only have to be calculated once.
            for b in entries:
                d[b] = d_int
        return d

    def _dr(self, r):
        r"""
        Determines the intersection number d_r.
        We use:
        .. math::
            \begin{align}
             d_{r} = \int_X J_r = \int_A \mu \wedge J_r
            \end{align}
        where \mu is the top form

        .. math::
            \begin{align}
            \mu = \bigwedge^K_{a=1} \left(  \sum_{p=1}^{m} q_a^p J_p  \right) \; .
            \end{align}
        Parameters
        ----------
        r : int
            index r.

        Returns
        -------
        dr: float
            Returns the intersection number dr.

        Example
        -------
        >>> M = CICY([[2,3]])
        >>> M.drst(1)
        3.0
        """
        dr = 0
        i = 0
        combination, count = np.zeros(len(self.monomials), dtype=int), np.zeros(len(self.kmoduli), dtype=int)
        # now we want to fill combination and run over all m Projective spaces, and how often they occur
        for j in range(len(self.kmoduli)):
            if j == r:
                count[j] = self.p_conf[j][0] - 1
                combination[i:i + count[j]] = j
                i += self.p_conf[j][0] - 1
            else:
                count[j] = self.p_conf[j][0]
                combination[i:i + count[j]] = j
                i += self.p_conf[j][0]
        mu = sp.utilities.iterables.multiset_permutations(combination)
        for a in mu:
            v = 1
            for j in range(len(self.monomials)):
                if self.p_conf[a[j]][j + 1] == 0:
                    v = 0
                    break
                else:
                    v *= self.p_conf[a[j]][j + 1]
            dr += v
        return float(dr)

    def _drs(self, r, s):
        r"""
        Determines the intersection number d_rs.
        We use:
        .. math::
            \begin{align}
             d_{rs} = \int_X J_r \wedge J_s = \int_A \mu \wedge J_r \wedge J_s
            \end{align}
        where \mu is the top form

        .. math::
            \begin{align}
            \mu = \bigwedge^K_{a=1} \left(  \sum_{p=1}^{m} q_a^p J_p  \right) \; .
            \end{align}
        Parameters
        ----------
        r : int
            index r.
        s : int
            index s.

        Returns
        -------
        drs: float
            Returns the intersection number drs.

        Example
        -------
        >>> M = CICY([[3,4]])
        >>> M.drst(0)
        4.0
        """
        drs = 0
        # Define the relevant part of \mu := \wedge^K_j \sum_r q_r^j J_r
        combination, count = np.zeros(len(self.monomials), dtype=int), np.zeros(len(self.kmoduli), dtype=int)
        # now there are 2 distinct cases:
        # 1) r=s or 2) r != s
        # 1)
        if r == s:
            if self.p_conf[r][0] < 2:
                # then drs is zero
                return 0
            else:
                i = 0
                # now we want to fill combination and run over all m Projective spaces,
                # and how often they occur
                for j in range(len(self.kmoduli)):
                    if j == r:
                        count[j] = self.p_conf[j][0] - 2
                        combination[i:i + count[j]] = j
                        i += self.p_conf[j][0] - 2
                    else:
                        count[j] = self.p_conf[j][0]
                        combination[i:i + count[j]] = j
                        i += self.p_conf[j][0]
        # 2)
        else:
            i = 0
            for j in range(len(self.kmoduli)):
                if j == r or j == s:
                    count[j] = self.p_conf[j][0] - 1
                    combination[i:i + count[j]] = j
                    i += self.p_conf[j][0] - 1
                else:
                    count[j] = self.p_conf[j][0]
                    combination[i:i + count[j]] = j
                    i += self.p_conf[j][0]

        mu = sp.utilities.iterables.multiset_permutations(combination)
        for a in mu:
            v = 1
            for j in range(len(self.monomials)):
                if self.p_conf[a[j]][j + 1] == 0:
                    v = 0
                    break
                else:
                    v *= self.p_conf[a[j]][j + 1]
            drs += v
        return drs

    def _drst(self, r, s, t):
        r"""
        Determines the triple intersection number d_rst.
        We use:
        .. math::
            \begin{align}
             d_{rst} = \int_X J_r \wedge J_s \wedge J_t = \int_A \mu \wedge J_r \wedge J_s \wedge J_t
            \end{align}
        where \mu is the top form

        .. math::
            \begin{align}
            \mu = \bigwedge^K_{a=1} \left(  \sum_{p=1}^{m} q_a^p J_p  \right) \; .
            \end{align}
        Parameters
        ----------
        r : int
            index r.
        s : int
            index s.
        t : int
            index t.

        Returns
        -------
        drst: float
            Returns the triple intersection number drst.

        Example
        -------
        >>> M = CICY([[2,2,1],[3,1,3]])
        >>> M.drst(0,1,1)
        7.0
        """
        drst = 0
        # Define the relevant part of \mu := \wedge^K_j \sum_r q_r^j J_r
        combination = np.array([0 for _ in range(len(self.monomials))])
        count = [0 for _ in range(len(self.kmoduli))]
        # now there are 5 distinct cases:
        # 1) r=s=t or 2) all neqal or the 2-5) three cases where two are equal
        # 1)
        if r == s == t:
            if self.p_conf[r][0] < 3:
                # then drst is zero
                return 0
            else:
                i = 0
                # now we want to fill combination and run over all m Projective spaces,
                # and how often they occur
                for j in range(len(self.kmoduli)):
                    if j == r:
                        # we obviously have to subtract 3 in the case of three
                        # times the same index since we already have three kähler forms
                        # in Ambient space coming from the intersection number
                        count[j] = self.p_conf[j][0] - 3
                        combination[i:i + count[j]] = j
                        i += self.p_conf[j][0] - 3
                    else:
                        count[j] = self.p_conf[j][0]
                        combination[i:i + count[j]] = j
                        i += self.p_conf[j][0]
        # 2)
        if r != s and r != t and s != t:
            i = 0
            for j in range(len(self.kmoduli)):
                if j == r or j == s or j == t:
                    count[j] = self.p_conf[j][0] - 1
                    combination[i:i + count[j]] = j
                    i += self.p_conf[j][0] - 1
                else:
                    count[j] = self.p_conf[j][0]
                    combination[i:i + count[j]] = j
                    i += self.p_conf[j][0]
        # 3)
        if r == s and r != t:
            if self.p_conf[r][0] < 2:
                return 0
            else:
                i = 0
                for j in range(len(self.kmoduli)):
                    if j == r:
                        count[j] = self.p_conf[j][0] - 2
                        combination[i:i + count[j]] = j
                        i += self.p_conf[j][0] - 2
                    else:
                        if j == t:
                            count[j] = self.p_conf[j][0] - 1
                            combination[i:i + count[j]] = j
                            i += self.p_conf[j][0] - 1
                        else:
                            count[j] = self.p_conf[j][0]
                            combination[i:i + count[j]] = j
                            i += self.p_conf[j][0]
        # 4)
        if r == t and r != s:
            i = 0
            if self.p_conf[r][0] < 2:
                return 0
            else:
                i = 0
                for j in range(len(self.kmoduli)):
                    if j == r:
                        count[j] = self.p_conf[j][0] - 2
                        combination[i:i + count[j]] = j
                        i += self.p_conf[j][0] - 2
                    else:
                        if j == s:
                            count[j] = self.p_conf[j][0] - 1
                            combination[i:i + count[j]] = j
                            i += self.p_conf[j][0] - 1
                        else:
                            count[j] = self.p_conf[j][0]
                            combination[i:i + count[j]] = j
                            i += self.p_conf[j][0]
        # 5)
        if s == t and s != r:
            i = 0
            if self.p_conf[s][0] < 2:
                return 0
            else:
                i = 0
                for j in range(len(self.kmoduli)):
                    if j == s:
                        count[j] = self.p_conf[j][0] - 2
                        combination[i:i + count[j]] = j
                        i += self.p_conf[j][0] - 2
                    else:
                        if j == r:
                            count[j] = self.p_conf[j][0] - 1
                            combination[i:i + count[j]] = j
                            i += self.p_conf[j][0] - 1
                        else:
                            count[j] = self.p_conf[j][0]
                            combination[i:i + count[j]] = j
                            i += self.p_conf[j][0]
        # the combinations of mu grow exponentially with len(self.monomials) and the number of ambient spaces
        # Check, when the number of multiset_permutations become to large to handle
        if len(self.monomials) < 8 and len(np.unique(combination)) < 6:
            # Hence, for large K and small len(self.kmoduli), this might take really long.
            mu = sp.utilities.iterables.multiset_permutations(combination)
            # (len(self.monomials))!/(#x_1!*...*#x_n!)
            for a in mu:
                v = 1
                for j in range(len(self.monomials)):
                    if self.p_conf[a[j]][j + 1] == 0:
                        v = 0
                        break
                    else:
                        v *= self.p_conf[a[j]][j + 1]
                drst += v
            return drst
        else:
            # here we calculate the nonzero paths through the CICY
            # much faster since CICYs with large K and large len(self.kmoduli) tend to
            # be pretty sparse
            nonzero = [[] for _ in range(len(self.monomials))]
            combination = np.sort(combination)
            count_2 = [0 for _ in range(len(self.kmoduli))]
            # run over all K to find possible paths
            for i in range(len(self.monomials)):
                for j in range(len(self.kmoduli)):
                    # possible paths are non zero and in combination
                    if self.p_conf[j][i + 1] != 0 and j in combination:
                        nonzero[i] += [j]
                        count_2[j] += 1
            # Next we run over all entries in count to see if any are fixed by number of occurence
            for i in range(len(self.kmoduli)):
                if count[i] == count_2[i]:
                    # if equal we run over all entries in nonzero
                    # count[i] = 0
                    for j in range(len(self.monomials)):
                        # and fix them to i if they contain it
                        if i in nonzero[j]:
                            # and len(nonzero[j]) != 1
                            nonzero[j] = [i]
            # There are some improvements here:
            # 1) take the counts -= 1 if fixed and compare if the left allowed
            # 2) here it would be even more efficient to write a product that respects
            #   the allowed combinations from count.
            mu = itertools.product(*nonzero)
            # len(nonzero[0])*...*len(nonzero[K])
            # since we in principle know the complexity of both calculations
            # one could also do all the stuff before and then decide which way is faster
            for a in mu:
                # if allowed by count
                c = list(a)
                if np.array_equal(np.sort(c), combination):
                    v = 1
                    for j in range(len(self.monomials)):
                        if self.p_conf[c[j]][j + 1] == 0:
                            break
                        else:
                            v *= self.p_conf[c[j]][j + 1]
                    drst += v
            return drst

    def _drstu(self, r, s, t, u):
        r"""
        Determines the quadruple intersection numbers, d_rstu, for Calabi Yau 4-folds.

        Parameters
        ----------
        r : int
            the index r.
        s : int
            the index s.
        t : int
            the index t.
        u : int
            the index u.

        Returns
        -------
        drstu: float
            The quadruple intersection number d_rstu.

        Example
        -------
        >>> M = CICY([[2,3],[2,3],[1,2]])
        >>> M.drstu(0,1,1,2)
        3
        References
        ----------
        .. [1] All CICY four-folds, by J. Gray, A. Haupt and A. Lukas.
            https://arxiv.org/pdf/1303.1832.pdf
        """

        if self.nfold != 4:
            logger.warning('CICY is not a 4-fold.')

        drstu = 0
        # Define the relevant part of \mu := \wedge^K_j \sum_r q_r^j J_r
        combination = np.array([0 for _ in range(len(self.monomials))])
        count = [0 for _ in range(len(self.kmoduli))]
        # now there are 5 distinct cases:
        # 1) r=s=t=u or 2) all neqal or the 3) two equal, two nonequal
        # 4) two equal and two equal 5) three equal
        un, unc = np.unique([r, s, t, u], return_counts=True)
        for i in range(len(un)):
            if self.p_conf[un[i]][0] < unc[i]:
                return 0
        i = 0
        for j in range(len(self.kmoduli)):
            # if j in rstu subtract
            # else go full
            contained = False
            for a in range(len(un)):
                if j == un[a]:
                    contained = True
                    count[j] = self.p_conf[j][0] - unc[a]
                    combination[i:i + count[j]] = j
                    i += self.p_conf[j][0] - unc[a]
            if not contained:
                count[j] = self.p_conf[j][0]
                combination[i:i + count[j]] = j
                i += self.p_conf[j][0]
        # just copy from drst
        # the combinations of mu grow exponentially with len(self.monomials) and the number of ambient spaces
        # Check, when the number of multiset_permutations become to large to handle
        if len(self.monomials) < 8 and len(np.unique(combination)) < 6:
            # Hence, for large K and small, this might take really long.
            mu = sp.utilities.iterables.multiset_permutations(combination)
            # (len(self.monomials))!/(#x_1!*...*#x_n!)
            for a in mu:
                v = 1
                for j in range(len(self.monomials)):
                    if self.p_conf[a[j]][j + 1] == 0:
                        v = 0
                        break
                    else:
                        v *= self.p_conf[a[j]][j + 1]
                drstu += v
            return drstu
        else:
            # here we calculate the nonzero paths through the CICY
            nonzero = [[] for _ in range(len(self.monomials))]
            combination = np.sort(combination)
            count_2 = [0 for _ in range(len(self.kmoduli))]
            # run over all len(self.monomials) to find possible paths
            for i in range(len(self.monomials)):
                for j in range(len(self.kmoduli)):
                    # possible paths are non zero and in combination
                    if self.p_conf[j][i + 1] != 0 and j in combination:
                        nonzero[i] += [j]
                        count_2[j] += 1
            # Next we run over all entries in count to see if any are fixed by number of occurence
            for i in range(len(self.kmoduli)):
                if count[i] == count_2[i]:
                    # if equal we run over all entries in nonzero
                    # count[i] = 0
                    for j in range(len(self.monomials)):
                        # and fix them to i if they contain it
                        if i in nonzero[j]:
                            # and len(nonzero[j]) != 1
                            nonzero[j] = [i]
            # There are some improvements here:
            # 1) take the counts -= 1 if fixed and compare if the left allowed
            # 2) here it would be even more efficient to write a product that respects
            #   the allowed combinations from count, but I can't be bothered to do it atm.
            mu = itertools.product(*nonzero)
            # len(nonzero[0])*...*len(nonzero[K])
            # since we in principle know the complexity here and from the other
            # one should also do all the stuff before and then decide which way is faster
            for a in mu:
                # if allowed by count
                c = list(a)
                if np.array_equal(np.sort(c), combination):
                    v = 1
                    for j in range(len(self.monomials)):
                        if self.p_conf[c[j]][j + 1] == 0:
                            break
                        else:
                            v *= self.p_conf[c[j]][j + 1]
                    drstu += v
            return drstu

    def get_volume_from_intersections(self, ts):
        if self.nfold == 1:
            vol = np.einsum("a,a", self.intersection_tensor, ts)
        elif self.nfold == 2:
            vol = np.einsum("ab,a,b", self.intersection_tensor, ts, ts)
        elif self.nfold == 3:
            vol = np.einsum("abc,a,b,c", self.intersection_tensor, ts, ts, ts)
        elif self.nfold == 4:
            vol = np.einsum("abcd,a,b,c,d", self.intersection_tensor, ts, ts, ts, ts)
        else:
            raise NotImplementedError("Computation of intersection numbers is not supported for {}-folds".format(self.nfold))
        return vol
        
    def _implicit_diff(self, i, j):
        r"""Compute the implicit derivative of

        .. math:: dz_i/dz_j

        given the defining CY equation in the sympy polynomial 'self.poly'.

        Args:
            i (int): i index
            j (int): j index

        Returns:
            sympy poly: implicit derivative
        """
        return idiff(self.poly, self.x[i], self.x[j]) if i != j else 0

    def _frac_to_monomials(self, frac):
        r"""Takes a sympy fraction and returns tuples of
        monomials and coefficients for numerator and
        denominator.

        Args:
            frac (sympy.expr): sympy fraction

        Returns:
            ((num_basis, denom_basis), (num_factor, denom_factor))
        """
        num, den = frac.as_numer_denom()
        num_free, den_free = num.free_symbols, den.free_symbols

        # polys as dict
        num = num.as_poly().as_dict()
        den = den.as_poly().as_dict()

        # coordinate mask
        num_mask = [True if self.x[i] in num_free else False for i in range(self.ncoords)]
        den_mask = [True if self.x[i] in den_free else False for i in range(self.ncoords)]

        # initialize output
        num_monomials = np.zeros((len(num), self.ncoords), dtype=np.int32)
        denmonomials = np.zeros((len(den), self.ncoords), dtype=np.int32)
        num_factor = np.zeros(len(num), dtype=np.complex128)
        den_factor = np.zeros(len(den), dtype=np.complex128)

        # fill monomials and factors
        for i, entry in enumerate(num):
            num_monomials[i, num_mask] = entry
            num_factor[i] = num[entry]
        for i, entry in enumerate(den):
            denmonomials[i, den_mask] = entry
            den_factor[i] = den[entry]

        return ((num_monomials, denmonomials), (num_factor, den_factor))

    def _generate_padded_basis(self):
        r"""Generates a padded basis, i.e. padds the monomials in dQdz (and
        dzdz) with zeros at the end if they have uneven length to allow for
        vectorized computations.
        """
        self.BASIS = {}
        shape = np.array([np.shape(mb) for mb in self.dQdz_basis])
        DQDZB = np.zeros((len(shape), np.max(shape[:, 0]), len(shape)), dtype=np.complex64)
        DQDZF = np.zeros((len(shape), np.max(shape[:, 0])), dtype=np.complex64)
        for i, m in enumerate(zip(self.dQdz_basis, self.dQdz_factors)):
            DQDZB[i, 0:shape[i, 0]] += m[0]
            DQDZF[i, 0:shape[i, 0]] += m[1]
        if self.dzdz_generated:
            shapes = np.array([[[np.shape(t[0]), np.shape(t[1])]
                                if i != j else [[-1, -1], [-1, -1]] for i, t in enumerate(zi)]
                               for j, zi in enumerate(self.dzdz_basis)])
            DZDZB_d = np.zeros((len(shapes), len(shapes), np.max(shapes[:, :, 0, 0]), len(shapes)), dtype=np.int64)
            DZDZB_n = np.zeros((len(shapes), len(shapes), np.max(shapes[:, :, 1, 0]), len(shapes)), dtype=np.int64)
            DZDZF_d = np.zeros((len(shapes), len(shapes), np.max(shapes[:, :, 0, 0])), dtype=np.complex64)
            DZDZF_n = np.zeros((len(shapes), len(shapes), np.max(shapes[:, :, 1, 0])), dtype=np.complex64)
            for i in range(len(shapes)):
                for j in range(len(shapes)):
                    if i != j:
                        DZDZB_d[i, j, 0:shapes[i, j, 0, 0]] += self.dzdz_basis[i][j][0]
                        DZDZB_n[i, j, 0:shapes[i, j, 1, 0]] += self.dzdz_basis[i][j][1]
                        DZDZF_d[i, j, 0:shapes[i, j, 0, 0]] += self.dzdz_factor[i][j][0]
                        DZDZF_n[i, j, 0:shapes[i, j, 1, 0]] += self.dzdz_factor[i][j][1]
            self.BASIS['DZDZB_d0'] = DZDZB_d
            self.BASIS['DZDZB_n0'] = DZDZB_n
            self.BASIS['DZDZF_d0'] = DZDZF_d
            self.BASIS['DZDZF_n0'] = DZDZF_n
        self.BASIS['DQDZB0'] = DQDZB
        self.BASIS['DQDZF0'] = DQDZF
        self.BASIS['QB0'] = self.monomials
        self.BASIS['QF0'] = self.coefficients

    def generate_points(self, n_p, nproc=-1, batch_size=5000):
        r"""Generates complex points on the CY.

        The points are automatically scaled, such that the largest 
        coordinate in each projective space is 1+0.j.

        Args:
            n_p (int): # of points.
            nproc (int, optional): # of jobs used. Defaults to -1. Then
                uses all available resources.
            batch_size (int, optional): batch_size of Parallel. 
                Defaults to 5000.

        Returns:
            ndarray[(n_p, ncoords), np.complex128]: rescaled points
        """
        max_ts = np.max(self.selected_t)
        max_degree = self.ambient[self.selected_t.astype(bool)] + 1
        n_p_red = int(n_p / max_degree) + 1
        pn_pnts = np.zeros((n_p_red, self.ncoords, max_ts + 1),
                           dtype=np.complex128)
        for i in range(len(self.ambient)):
            for k in range(self.selected_t[i] + 1):
                s = np.sum(self.ambient[:i]) + i
                e = np.sum(self.ambient[:i + 1]) + i + 1
                pn_pnts[:, s:e, k] += self.generate_pn_points(n_p_red, self.ambient[i])
        # TODO: vectorize this nicely
        points = Parallel(n_jobs=nproc, backend=self.backend, batch_size=batch_size)(
            delayed(self._take_roots)(pi) for pi in pn_pnts)
        points = np.vstack(points)
        return self._rescale_points(points)

    def _generate_points_Q(self, n_p, nproc=-1, batch_size=10000):
        r"""Generates complex points using a single intersecting line 
        through *all* projective spaces. This correlates the points somewhat
        and the correct measure is currently unknown.

        NOTE: Legacy code.

        Args:
            n_p (int): # of points.
            nproc (int, optional): # of jobs used. Defaults to -1. Then
                uses all available resources.
            batch_size (int, optional): batch_size of Parallel. 
                Defaults to 10000.

        Returns:
            ndarray[(n_p, ncoords), np.complex128]: rescaled points
        """
        self._generate_root_basis_Q()
        p = np.hstack([self.generate_pn_points(int(n_p / self.ncoords) + 1, n) for n in self.ambient])
        q = np.hstack([self.generate_pn_points(int(n_p / self.ncoords) + 1, n) for n in self.ambient])

        # TODO: vectorize this nicely
        points = np.vstack(
            Parallel(n_jobs=nproc, backend=self.backend, batch_size=batch_size)(
                delayed(self._take_roots_Q)(pi, qi) for pi, qi in zip(p, q)))

        return self._rescale_points(points)

    def _rescale_points(self, points):
        r"""Rescales points in place such that for every P_i^n
        max(abs(coords_i)) == 1 + 0j.

        Args:
            points (ndarray[(n_p, ncoords), complex]): Points.

        Returns:
            ndarray[(n_p, ncoords), complex]: rescaled points
        """
        # iterate over all projective spaces and rescale in each
        for i in range(len(self.ambient)):
            s = np.sum(self.degrees[0:i])
            e = np.sum(self.degrees[0:i + 1])
            points[:, s:e] = points[:, s:e] * (points[np.arange(len(points)), s + np.argmax(np.abs(points[:, s:e]), axis=-1)].reshape((-1, 1))) ** -1
        return points

    @staticmethod
    def generate_pn_points(n_p, n):
        r"""Generates points on the sphere :math:`S^{2n+1}`.

        Args:
            n_p (int): number of points.
            n (int): degree of projective space.

        Returns:
            ndarray[(np, n+1), np.complex128]: complex points
        """
        # to not get a higher concentration from the corners of the hypercube,
        #  sample with gaussian
        points = np.random.randn(n_p, 2 * (n + 1))
        # put them on the sphere
        norm = np.expand_dims(np.linalg.norm(points, axis=-1), -1)
        # make them complex
        return (points / norm).view(dtype=np.complex128)

    def _point_from_sol(self, p, sol):
        r"""Generates a point on the CICY.

        Args:
            p (ndarray[(ncoords, t-max-deg), np.complex128]): Values 
                for points on the spheres p, q, ...
            sol (ndarray[(nHyper), np.complex]): Complex t-values.

        Returns:
            ndarray[(ncoords), np.complex128]: Point on the (CI-)CY.
        """
        # use this over point from sol sympy >100 factor improvement
        t = np.ones_like(p)
        j = 0
        for i in range(len(self.ambient)):
            for k in range(1, self.selected_t[i] + 1):
                s = np.sum(self.ambient[:i]) + i
                e = np.sum(self.ambient[:i + 1]) + i + 1
                t[s:e, k] = sol[j] * np.ones_like(t[s:e, k])
                j += 1
        point = np.sum(p * t, axis=-1)
        return point

    def _take_roots(self, p):
        r"""We generate points on Q by defining a line p*t+q 
        in *one* of the projective ambient spaces and taking all
        the intersections with Q.

        Args:
            p (ndarray[(ncoords, t-max-deg), np.complex128]): Values 
                for points on the spheres p, q, ...

        Returns:
            ndarray[(nsol, ncoords), np.complex128]: all points from the 
                intersection
        """
        all_sums = [
            np.sum(c * np.multiply.reduce(np.power(p.flatten(), m), axis=-1))
            for m, c in zip(self.root_monomials, self.root_factors)]
        roots = np.roots(all_sums)
        # we give [t] to work with more general hypersurfaces.
        return np.array([self._point_from_sol(p, [t]) for t in roots])

    def _take_roots_Q(self, p, q):
        r"""We generate points on Q by taking two points
        p, q \in A defining the line p*t+q and taking all
        the intersections with Q.

        Args:
            p (ndarray[(ncoords), np.complex128]): Points on spheres.
            q (ndarray[(ncoords), np.complex128]): Points on spheres.

        Returns:
            ndarray[(nsol, ncoords), np.complex128]: all points 
                from the intersection
        """
        all_sums = [
            np.sum(c * np.multiply.reduce(np.power(p, m[0]), axis=-1) *
                   np.multiply.reduce(np.power(q, m[1]), axis=-1))
            for m, c in zip(self.root_monomials_Q, self.root_factors_Q)]
        return np.array([p * t + q for t in np.roots(all_sums)])

    def generate_point_weights(self, n_pw, omega=False, normalize_to_vol_j=False):
        r"""Generates a numpy dictionary of point weights.

        Args:
            n_pw (int): # of point weights.
            omega (bool, optional): If True adds Omega to dict. Defaults to False.
            normalize_to_vol_j (bool, optional): Whether the weights should be normalized by the factor self.vol_j_norm.
                                                 Defaults to False

        Returns:
            np.dict: point weights
        """
        data_types = [
            ('point', np.complex128, self.ncoords),
            ('weight', np.float64)
        ]
        data_types = data_types + [('omega', np.complex128)] if omega else data_types
        dtype = np.dtype(data_types)
        points = self.generate_points(n_pw)

        # Throw away points for which the patch is ambiguous, since too many coordiantes are too close to 1
        inv_one_mask = np.isclose(points, complex(1, 0))
        bad_indices = np.where(np.sum(inv_one_mask, -1) != len(self.kmoduli))
        point_mask = np.ones(len(points), dtype=bool)
        point_mask[bad_indices] = False
        points = points[point_mask]

        n_p = len(points)
        n_p = n_p if n_p < n_pw else n_pw
        weights = self.point_weight(points, normalize_to_vol_j=normalize_to_vol_j)
        point_weights = np.zeros((n_p), dtype=dtype)
        point_weights['point'], point_weights['weight'] = points[0:n_p], weights[0:n_p]
        if omega:
            point_weights['omega'] = self.holomorphic_volume_form(points[0:n_p])
        return point_weights

    def holomorphic_volume_form(self, points, j_elim=None):
        r"""We compute the holomorphic volume form
        at all points by solving the residue theorem:

        .. math::

            \Omega &= \int_\rho \frac{1}{Q} \wedge^n dz_i \\
                   &= \frac{1}{\frac{\partial Q}{\partial z_j}}\wedge^{n-1} dz_a

        where the index a runs over the local n-fold good coordinates.

        Args:
            points (ndarray[(n_p, ncoords), np.complex128]): Points.
            j_elim (ndarray([n_p], np.int64)): index to be eliminated. 
                Defaults not None. If None eliminates max(dQdz).

        Returns:
            ndarray[(n_p), np.complex128]: Omega evaluated at each point.
        """
        indices = self._find_max_dQ_coords(points) if j_elim is None else j_elim
        omega = np.power(np.expand_dims(points, 1),
                         self.BASIS['DQDZB0'][indices])
        omega = np.multiply.reduce(omega, axis=-1)
        omega = np.add.reduce(self.BASIS['DQDZF0'][indices] * omega, axis=-1)
        # compute (dQ/dzj)**-1
        return 1 / omega

    def _find_max_dQ_coords(self, points):
        r"""Finds the coordinates for which |dQ/dz| is largest.

        Args:
            points (ndarray[(n_p, ncoords), np.complex128]): Points.

        Returns:
            ndarray[(n_p), np.int64]: max(dQdz) indices
        """
        dQdz = np.abs(self._compute_dQdz(points))
        dQdz = dQdz * (~np.isclose(points, complex(1, 0)))
        return np.argmax(dQdz, axis=-1)

    def _find_good_coordinate_mask(self, points):
        r"""Computes a mask for points with True
        in the position of the local three 'good' coordinates.

        Args:
            points (ndarray[(n_p, ncoords), np.complex128]): Points.

        Returns:
            ndarray[(n_p, ncoords), bool]: good coordinate mask
        """
        one_mask = ~np.isclose(points, complex(1, 0))
        dQdz = self._compute_dQdz(points)
        dQdz = dQdz * one_mask
        indices = np.argmax(np.abs(dQdz), axis=-1)
        dQdz_mask = -1 * np.eye(self.ncoords)[indices]
        full_mask = one_mask + dQdz_mask
        return full_mask.astype(bool)

    def _compute_dQdz(self, points):
        r"""Computes dQdz at each point.

        Args:
            points (ndarray([n_p, ncoords], np.complex128)): Points.

        Returns:
            ndarray([n_p, ncoords], np.complex): dQdz at each point.
        """
        p_exp = np.expand_dims(np.expand_dims(points, 1), 1)
        dQdz = np.power(p_exp, self.BASIS['DQDZB0'])
        dQdz = np.multiply.reduce(dQdz, axis=-1)
        dQdz = np.multiply(self.BASIS['DQDZF0'], dQdz)
        dQdz = np.add.reduce(dQdz, axis=-1)
        return dQdz

    def point_weight(self, points, normalize_to_vol_j=False, j_elim=None):
        r"""We compute the weight/mass of each point:

        .. math::

            w &= \frac{d\text{Vol}_\text{cy}}{dA}|_p \\
              &\sim \frac{|\Omega|^2}{\det(g^\text{FS}_{ab})}|_p

        the weight depends on the distribution of free parameters during 
        point sampling. We employ a theorem due to Shiffman and Zelditch. 
        See also: [9803052].

        Args:
            points (ndarray([n_p, ncoords], np.complex128)): Points.
            normalize_to_vol_j (bool, optional): Normalize such that

                .. math::

                    \int_X \det(g) &= \sum_i \det(g) \cdot w|_{x_i}\\
                                &= d^{ijk} t_i t_j t_k.

                Defaults to False.
            j_elim (ndarray([n_p, nhyper], np.int64)): Index to be eliminated. 
                Defaults to None. If None eliminates max(dQdz).

        Returns:
            ndarray([n_p], np.float64): weight at each point.
        """
        omegas = self.holomorphic_volume_form(points, j_elim=j_elim)
        pbs = self.pullbacks(points, j_elim=j_elim)
        # find the nfold wedge product of omegas
        all_omegas = self.ambient - self.selected_t
        ts = np.zeros((self.nfold, len(all_omegas)))
        j = 0
        for i in range(len(all_omegas)):
            for _ in range(all_omegas[i]):
                ts[j, i] += 1
                j += 1
        fs_pbs = []
        for t in ts:
            fs = self.fubini_study_metrics(points, vol_js=t)
            fs_pbs += [np.einsum('xai,xij,xbj->xab', pbs, fs, np.conj(pbs))]
        # do antisymmetric tensor contraction. is there a nice way to do this
        # in arbitrary dimensions? Not that anyone would study 6-folds ..
        detg_norm = 1.
        if self.nfold == 1:
            detg_norm = np.einsum('xab->x', fs_pbs[0])
        elif self.nfold == 2:
            detg_norm = np.einsum('xab,xcd,ac,bd->x',
                                  fs_pbs[0], fs_pbs[1],
                                  self.lc, self.lc)
        elif self.nfold == 3:
            detg_norm = np.einsum('xab,xcd,xef,ace,bdf->x',
                                  fs_pbs[0], fs_pbs[1], fs_pbs[2],
                                  self.lc, self.lc)
        elif self.nfold == 4:
            detg_norm = np.einsum('xab,xcd,xef,xgh,aceg,bdfh->x',
                                  fs_pbs[0], fs_pbs[1], fs_pbs[2], fs_pbs[3],
                                  self.lc, self.lc)
        elif self.nfold == 5:
            detg_norm = np.einsum('xab,xcd,xef,xgh,xij,acegi,bdfhj->x',
                                  fs_pbs[0], fs_pbs[1], fs_pbs[2], fs_pbs[3],
                                  fs_pbs[4], self.lc, self.lc)
        else:
            logger.error('Weights are only implemented for nfold <= 5.'
                         'Run the tensorcontraction yourself :).')
        omega_squared = np.real(omegas * np.conj(omegas))
        weight = np.real(omega_squared / detg_norm)
        if normalize_to_vol_j:
            fs_ref = self.fubini_study_metrics(points, vol_js=np.ones_like(self.kmoduli))
            fs_ref_pb = np.einsum('xai,xij,xbj->xab', pbs, fs_ref, np.conj(pbs))
            norm_fac = self.vol_j_norm / np.mean(np.real(np.linalg.det(fs_ref_pb)) / detg_norm)
            weight = norm_fac * weight
        return weight

    def pullbacks(self, points, j_elim=None):
        r"""Computes the pullback from ambient space to local CY coordinates
        at each point. 
        
        Denote the ambient space coordinates with z_i and the CY
        coordinates with x_a then

        .. math::

            J^i_a = \frac{dz_i}{dx_a}

        Args:
            points (ndarray([n_p, ncoords], np.complex128)): Points.
            j_elim (ndarray([n_p, nhyper], np.int64)): Index to be eliminated. 
                Defaults to None. If None eliminates max(dQdz).

        Returns:
            ndarray([n_p, nfold, ncoords], np.complex128): Pullback tensor 
                at each point.
        """
        inv_one_mask = ~np.isclose(points, complex(1, 0))
        if j_elim is None:
            j_elim = self._find_max_dQ_coords(points)
        if len(j_elim.shape) == 1:
            j_elim = np.reshape(j_elim, (-1, 1))
        full_mask = np.copy(inv_one_mask)
        for i in range(self.nhyper):
            full_mask[np.arange(len(points)), j_elim[:, i]] = np.zeros(len(points), dtype=bool)

        # fill the diagonal ones in pullback
        x_indices, z_indices = np.where(full_mask)
        pullbacks = np.zeros((len(points), self.nfold, self.ncoords), dtype=np.complex128)
        y_indices = np.repeat(np.expand_dims(np.arange(self.nfold), 0), len(points), axis=0)
        y_indices = np.reshape(y_indices, (-1))
        pullbacks[x_indices, y_indices, z_indices] = np.ones(self.nfold * len(points), dtype=np.complex128)
        # next fill the dzdz from every hypersurface
        B_matrix = np.zeros((len(points), self.nhyper, self.nhyper), dtype=np.complex128)
        dz_hyper = np.zeros((len(points), self.nhyper, self.nfold), dtype=np.complex128)
        fixed_indices = np.reshape(j_elim, (-1))
        for i in range(self.nhyper):
            # compute p_i\alpha eq (5.24)
            pia_polys = self.BASIS['DQDZB' + str(i)][z_indices]
            pia_factors = self.BASIS['DQDZF' + str(i)][z_indices]
            pia = np.power(np.expand_dims(
                np.repeat(points, self.nfold, axis=0), 1), pia_polys)
            pia = np.multiply.reduce(pia, axis=-1)
            pia = np.add.reduce(np.multiply(pia_factors, pia), axis=-1)
            pia = np.reshape(pia, (-1, self.nfold))
            dz_hyper[:, i, :] += pia
            # compute p_ifixed
            pif_polys = self.BASIS['DQDZB' + str(i)][fixed_indices]
            pif_factors = self.BASIS['DQDZF' + str(i)][fixed_indices]
            pif = np.power(np.expand_dims(
                np.repeat(points, self.nhyper, axis=0), 1), pif_polys)
            pif = np.multiply.reduce(pif, axis=-1)
            pif = np.add.reduce(np.multiply(pif_factors, pif), axis=-1)
            pif = np.reshape(pif, (-1, self.nhyper))
            B_matrix[:, i, :] += pif
        all_dzdz = np.einsum('xij,xjk->xki',
                             np.linalg.inv(B_matrix),
                             complex(-1., 0.) * dz_hyper)
        for i in range(self.nhyper):
            pullbacks[np.arange(len(points)), :, j_elim[:, i]] += all_dzdz[:, :, i]
        return pullbacks

    def _pullbacks_dzdz(self, points, j_elim=None):
        r"""Computes the pullback from ambient space to local CY coordinates
        at each point. 
        
        Denote the ambient space coordinates with z_i and the CY
        coordinates with x_a then

        .. math::

            J^i_a = \frac{dz_i}{dx_a}

        NOTE: Uses the implicit derivatives dz/dz rather than inverse matrix.

        Args:
            points (ndarray([n_p, ncoords], np.complex128)): Points.
            j_elim (ndarray([n_p], np.int)): index to be eliminated. 
                Defaults not None. If None eliminates max(dQdz).

        Returns:
            ndarray([n_p, nfold, ncoords], np.complex128): Pullback tensor
                at each point.
        """
        if not self.dzdz_generated:
            # This will take some time when called for the first time.
            self.dzdz_generated = True
            self._generate_dzdz_basis()
            self._generate_padded_basis()
        one_mask = ~np.isclose(points, complex(1, 0))
        if j_elim is None:
            dQdz = self._compute_dQdz(points)
            dQdz = dQdz * one_mask
            dQdz_indices = np.argmax(np.abs(dQdz), axis=-1)
        else:
            dQdz_indices = j_elim
        dQdz_mask = np.eye(self.ncoords)[dQdz_indices]
        full_mask = one_mask - dQdz_mask
        full_mask = full_mask.astype(bool)
        x_indices, z_indices = np.where(full_mask)
        nrepeat = self.nfold
        dQdz_indices = np.repeat(dQdz_indices, nrepeat)

        # compute everything
        numerators = self.BASIS['DZDZB_n0'][dQdz_indices, z_indices]
        num_factors = self.BASIS['DZDZF_n0'][dQdz_indices, z_indices]
        denominators = self.BASIS['DZDZB_d0'][dQdz_indices, z_indices]
        den_factors = self.BASIS['DZDZF_d0'][dQdz_indices, z_indices]
        num_res = np.power(np.expand_dims(
            np.repeat(points, nrepeat, axis=0), 1), numerators)
        num_res = np.multiply.reduce(num_res, axis=-1)
        num_res = np.add.reduce(np.multiply(num_factors, num_res), axis=-1)
        den_res = np.power(np.expand_dims(
            np.repeat(points, nrepeat, axis=0), 1), denominators)
        den_res = np.multiply.reduce(den_res, axis=-1)
        den_res = np.add.reduce(np.multiply(den_factors, den_res), axis=-1)
        all_dzdz = num_res / den_res

        # fill everything
        x_indices = np.concatenate((x_indices, x_indices))
        z_indices = np.concatenate((z_indices, dQdz_indices))
        y_indices = np.repeat(np.expand_dims(
            np.arange(nrepeat), 0), len(points), axis=0)
        y_indices = np.reshape(y_indices, (-1))
        y_indices = np.concatenate((y_indices, y_indices))
        all_values = np.concatenate(
            (np.ones(nrepeat * len(points), dtype=np.complex128), all_dzdz),
            axis=0)
        pullbacks = np.zeros(
            (len(points), nrepeat, self.ncoords), dtype=np.complex128)
        pullbacks[x_indices, y_indices, z_indices] = all_values
        return pullbacks

    def compute_kappa(self, points, weights, omegas):
        r"""We compute kappa from the Monge-Ampère equation

        .. math:: J^3 = \kappa |\Omega|^2
        
        such that after integrating we find

        .. math::

            \kappa = \frac{J^3}{|\Omega|^2} =
                \frac{\text{Vol}_K}{\text{Vol}_{\text{CY}}}

        Args:
            points (ndarray[(n_p, ncoords), np.complex128]): Points.
            weights (ndarray[n_p, np.float64]): weights of the points.
            omegas (ndarray[n_p, np.complex128]): Omega \wedge Omega* of the points.

        Returns:
            np.float: kappa
        """
        weights, omegas = weights.flatten(), omegas.flatten()
        pbs = self.pullbacks(points)
        gFS = self.fubini_study_metrics(points)
        gFS_pbs = np.einsum('xai,xij,xbj->xab', pbs, gFS, np.conj(pbs))
        dets = np.real(np.linalg.det(gFS_pbs))

        vol_k = np.mean(weights * dets / omegas)
        vol_cy = np.mean(weights)
        logger.info('Vol_k: {}, Vol_cy: {}.'.format(vol_k, vol_cy))
        kappa = vol_k / vol_cy
        return kappa

    def fubini_study_metrics(self, points, vol_js=None):
        r"""Computes the FS metric at each point.

        Args:
            points (ndarray[(n_p, ncoords), np.complex128]): Points.
            vol_js (ndarray[(h^{(1,1)}), np.complex128]): vol_j factor.
                Defaults to None.

        Returns:
            ndarray[(n_p, ncoords, ncoords), np.complex128]: g^FS
        """
        gFS = np.zeros((len(points), self.ncoords, self.ncoords),
                       dtype=np.complex128)
        kmoduli = self.kmoduli if vol_js is None else vol_js
        for i in range(len(self.ambient)):
            s = np.sum(self.degrees[0:i])
            e = np.sum(self.degrees[0:i + 1])
            gFS[:, s:e, s:e] += self._fubini_study_n_metrics(points[:, s:e], vol_j=kmoduli[i])
        return gFS

    @staticmethod
    def _fubini_study_n_metrics(points, vol_j=1. + 0.j):
        r"""Computes the FS metric for a single projective space of points.

        Args:
            point (ndarray[(n_p, n), np.complex128]): Points.
            vol_j (complex): Volume factor. Defaults to 1+0.j.

        Returns:
            ndarray[(n_p, n, n), np.complex128]: g^FS
        """
        # we want to compute d_i d_j K^FS
        point_square = np.add.reduce(np.abs(points) ** 2, axis=-1)
        outer = np.einsum('xi,xj->xij', np.conj(points), points)
        gFS = np.einsum('x,ij->xij', point_square, np.eye(points.shape[1]))
        gFS = gFS.astype(np.complex128) - outer
        return np.einsum('xij,x->xij', gFS, 1 / (point_square ** 2)) * vol_j / np.pi

    def prepare_basis(self, dirname, kappa=1.):
        r"""Prepares pickled monomial basis for the tensorflow models.

        Args:
            dirname (str): dir name to save

        Returns:
            int: 0
        """
        return prepare_basis_pickle(self, dirname, kappa)

    def prepare_dataset(self, n_p, dirname, val_split=0.1, ltails=0, rtails=0):
        r"""Prepares training and validation data.

        Args:
            n_p (int): Number of points to generate.
            dirname (str): Directory name to save dataset in.
            val_split (float, optional): train-val split. Defaults to 0.1.
            ltails (float, optional): Percentage discarded on the left tail
                of weight distribution. Defaults to 0.
            rtails (float, optional): Percentage discarded on the right tail
                of weight distribution. Defaults to 0.

        Returns:
            np.float: kappa = vol_k / vol_cy
        """
        return prepare_dataset(self, n_p, dirname, val_split=val_split, ltails=ltails, rtails=rtails)

    def cy_condition(self, points):
        r"""Computes the CY condition at each point.

        Args:
            points (ndarray([n_p, ncoords], np.complex128)): Points (on the CY).

        Returns:
            ndarray(n_p, np.complex128): CY condition
        """
        cy_condition = np.power(np.expand_dims(points, 1), self.monomials)
        cy_condition = np.multiply.reduce(cy_condition, axis=-1)
        cy_condition = np.add.reduce(self.coefficients * cy_condition, axis=-1)
        return cy_condition

    def __call__(self, points, vol_js=None):
        r"""Computes the FS metric at each point.

        Args:
            points (ndarray[(n_p, ncoords), np.complex128]): Points.
            vol_js (ndarray[(h^{(1,1)}), np.complex128]): vol_j factors. 
                Defaults to None.

        Returns:
            ndarray[(n_p, ncoords, ncoords), np.complex128]: g^FS
        """
        return self.fubini_study_metrics(points, vol_js=vol_js)
