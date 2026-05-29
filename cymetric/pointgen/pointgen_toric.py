"""
ToricPointGenerator module.

:Authors:
    Fabian Ruehle f.ruehle@northeastern.edu
"""
import numpy as np
import logging
import sympy as sp
import scipy.optimize as opt
from joblib import Parallel, delayed
import random
from cymetric.pointgen.nphelper import prepare_dataset
from cymetric.pointgen.pointgen import PointGenerator
from cymetric.pointgen.nphelper import generate_monomials, get_all_patch_degrees
logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s')
logger = logging.getLogger('toricpointgen')


class ToricPointGenerator(PointGenerator):
    r"""ToricPointGenerator class.

    The numerics are entirely done in numpy.

    Use this class if you want to generate points and data on a toric CY
    given by one hypersurface as in the Kreuzer-Skarke list.
    
    Example:
        We assume toric_data has been generated using the sage lib beforehand.
        Check https://doc.sagemath.org/html/en/reference/schemes/sage/schemes/toric/variety.html
        for infos how to construct ToricVarieties/Fans in sage.

        Define in sage your favourite polytope.

        >>> from cymetric.sage.sagelib import prepare_toric_cy_data
        >>> # [...] generate fan from triangulations from vertices 
        >>> TV = ToricVariety(fan)
        >>> fname = "toric_data.pickle"
        >>> toric_data = prepare_toric_cy_data(TV, fname)

        then we can start another python kernel or continue in sage with

        >>> import numpy as np
        >>> from cymetric.pointgen.pointgen_toric import ToricPointGenerator
        >>> kmoduli = np.ones(len(toric_data['exps_sections']))
        >>> # [...] load toric_data with pickle
        >>> pg = ToricPointGenerator(toric_data, kmoduli)
        
        Once the ToricPointGenerator is initialized you can generate a training
        dataset with 

        >>> pg.prepare_dataset(number_of_points, dir_name) 

        and prepare the required tensorflow model data with 
        
        >>> pg.prepare_basis(dir_name)
        
    """
    def __init__(self, toric_data, kmoduli, **kwargs):
        r"""Initializer.

        ``**kwargs`` may contain the usual key-arguments from `PointGenerator`.

        Args:
            toric_data (dict): generated from 
                >>> sage_lib.prepare_toric_cy_data()
            kmoduli (ndarray[(h^{(1,1)})]): The kaehler moduli.
        """
        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
            if verbose == 1:
                level = logging.DEBUG
            elif verbose == 2:
                level = logging.INFO
            else:
                level = logging.WARNING
            logger.setLevel(level=level)
        logger.warning("This is outdated code. The integration weight computation\
            has not been updated yet to match the ToricPointGeneratorMathematica.\
            In it's current form the integration weights are computed wrongly and\
            won't give you the correct volume in the MC integration. \
            Get involved and translate the Mathematica functions to python!\
            Proceed on your own risk, you have been warned.")
        self.toric_data = toric_data
        self.sections = [np.array(m) for m in self.toric_data['exps_sections']]
        self.nsections = len(self.sections)
        self.num_sections = [np.shape(m) for m in self.sections]
        self.patch_masks = np.array(self.toric_data['patch_masks'],
                                    dtype=bool)
        self.glsm_charges = np.array(self.toric_data["glsm_charges"])
        self.nPatches = len(self.patch_masks)
        self.nProjective = len(self.toric_data["glsm_charges"])
        self.patch_degrees = get_all_patch_degrees(self.glsm_charges,
                                                   self.patch_masks)
        # ambient degree of sections.
        self.dim_ps = np.array([s for s, _ in self.num_sections])
        if 'dzdz' in kwargs:
            # NOTE: Don't use this. some issues with sympys implicit derivative.
            self.dzdz_generated = kwargs['dzdz']
            del kwargs['dzdz']
        else:
            self.dzdz_generated = False
        # need to give it some ambient, but all function including it will be overwritten
        fambient = np.array([toric_data['dim_cy']+1])
        super(ToricPointGenerator, self).__init__(
            monomials=np.array(self.toric_data['exp_aK']),
            coefficients=np.array(self.toric_data['coeff_aK']),
            kmoduli=kmoduli,
            ambient=fambient, **kwargs)
        # TODO: make a better guess with GLSM charges
        # HACK: We use self.selected_t differently to CICY pointgen
        # self.ambient is not used other than in point_weight.
        self.ambient = 2*self.selected_t
        self.vol_j_norm = toric_data['vol_j_norm']

    def _generate_all_bases(self):
        r"""This function calls a bunch of others
        which then generate various monomial basis
        needed for point generation, residue theorem
        and the Jacobian tensor.
        """
        self.all_ts = self._generate_all_freets()
        self.selected_t = self._find_degrees()
        self._generate_dQdz_basis()
        if self.dzdz_generated:
            self._generate_dzdz_basis()
        self._generate_padded_basis()

    def _generate_all_freets(self):
        r"""Generates all possible ways for nfold-t-distribution."""
        # this will blow up for large h11, then either disable or 
        # implement a more efficient way.
        free_ts = list(generate_monomials(len(self.kmoduli), self.nfold))
        free_ts = np.array(free_ts)
        good = np.reshape(self.dim_ps-1, (1, -1)) - free_ts >= 0
        good = np.logical_and.reduce(good, axis=-1)
        return free_ts[good]

    def _root_polynomial(self, x_guess, p, patch_mask):
        r"""Function to be optimized by scipy.opt.fsolve.
        Computes the difference from zero when plugging proposed solution
        into CY and section polynomials

        NOTE:
            Takes real arguments in x and returns real output as [Re, Im] of
            the complex computations.

        Args:
            x (ndarray[ncoords, np.float64]): Estimated solution coordinates
            p (ndarray[(sum(dim_ps), t-max-deg), np.complex128]): Values 
                for points on the spheres p, q, ...
            patch_mask (ndarray[ncoords, np.float64]): 1 if coordinate can
                be scaled to 1.

        Returns:
            ndarray[2*ncoords, np.float64]: difference from zero.
        """
        x = x_guess.view(np.complex128)
        patches = np.where(patch_mask, x-patch_mask, np.zeros(self.ncoords))
        eqs = [np.add.reduce(self.coefficients * np.multiply.reduce(np.power(x, self.monomials), axis=-1), axis=-1)]
        for j, t in enumerate(self.selected_t):
            section_monom = np.power(x, self.sections[j])
            section_monom = np.multiply.reduce(section_monom, axis=-1)
            for i in range(t):
                s = np.sum(self.dim_ps[:j])
                e = np.sum(self.dim_ps[:j+1])
                coeff = p[s:e, i]
                new_eq = np.add.reduce(coeff * section_monom, axis=-1)
                eqs += [new_eq]
        final_eq = eqs
        patches[~(patch_mask.astype(bool))] += np.array(final_eq)
        return patches.view(np.float64)

    def _root_polynomial_old(self, x_guess, patch_mask, coeffs):
        r"""Function to be optimized by scipy.opt.fsolve.
        Computes the difference from zero when plugging proposed solution
        into CY and section polynomials.
        
        NOTE:
            Takes real arguments in x and returns real output as [Re, Im] of
            the complex computations.

        Args:
            x (ndarray[num_eqns, np.float]): estimated solution coordinates
            patch_mask (ndarray[num_coords, np.float]): 1 if coordinate can 
                be scaled to 1
            coeffs (list[(num_sections, 2*num_monoms_in_sections, np.float]): 
                random real and imaginary part of coefficients to build 
                random sections of kahler cone divisors.

        Returns:
            ndarray[2*num_eqns, np.float]: difference from zero.
        """
        x = x_guess.view(np.complex128)
        eqs = [np.add.reduce(self.coefficients * np.multiply.reduce(np.power(x, self.monomials), axis=-1), axis=-1)]
        num_eqns_in_pn = np.zeros(self.nsections, dtype=np.int32)
        # TODO: vectorize this.
        for i, (e, p) in enumerate(zip(x, patch_mask)):
            if p == 1:
                eqs += [e - 1]
                continue
            if sum(num_eqns_in_pn) == self.ncoords - self.nsections - 1:
                continue
            for jj in range(self.nsections):
                section_monom = [np.multiply.reduce(np.power(x, s), axis=-1) for s in self.sections[jj]]
                if num_eqns_in_pn[jj] >= self.dim_ps[jj]:
                    continue
                tmp_mask = np.sum(self.sections[jj], axis=-2)
                if tmp_mask[i] != 0:
                    coeff = coeffs[jj][num_eqns_in_pn[jj]]
                    # TODO vectorize this.
                    new_eq = np.sum([(coeff[k, 0] + 1.j * coeff[k, 1]) * section_monom[k] for k in range(len(section_monom))])
                    eqs += [new_eq]
                    num_eqns_in_pn[jj] += 1
                    break
        return np.array(eqs).view(np.float64)

    def generate_points(self, n_p, nproc=-1, nattempts=1, acc=1e-8, fprime=None, batch_size=1000):
        r"""Generates complex points using scipy's optimizer fsolve.

        Args:
            n_p (int): # of points.
            nproc (int, optional): # of jobs used. Defaults to -1. Then
                uses all available resources.
            nattempts (int, optional): # of tries to find a solution given a 
                set of points on the ambient spheres. Defaults to 1.
            acc (float, optional): Required CY accuracy. Defaults to 1e-8.
            fprime (function, optional): fprime of CY conditions with float
                returns.
            batch_size (int, optional): batch_size of Parallel.
                Defaults to 1000.
            
        Returns:
            nd.array[(n_p, ncoord), np.complex128]: points on CY
        """
        logger.debug("Generating {:} points...".format(n_p))
        sphere_points = self._get_sphere_points(n_p)
        points = Parallel(n_jobs=nproc, batch_size=batch_size, backend=self.backend)(delayed(self._get_point)(pi, acc, nattempts, fprime) for pi in sphere_points)
        points = np.array(points)
        # remove points for which opt didnt converge to zero
        cy_cond = self.cy_condition(points)
        cy_mask = np.abs(cy_cond) < acc
        logger.debug("{} out of {} solutions are on the CY with acc {}.".format(
            np.sum(cy_mask), n_p, acc))
        points = points[cy_mask]
        if np.sum(cy_mask) < n_p:
            ratio = n_p/np.sum(cy_mask)
            for _ in range(5):
                # hopefully only need one iteration, but might get unlucky
                missing = n_p-len(points)
                n_p_more = int(ratio*missing + 50)
                logger.debug('generating {} more points.'.format(n_p_more))
                new_sphere_points = self._get_sphere_points(n_p_more)
                new_points = Parallel(n_jobs=nproc, batch_size=batch_size, backend=self.backend)(delayed(self._get_point)(pi, acc, nattempts, fprime) for pi in new_sphere_points)
                new_points = np.array(new_points)
                cy_cond = self.cy_condition(new_points)
                cy_mask = np.abs(cy_cond) < acc
                new_points = new_points[cy_mask]
                logger.debug('found {} out of {} expected.'.format(
                    len(new_points), missing))
                if len(points) + len(new_points) > n_p:
                    points = np.concatenate(
                        [points, new_points[0:n_p-len(points)]])
                    break
                else:
                    points = np.concatenate([points, new_points])
        return self._rescale_points(points)

    def _get_sphere_points(self, n_p):
        r"""Creates points on the ambient sphere.
        
        Args:
            n_p (int): # of points.

        Returns:
            ndarray[(n_p, sum(dim_ps), t-max_deg), np.complex128]: All ambient
                sphere.
        """
        max_deg = np.max(self.selected_t)
        # NOTE: if there are allowed of section this might become too large.
        # We then have to further batch the generate point function.
        pn_pnts = np.zeros((n_p, np.sum(self.dim_ps), max_deg),
                           dtype=np.complex128)
        for i in range(len(self.selected_t)):
            for k in range(self.selected_t[i]):
                s = np.sum(self.dim_ps[:i])
                e = np.sum(self.dim_ps[:i+1])
                pn_pnts[:, s:e, k] += self.generate_pn_points(n_p, self.dim_ps[i]-1)
        return pn_pnts

    def _get_point(self, p, acc=1e-8, nattempts=1, fprime=None):
        r"""Generates a single point on the CY for given points
         on the ambient spheres.

        NOTE:
            It is not guaruanteed that fsolve converges to a point on the CY.

        Args:
            p (ndarray[(sum(dim_ps), t-max-deg), np.complex128]): Values 
                for points on the spheres p, q, ...
            acc (float, optional): Accuracy for accepted solution. 
                Defaults to 1e-8.
            nattempts (int, optional): Attempts to find a solution with given
                accuracy. Defaults to 1.
            fprime (function, optional): function for fprime in opt.fsolve.
                Defaults to None.

        Returns:
            ndarray[(ncoords), np.complex128]: (potential) Point on the CY.
        """
        # random coefficients to build random sections of Kahler cone divisors
        best_sol = np.random.randn(2*self.ncoords).view(np.complex128)
        best_acc = self.cy_condition(best_sol[np.newaxis, :])[0]
        for _ in range(nattempts):
            patch_mask = random.choice(self.patch_masks)
            sol = opt.fsolve(
                self._root_polynomial, np.random.randn(2*self.ncoords),
                args=(p, patch_mask), fprime=fprime)
            sol_c = sol.view(np.complex128)
            acc1 = self.cy_condition(sol_c[np.newaxis, :])[0]
            if acc1 < best_acc:
                best_acc = acc1
                best_sol = sol_c
                if acc1 < acc:
                    break
        return best_sol

    def _find_degrees(self):
        r"""Generates t-degrees in each section.
        We give one to each section and the remaining to the largest
        pseudo ambient space.
        """
        degrees = np.zeros(len(self.num_sections), dtype=np.int32)
        available_degrees = np.array(self.num_sections)
        for _ in range(int(self.nfold)):
            largest_proj = np.argmax(available_degrees[:, 0])
            degrees[largest_proj] += 1
            available_degrees[largest_proj, 0] -= 1
        return degrees
    
    def _scale_point_old(self, point):
        r"""Use toric scaling to put the point in a patch s.t. the largest
        coordinate is 1. We do this by trying all toric patches.

        NOTE:
            This function symbolically solves for each point and is thus slow.
            Use '_rescale_points()', which is vectorized and does not
            require sympy, instead.

        Args:
            point (ndarray[(ncoords), np.complex128]): Single point on the CY 
                in the toric ambient space.

        Returns:
            ndarray[(ncoords), np.complex128]: scaled point.
        """
        lambdas = sp.var('l0:'+str(len(self.glsm_charges)))
        for p in self.patch_masks:
            patch_coords = [i for i in range(len(p)) if p[i] == 1]
            scalings = np.ones(len(p)).tolist()
            for l, qs in zip(lambdas, self.glsm_charges):
                scalings = [s * l**q for s, q in zip(scalings, qs)]
            eq = [1 - point[i] * scalings[i] for i in patch_coords]
            sol = sp.solvers.solve(eq, lambdas, dict=True)
            if len(sol) > 0:
                sol = [complex(s) for s in list(sol[0].values())]
                scalings = np.ones(len(p))
                for l, qs in zip(sol, self.glsm_charges):
                    scalings = [s * l**q for s, q in zip(scalings, qs)]
                tmp_point = point * scalings
                if np.isclose(np.max(np.abs(tmp_point)), 1.):
                    return tmp_point
        logger.debug("Could not scale point {}.".format(point))
        return point

    def _get_patch_coordinates(self, points, patch_index):
        r"""Computes coordinates in patch specified by patch_index.
        
        Args:
            points (ndarray([n_p, ncoords], np.complex128)): Points.
            patch_index (ndarry([n_p], np.int64)): Patch indices.

        Returns:
            ndarray([n_p, ncoords], np.complex128): Rescaled coordinates.
        """
        degrees = self.patch_degrees[patch_index]
        scaled_points = points[:, np.newaxis, :]
        scaled_points = np.power(scaled_points, degrees)
        return np.multiply.reduce(scaled_points, axis=-1)

    def _rescale_points(self, points):
        r"""Rescales points to patches, such that largest coordinate *and*
        patch coordinate is 1+0.j.

        Args:
            points (ndarray([n_p, ncoords], np.complex128)): Points.

        Returns:
            ndarray([n_p, ncoords], np.complex128): Rescaled points.
        """
        scaled_points = np.zeros(points.shape, dtype=points.dtype)
        missing_points = np.ones(len(points), dtype=bool)
        for i in range(len(self.patch_masks)):
            tmp_points = self._get_patch_coordinates(points[missing_points],
                                                     np.zeros(np.sum(missing_points), dtype=np.int32)+i)
            abs_points = np.logical_and.reduce(np.abs(tmp_points) < 1.0001, axis=-1)
            # update
            scaled_points[missing_points] = np.where(abs_points[:, np.newaxis], tmp_points, scaled_points[missing_points])
            missing_points[missing_points] = np.where(abs_points, False, True)
            logger.debug("Rescaled {} ouf of {} points.".format(
                np.sum(~missing_points), len(scaled_points)))
            if np.sum(missing_points) == 0:
                break
        return scaled_points

    def fubini_study_metrics(self, points, vol_js=None):
        r"""Computes the toric equivalent to FS metric at points.

        Args:
            points (ndarray[(n_p, ncoords), np.complex128]): Points.
            vol_js (ndarray[(h^{(1,1)}), np.complex128]): vol_j factor.

        Returns:
            ndarray[(len(points), ncoords, ncoords), np.complex128]: g^FS
        """
        kfactors = self.kmoduli if vol_js is None else vol_js
        Js = np.zeros([len(points), self.ncoords, self.ncoords], dtype=np.complex128)
        for alpha in range(len(kfactors)):
            degrees = self.sections[alpha]
            ms = np.power(points[:,np.newaxis,:], degrees[np.newaxis,:,:])
            ms = np.multiply.reduce(ms, axis=-1)
            mss = ms*np.conj(ms)
            kappa_alphas = np.sum(mss, -1)
            point_sq = points[:, :, np.newaxis]*np.conj(points[:, np.newaxis, :])
            J_alphas = 1/point_sq
            J_alphas = np.einsum('x,xab->xab', 1/(kappa_alphas**2), J_alphas)
            degrees = degrees.astype(points.dtype)
            coeffs = np.einsum('xa,xb,ai,aj->xij', mss, mss, degrees, degrees)
            coeffs -= np.einsum('xa,xb,ai,bj->xij', mss, mss, degrees, degrees)
            Js += J_alphas * coeffs * complex(kfactors[alpha]) / np.pi
        return Js

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
        return prepare_dataset(self, n_p, dirname, val_split=val_split, ltails=ltails, rtails=rtails, normalize_to_vol_j=True)
