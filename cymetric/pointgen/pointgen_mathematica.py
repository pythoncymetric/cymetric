"""
Class to call Mathematica point generator. 

Authors
-------
Fabian Ruehle fabian.ruehle@cern.ch
Robin Schneider robin.schneider@physics.uu.se
"""
import os
import pickle
import numpy as np
import sympy as sp
import decimal
import re

from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr
from wolframclient.deserializers import WXFConsumer, binary_deserialize
from wolframclient.language import Global as wlGlobal
from wolframclient.serializers import export as wlexport
from cymetric.pointgen.pointgen_cicy import CICYPointGenerator
from cymetric.pointgen.nphelper import get_levicivita_tensor, prepare_dataset

import logging
logger = logging.getLogger('pointgenMathematica')
logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s')


class PointGeneratorMathematica(CICYPointGenerator):
    r"""PointGeneratorMathematica class.

    This uses mathematica as a backand to carry out the computations

    Example:
        A simple example on a generic CY manifold of the family defined by the following configuration matrix:

        .. math::
            X \in [5|33]

        can be set up with

        >>> from cymetric.pointgen.pointgen_mathematica import PointGeneratorMathematica
        >>> from cymetric.pointgen.nphelper import generate_monomials
        >>> monomials = np.array(list(generate_monomials(6, 3)))
        >>> monomials_per_hyper = [monomials, monomials]
        >>> coeff = [np.random.randn(len(m)) for m in monomials_per_hyper]
        >>> kmoduli = np.ones(1)
        >>> ambient = np.array([5])
        >>> pg = PointGeneratorMathematica(monomials_per_hyper, coeff, kmoduli, ambient)

        Once PointGeneratorMathematica is initialized you can generate a training dataset with

        >>> pg.prepare_dataset(number_of_points, dir_name)

        and prepare the required tensorflow model data with

        >>> pg.prepare_basis(dir_name)
    """

    def __init__(self, *args, **kwargs):
        r"""PointGeneratorMathematica uses Mathematica as a backend for computations.

            Args:
                monomials (list(ndarray[(nMonomials, ncoord), np.int])): list of length nHyper with monomials for each
                                                                         defining equation.
                coefficients (list(ndarray[(nMonomials)])): list of coefficients in front of each monomial.
                kmoduli (ndarray[(nProj)]): The kaehler moduli.
                ambient (ndarray[(nProj), np.int]): the direct product of  projective spaces making up the ambient space.
                vol_j_norm (float, optional): Normalization of the volume of the Calabi-Yau X as computed from
                    .. math:: \int_X J^n \; \text{ at } \; t_1=t_2=...=t_n = 1.
                    Defaults to 1.
                verbose (int, optional): Controls logging. 1-Debug, 2-Info,  else Warning. Defaults to 2.
                precision (int, optional): Number of valid digits. Defaults to 10
                point_file_path (str, optional): Path where points are stored. This is only important if Mathematica
                                                 is also used as a frontend
                selected_t (ndarray[(nProj)]): The ambient spaces from which the points were sampled.
                                               This is only important if Mathematica is also used as a frontend
        """
        self.precision = kwargs.get('precision', 10)
        if 'precision' in kwargs.keys():
            del kwargs['precision']
        self.point_file_path = kwargs.get('point_file_path', None)
        if 'point_file_path' in kwargs.keys():
            del kwargs['point_file_path']
        
        selected_t = kwargs.get('selected_t', None)
        if 'selected_t' in kwargs.keys():
            del kwargs['selected_t']
            
        super(PointGeneratorMathematica, self).__init__(*args, **kwargs)
        
        # NOTE: This is computed in the constructor, but we need to use the distribution that the Mathematica point gen used
        self.selected_t = selected_t if selected_t is not None else np.zeros((len(self.ambient)), dtype=np.int)
        
        self.verbose = kwargs.get('verbose', 1)
        if self.verbose == 1:
            self.level = logging.DEBUG
        elif self.verbose == 2:
            self.level = logging.INFO
        else:
            self.level = logging.WARNING
        logger.setLevel(level=self.level)
        self.wl_session = None

    def __del__(self):
        try:
            self.wl_session.terminate()
        except:
            pass
    
    @staticmethod
    def _setup_session(mathematica_session):
        # surpress some of the debug messages, they are a bit too much
        mathematica_session.evaluate(wlexpr('ClientLibrary`SetErrorLogLevel[]'))
        # read in mathematica functions as string
        file_name = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../wolfram/PointGeneratorMathematica.m')
        with open(file_name, 'r') as file:
            mathematica_functions_str = file.read().replace('\n', '')

        # Compile functions in mathematica session
        mathematica_session.evaluate(wlexpr(mathematica_functions_str))
        
    @staticmethod
    def _start_parallel_kernels(mathematica_session, nproc=-1):
        num_kernels = mathematica_session.evaluate(wlexpr('If[$ConfiguredKernels!={},$ConfiguredKernels[[1, 1]],-1]'))
        if nproc <= 0 or nproc > num_kernels:
            nproc = ''
        
        # close all parallel Kernels
        mathematica_session.evaluate(wlexpr('Quiet[CloseKernels[]]'))
        # start multiple kernels for parallel evaluation
        mathematica_session.evaluate(wlexpr('Quiet[WaitAll[LaunchKernels[' + str(nproc) + ']]];'))
        logger.debug("Running with " + str(max(1, mathematica_session.evaluate(wlexpr('Length[Kernels[]]')))) + " Mathematica kernels.")
    
    def generate_points(self, n_p, nproc=-1):
        r"""Generates complex points by calling the mathematica point generator

        Args:
            n_p (int): # of points
            nproc (int, optional): # of processes used. Defaults to -1.

        Returns:
            nd.array[(np, ncoord), np.complex128]: [description]
        """
        with WolframLanguageSession(kernel_loglevel=self.level) as mathematica_session:
            self.wl_session = mathematica_session
            self._setup_session(mathematica_session)
            self._start_parallel_kernels(mathematica_session, nproc)
        
            # Define Python function from Mathematica function
            get_points_mathematica = mathematica_session.function(wlGlobal.GeneratePointsM)
        
            # Compute the points
            # specifying the consumer in the session with WolframLanguageSession(consumer=ComplexFunctionConsumer()) seems to not work as of wolframclient version 1.1.6
            # so we need to deserialize the object without a consumer, serialize it to an wxf expression, and deserialize that with a consumer

            # compute and deserialize the points
            pts = get_points_mathematica(n_p, self.ambient.tolist(), [x.tolist() for x in self.coefficients], [x.tolist() for x in self.monomials], self.precision, self.verbose, False)
            # serialize again to wxf
            pts = wlexport(pts, target_format='wxf')
            # deserialize converting Mathamtica complex numbers to python complex numbers
            pts = binary_deserialize(pts, consumer=ComplexFunctionConsumer())
        
        self.selected_t = np.array(pts[1], dtype=np.int)
        return np.array(pts[0])


class ToricPointGeneratorMathematica(PointGeneratorMathematica):
    r"""ToricPointGeneratorMathematica class.

    The numerics are done in mathematica, the toric computations have been done in SageMath.

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
        >>> from cymetric.pointgen.pointgen_mathematica import PointGeneratorMathematica
        >>> kmoduli = np.ones(len(toric_data['exps_sections']))
        >>> # [...] load toric_data with pickle
        >>> pg = PointGeneratorMathematica(toric_data, kmoduli)

        Once the ToricPointGenerator is initialized you can generate a training dataset with

        >>> pg.prepare_dataset(number_of_points, dir_name)

        and prepare the required tensorflow model data with

        >>> pg.prepare_basis(dir_name)

    """
    def __init__(self, toric_data, kmoduli,  verbose=2, precision=10, point_file_path=None, selected_t=None):
        r"""PointGeneratorMathematica uses Mathematica as a backend for computations.

             Args:
                 toric_data (np.ndarray): Toric data generated from sage by calling prepare_toric_cy_data(TV, fname)
                 kmoduli (ndarray[(nProj)]): The kaehler moduli.
                 verbose (int, optional): Controls logging. 1-Debug, 2-Info,  else Warning. Defaults to 2.
                 precision (int, optional): Number of valid digits. Defaults to 10
                 point_file_path (str, optional): Path where points are stored. This is only important if Mathematica
                                                  is also used as a frontend
                 selected_t (ndarray[(nProj)]): The ambient spaces from which the points were sampled.
                                                This is only important if Mathematica is also used as a frontend
         """
        self.toric_data = toric_data
        self.nfold = toric_data['dim_cy']
        self.monomials = [np.array(toric_data['exp_aK'])]  # since we inherit from CICY pointgen, need array of monomials (one for each defining poly)
        self.coefficients = [np.array(toric_data['coeff_aK'])]  # since we inherit from CICY pointgen, need array of coefficients (one for each defining poly)
        self.kmoduli = kmoduli
        self.sections = toric_data['exps_sections']
        self.non_CI_coeffs = toric_data['non_ci_coeffs']
        self.non_CI_exps = toric_data['non_ci_exps']
        self.patch_masks = toric_data['patch_masks']
        self.glsm_charges = toric_data['glsm_charges']
        self.precision = precision
        self.vol_j_norm = toric_data['vol_j_norm']
        self.verbose = verbose
        self.selected_t = selected_t
        self.ambient = 2 * self.selected_t  # hack to make the standard routine for auxiliary weight computation work. Note that for the toric case, the ambient space plays a different role and is not linked to the number of Kahler moduli
        self.ambient_dims = np.array([len(s) + 1 for s in toric_data['exps_sections']])
        self.lc = get_levicivita_tensor(int(self.nfold))
        
        if self.verbose == 1:
            self.level = logging.DEBUG
        elif self.verbose == 2:
            self.level = logging.INFO
        else:
            self.level = logging.WARNING
        logger.setLevel(level=self.level)
               
        self.nmonomials, self.ncoords = self.monomials[0].shape
        self.nhyper = 1
        self.wl_session = None  
        self.point_file_path = point_file_path
        
        # sympy variables
        self.x = sp.var('x0:' + str(self.ncoords))
        self.poly = sum(self.coefficients * np.multiply.reduce(np.power(self.x, self.monomials), axis=-1))
        
        # some more internal variables
        self._set_seed(2021)
        self._generate_dQdz_basis()
        self.dzdz_generated = False
        self._generate_padded_basis()
    
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

    def fubini_study_metrics(self, points, vol_js=None):
        return self._fubini_study_n_metrics(points, kfactors=vol_js)
        
    def generate_points(self, n_p, nproc=-1):
        r"""Generates complex points by calling the mathematica point generator

        Args:
            n_p (int): # of points
            nproc (int, optional): # of processes used. Defaults to -1.

        Returns:
            nd.array[(np, ncoord), np.complex128]: [description]
        """
        with WolframLanguageSession(kernel_loglevel=self.level) as mathematica_session:
            self.wl_session = mathematica_session
            self._setup_session(mathematica_session)
            self._start_parallel_kernels(mathematica_session, nproc)
        
            # Define Python function from Mathematica function
            get_points_mathematica = mathematica_session.function(wlGlobal.GenerateToricPointsM)
        
            # Compute the points
            # specifying the consumer in the session with WolframLanguageSession(consumer=ComplexFunctionConsumer()) seems to not work as of wolframclient version 1.1.6
            # so we need to deserialize the object without a consumer, serialize it to a wxf expression, and deserialize that with a consumer

            # compute and deserialize the points
            logger.debug("Initializing generation of {:} points...".format(n_p))
            pts = get_points_mathematica(n_p, self.nfold, self.coefficients[0].tolist(), self.monomials[0].tolist(), self.sections, self.non_CI_coeffs, self.non_CI_exps, self.patch_masks, self.glsm_charges, self.precision, self.verbose, False)
            # serialize again to wxf
            pts = wlexport(pts, target_format='wxf')
            # deserialize converting Mathamtica complex numbers to python complex numbers
            pts = binary_deserialize(pts, consumer=ComplexFunctionConsumer())
        
        self.selected_t = np.array(pts[1], dtype=np.int)
        return np.array(pts[0])
        
    def _fubini_study_n_metrics(self, points, kfactors=None):
        r"""Computes the FS metric of points.

        Args:
            point (ndarray[(np, n), np.complex]): point
            kfactors (list): volume factor.

        Returns:
            ndarray[(len(points), n, n), np.complex]: g^FS
        """
        kfactors = self.kmoduli if kfactors is None else kfactors
        Js = np.zeros([len(points), len(self.sections[0][0]), len(self.sections[0][0])], dtype=np.complex128)
        for alpha in range(len(kfactors)):
            ms = np.transpose(np.product([np.power(points, self.sections[alpha][a]) for a in range(len(self.sections[alpha]))], axis=-1), [1, 0])
            mss = ms*np.conj(ms)
            kappa_alphas = np.sum(mss, -1)
            J_alphas = 1/(points[:, :, np.newaxis] * np.conj(points[:, np.newaxis, :]))
            J_alphas = np.einsum('x,xab->xab', 1/(kappa_alphas**2), J_alphas)
            coeffs = np.einsum('xa,xb,ai,aj->xij', mss, mss, np.array(self.sections[alpha], dtype=np.complex128), np.array(self.sections[alpha], dtype=np.complex128)) - np.einsum('xa,xb,ai,bj->xij', mss, mss, np.array(self.sections[alpha], dtype=np.complex128), np.array(self.sections[alpha], dtype=np.complex128))
            Js += J_alphas * coeffs * np.complex(kfactors[alpha]) / np.complex(np.pi)
        return Js


# copied from the wolfram client library documentation, accessed June 2021
# https://reference.wolfram.com/language/WolframClientForPython/docpages/advanced_usages.html
class ComplexFunctionConsumer(WXFConsumer):
    """Implement a consumer that maps Complex to python complex types."""
    def build_function(self, head, args, **kwargs):
        # return a built in complex if head is Complex and argument length is 2.
        if head == wl.Complex and len(args) == 2:
            return complex(*args)
        # otherwise delegate to the super method (default case).
        else:
            return super().build_function(head, args, **kwargs)
    
    def consume_bigreal(self, current_token, tokens, **kwargs):
        """Parse a WXF big real as a WXF serializable big real.

        There is not such thing as a big real, in Wolfram Language notation, in Python. This
        wrapper ensures round tripping of big reals without the need of `ToExpression`.
        Introducing `ToExpression` would imply to marshall the big real data to avoid malicious
        code from being introduced in place of an actual real.
        """
        BIGREAL_RE = re.compile(r"([^`]+)(`[0-9.]+){0,1}(\*\^[0-9]+){0,1}")
        match = BIGREAL_RE.match(current_token.data)

        if match:
            num, prec, exp = match.groups()
            if exp:
                return decimal.Decimal("%se%s" % (num, exp[2:]))

            return complex(num)

        raise WolframParserException("Invalid big real value: %s" % current_token.data)
