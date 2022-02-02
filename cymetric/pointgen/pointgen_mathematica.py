"""
Class to call Mathematica point generator. 

Authors
-------
Fabian Ruehle fabian.ruehle@cern.ch
Robin Schneider robin.schneider@physics.uu.se
"""
# pip install wolframclient
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
from cymetric.pointgen.nphelper import get_levicivita_tensor

import logging
logger = logging.getLogger('pointgenMathematica')
logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s')


class PointGeneratorMathematica(CICYPointGenerator):
    def __init__(self, *args, **kwargs):
        
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
        
    def generate_point_weights(self, n_pw, omega=False, normalize_to_vol_j=False):
        data_types = [
            ('point', np.complex128, self.ncoords),
            ('weight', np.float64)
        ]
        data_types = data_types + [('omega', np.complex128)] if omega else data_types
        dtype = np.dtype(data_types)
        if self.point_file_path is None or not os.path.exists(self.point_file_path):
            points = self.generate_points(n_pw)
        else:
            points = np.array(pickle.load(open(self.point_file_path, 'rb')))
        n_p = len(points)
        n_p = n_p if n_p < n_pw else n_pw
        
        weights = self.point_weight(points, normalize_to_vol_j=normalize_to_vol_j)
        point_weights = np.zeros((n_p), dtype=dtype)
        point_weights['point'], point_weights['weight'] = points[0:n_p], weights[0:n_p]
        if omega:
            point_weights['omega'] = self.holomorphic_volume_form(points[0:n_p])
        return point_weights
    
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


class PointGeneratorToricMathematica(PointGeneratorMathematica):
    def __init__(self, nfold, monomials, coefficients, kmoduli, ambient, sections, non_CI_coeffs, non_CI_exps, patch_masks, glsm_charges, precision=10, vol_j_norm=1, verbose=2, point_file_path=None, selected_t=None):
        r"""Initializer.

        Args:
            monomials (ndarray[(nMonomials, ncoord), np.int]): monomials
            coefficients (ndarray[(nMonomials)]): coefficients in front of each
                monomial.
            kmoduli (ndarray[(nProj)]): The kaehler moduli.
            ambient (ndarray[(nProj), np.int]): The direct product of projective
                spaces in the ambient
            verbose (int, optional): Controls logging. Defaults to 2.
        """
        self.nfold = nfold
        self.monomials = [np.array(monomials)]  # since we inherit from CICY pointgen, need array of monomnials (one for each defining poly)
        self.coefficients = [np.array(coefficients)]  # since we inherit from CICY pointgen, need array of monomnials (one for each defining poly)
        self.kmoduli = kmoduli
        self.ambient = [int(a) for a in ambient]
        self.sections = sections
        self.non_CI_coeffs = non_CI_coeffs
        self.non_CI_exps = non_CI_exps
        self.patch_masks = patch_masks
        self.glsm_charges = glsm_charges
        self.precision = precision
        self.vol_j_norm = vol_j_norm
        self.verbose = verbose
        self.selected_t = selected_t
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
