import numpy as np

from cymetric.pointgen.pointgen_cicy import CICYPointGenerator


def cicy_init():
    monomials = np.array(list(generate_monomials(6, 3)))
    monomials_per_hyper = [monomials, monomials]
    coeff = [np.random.randn(len(m)) for m in monomials_per_hyper]
    kmoduli = np.ones(1)
    ambient = np.array([5])
    pg = CICYPointGenerator(monomials_per_hyper, coeff, kmoduli, ambient)
    return pg
