"""
Pytest for some PointGenerators.
"""
import numpy as np
import os as os
from cymetric.pointgen.pointgen import PointGenerator
from cymetric.pointgen.pointgen_cicy import CICYPointGenerator
from cymetric.pointgen.nphelper import generate_monomials
import itertools as it
#TODO: Test every non private function.

class TestPointGenerator:
    
    monomials = 5*np.eye(5, dtype=np.int)
    coeff = np.ones(5)
    kmoduli = np.ones(1)
    ambient = np.array([4])
    pg = PointGenerator(monomials, coeff, kmoduli, ambient, verbose=1)
    dir_name = 'fermat'
    n_p = 1000
    pg.prepare_dataset(n_p, dir_name)
    pg.prepare_basis(dir_name)

    def test_point_weights(self):
        points = self.pg.generate_points(self.n_p)
        assert np.allclose(np.abs(self.pg.cy_condition(points)), 0, atol=1e-7)
        assert np.sum(np.isclose(points, 1+0.j)) == \
            len(points)*len(self.pg.ambient)
        # check weights and related function
        # check that their default args match
        weights = self.pg.point_weight(points)
        assert weights.dtype == np.float
        pbs = self.pg.pullbacks(points)
        assert pbs.dtype == complex
        assert np.all(np.array(pbs.shape) == \
            np.array([len(pbs), self.pg.nfold, self.pg.ncoords]))
        omegas = self.pg.holomorphic_volume_form(points)
        assert omegas.dtype == complex
        gfs = self.pg.fubini_study_metrics(points)
        assert gfs.dtype == complex
        gFSpb = np.einsum('xai,xij,xbj->xab', pbs,
                          gfs, np.conjugate(pbs))
        det = np.linalg.det(gFSpb)
        assert np.all(np.real(det) > 0)

    def test_change_coeff(self):
        return None
    
    def test_compute_kappa(self):
        kappa = self.pg.compute_kappa()
        assert kappa.dtype == np.float

    def test_dataset(self):
        data = np.load(os.path.join(self.dir_name, 'dataset.npz'))
        assert len(data['X_train'])+len(data['X_val']) == self.n_p
        # check if default args remain the same
        cpoints = data['X_train'][:,0:self.pg.ncoords] + \
            1.j*data['X_train'][:,self.pg.ncoords:]
        weights = self.pg.point_weight(cpoints)
        assert np.allclose(weights, data['y_train'][:,-2])

class TestPointGeneratorCICYFermat(TestPointGenerator):

    monomials = 5*np.eye(5, dtype=np.int)
    coeff = np.ones(5)
    kmoduli = np.ones(1)
    ambient = np.array([4])
    pg = CICYPointGenerator([monomials], [coeff], kmoduli, ambient, verbose=1)
    dir_name = 'fermat_cicy'
    n_p = 1000
    pg.prepare_dataset(n_p, dir_name)
    pg.prepare_basis(dir_name)

    def test_compare(self):
        #Compare CICY with reg point generator
        # only works with TestPointGenerator has been called before.
        data = np.load(os.path.join('fermat', 'dataset.npz'))
        # check if default args remain the same
        cpoints = data['X_train'][:,0:self.pg.ncoords] + \
            1.j*data['X_train'][:,self.pg.ncoords:]
        omega = self.pg.holomorphic_volume_form(cpoints)
        omega2 = np.real(omega*np.conj(omega))
        assert np.allclose(omega2, data['y_train'][:,-1])
        weights = self.pg.point_weight(cpoints)
        assert np.allclose(weights, data['y_train'][:,-2])

class TestPointGeneratorCICY533(TestPointGenerator):

    monomials = np.array(list(generate_monomials(6, 3)))
    monomials_per_hyper = [monomials, monomials]
    coeff = [np.random.randn(len(m)) for m in monomials_per_hyper]
    kmoduli = np.ones(1)
    ambient = np.array([5])
    pg = CICYPointGenerator(monomials_per_hyper, coeff, kmoduli, ambient)
    dir_name = '533_cicy'
    n_p = 1000
    pg.prepare_dataset(n_p, dir_name)
    pg.prepare_basis(dir_name)

class TestPointGeneratorCICY2x2(TestPointGenerator):

    ambient = np.array([2,3])
    hyper = np.array([[1,2],[1,2]])
    kmoduli = np.ones(len(ambient))
    monomials = []
    coeff = []
    for h in hyper:
        tmp_m = []
        for i, p in enumerate(ambient):
            tmp_m += [np.array(list(generate_monomials(p+1, h[i])))]
        mbasis = []
        for c in it.product(*tmp_m, repeat=1):
            mbasis += [np.concatenate(c)]
        monomials += [np.array(mbasis)]
        tmp_coeff = np.random.randn(len(monomials[-1]))
        coeff += [tmp_coeff/np.max(tmp_coeff)]
    pg = CICYPointGenerator(monomials, coeff, kmoduli, ambient)
    dir_name = '2x2_cicy'
    n_p = 1000
    pg.prepare_dataset(n_p, dir_name)
    pg.prepare_basis(dir_name)

if __name__ == '__main__':
    print('Run pytest from cmd.')