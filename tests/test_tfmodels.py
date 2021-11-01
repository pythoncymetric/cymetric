"""
Pytest for some tensorflow models. 
Requires that `test_pointgen.py` has been run before.
"""
import pytest
import numpy as np
import os as os
#import pickle as pickle
import itertools as it
import tensorflow as tf
tfk = tf.keras

#TODO: Import all metrics and Measures and callbacks and ... then run them.
from cymetric.models.tfhelper import prepare_tf_basis
from cymetric.models.tfmodels import PhiFSModel, FreeModel
from cymetric.models.callbacks import RicciCallback, SigmaCallback, VolkCallback
from cymetric.models.metrics import SigmaLoss, KaehlerLoss, TransitionLoss, \
    VolkLoss, RicciLoss, TotalLoss

class TFModel:

    def __init__(self, work_dir, tfmodel):
        self.cmetrics = [
            TotalLoss(), SigmaLoss(), KaehlerLoss(),
            TransitionLoss(), VolkLoss(), RicciLoss()]
        self.epochs = 1
        self.bSize = 64
        self.alpha = np.ones(5)
        self.norms = np.ones(5)
        self.act = 'gelu'
        self.units = 64
        self.work_dir = work_dir
        self.tfmodel = tfmodel

    def run_tf_model(self):
        basis = self.get_basis(self.work_dir)
        data = np.load(os.path.join(self.work_dir, 'dataset.npz'))
        kappa = 1/np.mean(data['y_train'][:,-2])
        nfold = int(basis['NFOLD'].numpy().real)
        nvars = len(data['X_train'][0])
        n_out = 1 if self.tfmodel is PhiFSModel else nfold**2
        nn = self.get_nn(nvars, n_out)
        cb_list = self.get_cbs(data)
        # TODO: add all possible arguments.
        model = self.tfmodel(nn, basis, alpha=self.alpha, kappa=kappa,
                          norm=self.norms)
        model.compile(custom_metrics=self.cmetrics,
                      optimizer=tfk.optimizers.Adam())

        #Does tracing and training work?
        history = model.fit(
            data['X_train'], data['y_train'], epochs=self.epochs,
            validation_data=(data['X_val'], data['y_val'], data['y_val'][:,-2]),
            batch_size=self.bSize, verbose=1, callbacks=cb_list,
            sample_weight=data['y_train'][:,-2])
            
        #TODO add more asserts
        assert len(list(history.history.keys())) == \
            len(cb_list)+2*len(self.cmetrics)
        return history.history

    def get_nn(self, n_in, n_out):
        model = tfk.Sequential()
        model.add(tfk.Input(shape=(int(n_in))))
        model.add(
            tfk.layers.Dense(
                self.units, 
                activation=self.act,
            )
        )
        model.add(tfk.layers.Dense(n_out))
        return model

    def get_basis(self, work_dir):
        fname = os.path.join(work_dir, 'basis.pickle')
        basis = np.load(fname, allow_pickle=True)
        return prepare_tf_basis(basis)

    def get_cbs(self, data):
        rcb = RicciCallback((data['X_val'], data['y_val']),
                            data['val_pullbacks'])
        scb = SigmaCallback((data['X_val'], data['y_val']))
        volkcb = VolkCallback((data['X_val'], data['y_val']))
        return [rcb, scb, volkcb]

@pytest.mark.parametrize("test_model, test_dir", 
    list(
        it.product(
            [FreeModel, PhiFSModel],
            ['fermat', 'fermat_cicy', '533_cicy', '2x2_cicy'],
            repeat=1)
        )
    )
def test_tf_models(test_model, test_dir):
    #TODO use fixtures instead.
    tfmodel = TFModel(test_dir, test_model)
    history = tfmodel.run_tf_model()
    # Add some usefull checks
    assert history is not None


if __name__ == '__main__':
    print('Run pytest from cmd.'
        'Requires that pointgen test have been run before')