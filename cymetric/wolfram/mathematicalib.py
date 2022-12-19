import numpy as np
import sys
import os
import re
import logging
import pickle
logging.basicConfig(stream=sys.stdout)
mcy_logger = logging.getLogger('mathematica')

from cymetric.pointgen.pointgen_mathematica import PointGeneratorMathematica, ToricPointGeneratorMathematica
from cymetric.pointgen.nphelper import prepare_dataset, prepare_basis_pickle

import tensorflow as tf
import tensorflow.keras as tfk

tf.get_logger().setLevel('ERROR')

from cymetric.models.tfmodels import PhiFSModel, MultFSModel, FreeModel, MatrixFSModel, AddFSModel, PhiFSModelToric, MatrixFSModelToric
from cymetric.models.tfhelper import prepare_tf_basis, train_model
from cymetric.models.callbacks import SigmaCallback, KaehlerCallback, TransitionCallback, RicciCallback, VolkCallback, AlphaCallback
from cymetric.models.metrics import SigmaLoss, KaehlerLoss, TransitionLoss, RicciLoss, VolkLoss

from wolframclient.language import wl
from wolframclient.serializers import export as wlexport
from wolframclient.deserializers import WXFConsumer, binary_deserialize, WXFConsumerNumpy
Complex = np.complex64
class wlConsumer(WXFConsumer):
    def build_function(self, head, args, **kwargs):
        # return a built in complex if head is Complex and argument length is 2.
        if head == wl.Complex and len(args) == 2:
            return np.complex(*args)
        elif head == wl.NumericArray:
            return np.array(*args[0])
        # otherwise delegate to the super method (default case).
        else:
            return super().build_function(head, args, **kwargs)



def point_vec_to_complex(p):
    if len(p) == 0: 
        return np.array([[]])
    p = np.array(p)
    plen = len(p[0])//2
    return p[:, :plen] + 1.j*p[:, plen:]


def to_numpy_arrays(my_args):
    args_dict = {}
    for k, v in my_args.items():
        if isinstance(v, list) or isinstance(v, tuple):
            args_dict[k] = np.array(v) 
        elif type(v) == type(wl.NumericArray([0])):
            args_dict[k] = binary_deserialize(wlexport(v, target_format='wxf'), consumer=wlConsumer())
        else:
            args_dict[k] = v
    
    args_dict['logger_level'] = eval(args_dict['logger_level'])
    return args_dict


def generate_points(my_args):
    global mcy_logger
    args = to_numpy_arrays(my_args)
    mcy_logger.setLevel(args['logger_level'])
    mcy_logger.debug("Using output directory {}".format(os.path.abspath(args['Dir'])))
    
    # print ambient space
    amb_str = ""
    for d in args['ambient_dims']:
        amb_str += "P^{} x ".format(d)
    amb_str = amb_str[:-2]
    mcy_logger.debug("Ambient space: {}".format(amb_str))
    mcy_logger.debug("Kahler moduli: {}".format(args['KahlerModuli']))

    args_str = re.sub('\],\n', '], ', str(args))
    args_str = re.sub(' +', ' ', str(args_str))
    mcy_logger.debug(args_str)
    
    # need to specify monomials and their coefficients
    if args['monomials'] == [] or args['coeffs'] == []:
        raise ValueError("You need to specify both the monomials and their coefficients")

    point_gen = PointGeneratorMathematica([np.array(x) for x in args['monomials']], [np.array(x) for x in args['coeffs']], args['KahlerModuli'], args['ambient_dims'], precision=args['Precision'], point_file_path=args['point_file_path'], selected_t=args['selected_t'])

    # save point generator to pickle
    mcy_logger.info("Saving point generator to {:}".format(os.path.join(os.path.abspath(args['Dir']), "point_gen.pickle")))
    with open(os.path.join(os.path.abspath(args['Dir']), "point_gen.pickle"), 'wb') as hnd:
        pickle.dump(point_gen, hnd)
    
    kappa = prepare_dataset(point_gen, args['num_pts'], args['Dir'], normalize_to_vol_j=True)
    mcy_logger.info("Computing derivatives of J_FS, Omega, ...")
    prepare_basis_pickle(point_gen, args['Dir'], kappa)
    mcy_logger.debug("done")


def generate_points_toric(my_args):
    global mcy_logger
    args = to_numpy_arrays(my_args)
    mcy_logger.setLevel(args['logger_level'])
    mcy_logger.debug("Using output directory {}".format(os.path.abspath(args['Dir'])))
    
    # print ambient space
    args_str = re.sub('\], \n', '], ', str(args))
    args_str = re.sub(' +', ' ', str(args_str))
    mcy_logger.debug(args_str)

    with open(os.path.join(args['Dir'], 'toric_data.pickle'), 'rb') as f:
        toric_data = pickle.load(f)
    for key in toric_data:
        mcy_logger.debug(key)
        mcy_logger.debug(toric_data[key])

    point_gen = ToricPointGeneratorMathematica(toric_data, precision=args['Precision'], verbose=args['Verbose'], point_file_path=args['point_file_path'])

    # save point generator to pickle
    mcy_logger.info("Saving point generator to {:}".format(os.path.join(os.path.abspath(args['Dir']), "point_gen.pickle")))
    with open(os.path.join(os.path.abspath(args['Dir']), "point_gen.pickle"), 'wb') as hnd:
        pickle.dump(point_gen, hnd)
    
    kappa = prepare_dataset(point_gen, args['num_pts'], args['Dir'], normalize_to_vol_j=True)
    mcy_logger.info("Computing derivatives of J_FS, Omega, ...")
    prepare_basis_pickle(point_gen, args['Dir'], kappa)
    mcy_logger.debug("done")
    

def train_NN(my_args):
    global mcy_logger
    
    args = to_numpy_arrays(my_args)
    mcy_logger.setLevel(args['logger_level'])
    mcy_logger.debug(args)
    
    # get info of generated points
    data = np.load(os.path.join(args['Dir'], 'dataset.npz'))
    BASIS = prepare_tf_basis(pickle.load(open(os.path.join(args['Dir'], 'basis.pickle'), 'rb')))
    kappa = BASIS['KAPPA'].numpy()

    # load toric data if exists/needed
    toric_data = None
    if args['Model'] == 'PhiFSToric':
        if os.path.exists(args['toric_data_path']):
            toric_data = pickle.load(open(args['toric_data_path'], 'rb'))
        else:
            mcy_logger.error("Model set to {}, but {} with toric data not found.".format(args['Model'], args['toric_data_path']))

    # force GPU disable if argument is set:
    if args["DisableGPU"]:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # check whether Keras is running on GPU or CPU:
    tf_devices = "GPU"
    if len(tf.config.list_physical_devices('GPU')) == 0:
        tf_devices = "CPU"
    mcy_logger.debug("Using {} for computation.".format(tf_devices))
    
    # extract architecture for NN
    nfold = tf.cast(BASIS['NFOLD'], dtype=tf.float32).numpy()
    n_in = data['X_train'].shape[1]
    n_hiddens, acts = args["HiddenLayers"], args["ActivationFunctions"]
    n_out = nfold**2
    if args['Model'] == 'PhiFS' or args['Model'] == 'PhiFSToric':
        args['PrintLosses'][1] = False  # Kahler loss is automatically 0
        args['PrintMeasures'][1] = False  # Kahler loss is automatically 0
        n_out = 1
    
    # callbacks
    if args['EvaluateModel']:
        scb = SigmaCallback((data['X_val'], data['y_val']))
        kcb = KaehlerCallback((data['X_val'], data['y_val']))
        tcb = TransitionCallback((data['X_val'], data['y_val']))
        rcb = RicciCallback((data['X_val'], data['y_val']), data['val_pullbacks'])
        volkck = VolkCallback((data['X_val'], data['y_val']))
        cb_list = [scb, kcb, tcb, rcb, volkck]
        cb_list = [x for x, y in zip(cb_list, args['PrintMeasures']) if y]
    else:
        cb_list = []
    
    # metrics
    args['PrintLosses'][3] = False  # Ricci loss not computed at the moment
    cmetrics = [SigmaLoss(), KaehlerLoss(), TransitionLoss(), RicciLoss(), VolkLoss()]
    cmetrics = [x for x, y in zip(cmetrics, args['PrintLosses']) if y]
    
    # build model
    if args['Model'] == 'PhiFS' or args['Model'] == 'PhiFSToric':
        model = tf.keras.Sequential()
        model.add(tfk.Input(shape=(n_in,)))
        for n_hidden, act in zip(n_hiddens, acts):
            model.add(tfk.layers.Dense(n_hidden, activation=act))
        model.add(tfk.layers.Dense(n_out, use_bias=False))
#       # reproduces the FS Kahler potential for the bicubic
#       import math
#       def reorder_input(x):
#           x1 = x[:,0:x.shape[-1]//4]
#           x2 = x[:,x.shape[-1]//4:2*x.shape[-1]//4]
#           x3 = x[:,2*x.shape[-1]//4:3*x.shape[-1]//4]
#           x4 = x[:,3*x.shape[-1]//4:]
#           return tf.keras.layers.concatenate([x1,x3], axis=1), tf.keras.layers.concatenate([x2,x4], axis=1)
#       
#       inp1 = tf.keras.layers.Input(shape=(12,))
#       in1, in2 = tf.keras.layers.Lambda(reorder_input)(inp1)
#       x1 = tf.keras.layers.dot([in1, in1], axes=-1)
#       x2 = tf.keras.layers.dot([in2, in2], axes=-1)
#       for n_hidden, act in zip(n_hiddens, acts):
#         x1 = tf.keras.layers.Dense(n_hidden, activation=act)(x1)
#         x2 = tf.keras.layers.Dense(n_hidden, activation=act)(x2)
#       x1 = tfk.layers.Dense(n_out, use_bias=False, activation='sigmoid')(x1)
#       x2 = tfk.layers.Dense(n_out, use_bias=False, activation='sigmoid')(x2)
#       x1 = tf.math.log(x1)
#       x2 = tf.math.log(x2)
#       x = tf.keras.layers.add([0.1/math.pi * x1, 0.1/math.pi * x2])
#       x = tfk.layers.Dense(n_out)(0.0000000001*x)
#       
#       model = tf.keras.models.Model(inputs=[inp1], outputs=x)
    else:
        model = tf.keras.Sequential()
        model.add(tfk.Input(shape=(n_in,)))
        for n_hidden, act in zip(n_hiddens, acts):
            model.add(tfk.layers.Dense(n_hidden, activation=act))
        model.add(tfk.layers.Dense(n_out))
    
    mcy_logger.debug("Using model {}".format(args['Model']))
    if args['Model'] == 'PhiFS':
        fsmodel = PhiFSModel(model, BASIS, alpha=args['Alphas'])
    elif args['Model'] == 'PhiFSToric':
        fsmodel = PhiFSModelToric(model, BASIS, alpha=args['Alphas'], toric_data=toric_data)
    elif args['Model'] == 'MultFS':
        fsmodel = MultFSModel(model, BASIS, alpha=args['Alphas'])
    elif args['Model'] == 'MatrixMultFS':
        fsmodel = MatrixFSModel(model, BASIS, alpha=args['Alphas'])
    elif args['Model'] == 'MatrixMultFSToric':
        fsmodel = MatrixFSModelToric(model, BASIS, alpha=args['Alphas'], toric_data=toric_data)
    elif args['Model'] == 'AddFS':
        fsmodel = AddFSModel(model, BASIS, alpha=args['Alphas'])
    elif args['Model'] == 'Free':
        fsmodel = FreeModel(model, BASIS, alpha=args['Alphas'])
    else:
        mcy_logger.error("{} is not a recognized option for a model".format(args['Model']))
        return {}
    optimizer = tfk.optimizers.Adam(learning_rate=args['LearningRate'])
    model.summary(print_fn=mcy_logger.debug)

    # train model
    fsmodel, training_history = train_model(fsmodel, data, optimizer=optimizer, epochs=args['Epochs'], batch_sizes=args['BatchSizes'], verbose=2, custom_metrics=cmetrics, callbacks=cb_list)
        
    # save trained model
    fsmodel.model.save(os.path.join(args['Dir'], 'model'))
    
    return training_history


def get_g(my_args):
    global mcy_logger
    my_args = dict(my_args)
    pts = my_args['points']
    del my_args['points']
    
    # parse arguments
    args = to_numpy_arrays(my_args)
    mcy_logger.setLevel(args['logger_level'])
    mcy_logger.debug(args)

    # load toric data if exists/needed
    toric_data = None
    if args['Model'] == 'PhiFSToric':
        if os.path.exists(args['toric_data_path']):
            toric_data = pickle.load(open(args['toric_data_path'], 'rb'))
        else:
            mcy_logger.error("Model set to {}, but {} with toric data not found.".format(args['Model'], args['toric_data_path']))
        
    BASIS = prepare_tf_basis(pickle.load(open(os.path.join(args['Dir'], 'basis.pickle'), 'rb')))
    kappa = BASIS['KAPPA'].numpy()
    pts = tf.convert_to_tensor(pts, dtype=tf.float32)
    model = tfk.models.load_model(os.path.join(args['Dir'], 'model'))
    if args['Model'] == 'PhiFS':
        fsmodel = PhiFSModel(model, BASIS)
    elif args['Model'] == 'PhiFSToric':
        fsmodel = PhiFSModelToric(model, BASIS, toric_data=toric_data)
    elif args['Model'] == 'MultFS':
        fsmodel = MultFSModel(model, BASIS)
    elif args['Model'] == 'MatrixMultFS':
        fsmodel = MatrixFSModel(model, BASIS)
    elif args['Model'] == 'MatrixMultFSToric':
        fsmodel = MatrixFSModelToric(model, BASIS, toric_data=toric_data)
    elif args['Model'] == 'AddFS':
        fsmodel = AddFSModel(model, BASIS)
    elif args['Model'] == 'Free':
        fsmodel = FreeModel(model, BASIS)
    else:
        mcy_logger.error("{} is not a recognized option for a model".format(args['Model']))
        return []

    gs = fsmodel(pts)
    return gs.numpy()


def get_g_fs(my_args):
    global mcy_logger
    my_args = dict(my_args)
    pts = np.array(point_vec_to_complex(my_args['points']), dtype=np.complex128)
    del my_args['points']
    
    # parse arguments
    args = to_numpy_arrays(my_args)
    mcy_logger.setLevel(args['logger_level'])
    mcy_logger.debug(args)
            
    with open(os.path.join(os.path.abspath(args['Dir']), "point_gen.pickle"), 'rb') as hnd:
        point_gen = pickle.load(hnd)
    
    pbs = point_gen.pullbacks(pts)
    ts = args['ts'] if args['ts'] != [] else point_gen.kmoduli
    fs = point_gen.fubini_study_metrics(pts, vol_js=ts)
    fs_pbs = np.einsum('xai,xij,xbj->xab', pbs, fs, np.conj(pbs))
    
    return fs_pbs


def get_kahler_potential(my_args):
    global mcy_logger
    my_args = dict(my_args)
    pts = my_args['points']
    del my_args['points']
    
    # parse arguments
    args = to_numpy_arrays(my_args)
    mcy_logger.setLevel(args['logger_level'])
    mcy_logger.debug(args)

    # load toric data if exists/needed
    toric_data = None
    if args['Model'] == 'PhiFSToric':
        if os.path.exists(args['toric_data_path']):
            toric_data = pickle.load(open(args['toric_data_path'], 'rb'))
        else:
            mcy_logger.error("Model set to {}, but {} with toric data not found.".format(args['Model'], args['toric_data_path']))
        
    BASIS = prepare_tf_basis(pickle.load(open(os.path.join(args['Dir'], 'basis.pickle'), 'rb')))
    pts = tf.convert_to_tensor(pts, dtype=tf.float32)
    model = tfk.models.load_model(os.path.join(args['Dir'], 'model'))
    if args['Model'] == 'PhiFS':
        fsmodel = PhiFSModel(model, BASIS)
    elif args['Model'] == 'PhiFSToric':
        fsmodel = PhiFSModelToric(model, BASIS, toric_data=toric_data)
    else:
        mcy_logger.error("Calculating the Kahler potential for model {} is not supported".format(args['Model']))
        return []

    ks = fsmodel.get_kahler_potential(pts)
    return ks.numpy()

    
def get_weights(my_args):
    global mcy_logger
    my_args = dict(my_args)
    pts = point_vec_to_complex(my_args['points'])
    del my_args['points']
    
    # parse arguments
    args = to_numpy_arrays(my_args)
    mcy_logger.setLevel(args['logger_level'])
    mcy_logger.debug(args)
            
    with open(os.path.join(os.path.abspath(args['Dir']), "point_gen.pickle"), 'rb') as hnd:
        point_gen = pickle.load(hnd)
    return point_gen.point_weight(pts, normalize_to_vol_j=True)


def get_omegas(my_args):
    global mcy_logger
    my_args = dict(my_args)
    pts = point_vec_to_complex(my_args['points'])
    del my_args['points']
    
    # parse arguments
    args = to_numpy_arrays(my_args)
    mcy_logger.setLevel(args['logger_level'])
    mcy_logger.debug(args)
            
    with open(os.path.join(os.path.abspath(args['Dir']), "point_gen.pickle"), 'rb') as hnd:
        point_gen = pickle.load(hnd)
    
    omega = point_gen.holomorphic_volume_form(pts)
    return omega * np.conj(omega)

def get_pullbacks(my_args):
    global mcy_logger
    my_args = dict(my_args)
    pts = point_vec_to_complex(my_args['points'])
    del my_args['points']
    
    # parse arguments
    args = to_numpy_arrays(my_args)
    mcy_logger.setLevel(args['logger_level'])
    mcy_logger.debug(args)
            
    with open(os.path.join(os.path.abspath(args['Dir']), "point_gen.pickle"), 'rb') as hnd:
        point_gen = pickle.load(hnd)
    
    return point_gen.pullbacks(pts)
