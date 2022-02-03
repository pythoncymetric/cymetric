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
from cymetric.models.tfhelper import prepare_tf_basis
from cymetric.models.callbacks import RicciCallback, SigmaCallback, KaehlerCallback, AlphaCallback, VolkCallback, TransitionCallback
from cymetric.models.metrics import SigmaLoss, KaehlerLoss, TransitionLoss, VolkLoss


def to_numpy_arrays(my_args):
    for k, v in my_args.items():
        my_args[k] = np.array(v) if isinstance(v, list) else v
    
    return my_args


def generate_points_toric(my_args):
    global mcy_logger
    args = to_numpy_arrays(eval(my_args))
    mcy_logger.setLevel(args['logger_level'])
    mcy_logger.debug("Using output directory {}".format(os.path.abspath(args['outdir'])))
    
    # print ambient space
    args_str = re.sub('\], \n', '], ', str(args))
    args_str = re.sub(' +', ' ', str(args_str))
    mcy_logger.debug(args_str)

    with open(os.path.join(args['outdir'], 'toric_data.pickle'), 'rb') as f:
        toric_data = pickle.load(f)
    for key in toric_data:
        mcy_logger.debug(key)
        mcy_logger.debug(toric_data[key])

    point_gen = ToricPointGeneratorMathematica(toric_data, precision=args['precision'], verbose=args['verbose'], point_file_path=args['point_file_path'])

    # save point generator to pickle
    mcy_logger.info("Saving point generator to {:}".format(os.path.join(os.path.abspath(args['outdir']), "point_gen.pickle")))
    with open(os.path.join(os.path.abspath(args['outdir']), "point_gen.pickle"), 'wb') as hnd:
        pickle.dump(point_gen, hnd)
    
    kappa = prepare_dataset(point_gen, args['num_pts'], args['outdir'], normalize_to_vol_j=True)
    mcy_logger.info("Computing derivatives of J_FS, Omega, ...")
    prepare_basis_pickle(point_gen, args['outdir'], kappa)
    mcy_logger.debug("done")


def generate_points(my_args):
    global mcy_logger
    args = to_numpy_arrays(eval(my_args))
    mcy_logger.setLevel(args['logger_level'])
    mcy_logger.debug("Using output directory {}".format(os.path.abspath(args['outdir'])))
    
    # print ambient space
    amb_str = ""
    for d in args['ambient_dims']:
        amb_str += "P^{} x ".format(d)
    amb_str = amb_str[:-2]
    mcy_logger.debug("Ambient space: {}".format(amb_str))
    mcy_logger.debug("Kahler moduli: {}".format(args['k_moduli']))

    args_str = re.sub('\],\n', '], ', str(args))
    args_str = re.sub(' +', ' ', str(args_str))
    mcy_logger.debug(args_str)
    
    # need to specify monomials and their coefficients
    if args['monomials'] == [] or args['coeffs'] == []:
        raise ValueError("You need to specify both the monomials and their coefficients")

    point_gen = PointGeneratorMathematica([np.array(x) for x in args['monomials']], [np.array(x) for x in args['coeffs']], args['k_moduli'], args['ambient_dims'], precision=args['precision'], vol_j_norm=args['vol_j_norm'], point_file_path=args['point_file_path'], selected_t=args['selected_t'])

    # save point generator to pickle
    mcy_logger.info("Saving point generator to {:}".format(os.path.join(os.path.abspath(args['outdir']), "point_gen.pickle")))
    with open(os.path.join(os.path.abspath(args['outdir']), "point_gen.pickle"), 'wb') as hnd:
        pickle.dump(point_gen, hnd)
    
    kappa = prepare_dataset(point_gen, args['num_pts'], args['outdir'], normalize_to_vol_j=True)
    mcy_logger.info("Computing derivatives of J_FS, Omega, ...")
    prepare_basis_pickle(point_gen, args['outdir'], kappa)
    mcy_logger.debug("done")
    
    
#     def point_vec_to_complex(p):
#         plen = len(p)//2
#         return p[:plen] + 1.j*p[plen:]
#     
#     data = np.load(os.path.join(args['outdir'], 'dataset.npz'))
#     pts = np.array([point_vec_to_complex(p) for p in data['X_val']])
#     weights, omegas = data['y_val'][:,-2], data['y_val'][:,-1]
#     pbs = point_gen.pullbacks(pts)
#     gFS = point_gen.fubini_study_metrics(pts)
#     gFS_pbs = np.einsum('xai,xij,xbj->xab', pbs, gFS, np.conj(pbs))
#     dets = np.real(np.linalg.det(gFS_pbs))
# 
#     vol_k = np.mean(weights * dets / omegas)
#     vol_cy = np.mean(weights)
#     print('Without NN')
#     print('Vol_k: {}, Vol_cy: {}.'.format(vol_k, vol_cy))



def train_NN(my_args):
    global mcy_logger
    
    args = to_numpy_arrays(eval(my_args))
    mcy_logger.setLevel(args['logger_level'])
    mcy_logger.setLevel(logging.DEBUG)
    mcy_logger.debug(args)
    
    
    # get info of generated points
    data = np.load(os.path.join(args['outdir'], 'dataset.npz'))
    BASIS = prepare_tf_basis(pickle.load(open(os.path.join(args['outdir'], 'basis.pickle'), 'rb')))
    kappa = BASIS['KAPPA'].numpy()

    # load toric data if exists/needed
    toric_data = None
    if args['model'] == 'PhiFSToric':
        if os.path.exists(args['toric_data_path']):
            toric_data = pickle.load(open(args['toric_data_path'], 'rb'))
        else:
            mcy_logger.error("Model set to {}, but {} with toric data not found.".format(args['model'], args['toric_data_path']))

    # check whether Keras is running on GPU or CPU:
    tf_devices = "GPU"
    if len(tf.config.list_physical_devices('GPU')) == 0:
        tf_devices = "CPU"
    mcy_logger.debug("Using {} for computation.".format(tf_devices))
    
    # extract architecture for NN
    nfold = tf.cast(BASIS['NFOLD'], dtype=tf.float32).numpy()
    n_in = data['X_train'].shape[1]
    n_hiddens, acts = args["n_hiddens"], args["acts"]
    n_out = nfold**2
    if args['model'] == 'PhiFS' or args['model'] == 'PhiFSToric':
        n_out = 1
    
    # callbacks
    if args['callbacks']:
        scb = SigmaCallback((data['X_val'], data['y_val']))
        kcb = KaehlerCallback((data['X_val'], data['y_val']))
        tcb = TransitionCallback((data['X_val'], data['y_val']))
        rcb = RicciCallback((data['X_val'], data['y_val']), data['val_pullbacks'])
        volkck = VolkCallback((data['X_val'], data['y_val']))
        cb_list = [scb, kcb, tcb, rcb, volkck]
    else:
        cb_list = []
    
    # metrics
    cmetrics = [SigmaLoss(), KaehlerLoss(), TransitionLoss(), VolkLoss()]
    
    # build model
    if args['model'] == 'PhiFS' or args['model'] == 'PhiFSToric':
        model = tf.keras.Sequential()
        model.add(tfk.Input(shape=(n_in,)))
        for n_hidden, act in zip(n_hiddens, acts):
            model.add(tfk.layers.Dense(n_hidden, activation=act))
        model.add(tfk.layers.Dense(n_out))
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
    
    mcy_logger.debug("Using model ", args['model'])
    if args['model'] == 'PhiFS':
        fsmodel = PhiFSModel(model, BASIS, alpha=args['alphas'])
    elif args['model'] == 'PhiFSToric':
        fsmodel = PhiFSModelToric(model, BASIS, alpha=args['alphas'], toric_data=toric_data)
    elif args['model'] == 'MultFS':
        fsmodel = MultFSModel(model, BASIS, alpha=args['alphas'])
    elif args['model'] == 'MatrixMultFS':
        fsmodel = MatrixFSModel(model, BASIS, alpha=args['alphas'])
    elif args['model'] == 'MatrixMultFSToric':
        fsmodel = MatrixFSModelToric(model, BASIS, alpha=args['alphas'], toric_data=toric_data)
    elif args['model'] == 'AddFS':
        fsmodel = AddFSModel(model, BASIS, alpha=args['alphas'])
    elif args['model'] == 'Free':
        fsmodel = FreeModel(model, BASIS, alpha=args['alphas'])
    else:
        mcy_logger.error("{} is not a recognized option for a model".format(args['model']))
        return {}
    fsmodel.compile(custom_metrics=cmetrics, optimizer=tfk.optimizers.Adam(), loss=None)
    
    model.summary(print_fn=mcy_logger.debug)
    
#     weights = tf.cast(data['y_train'][:, -2], tf.float32)
#     pred = tf.cast(fsmodel.model(data['X_train']), tf.float32)
#     
#     phi_pred = tf.reshape(pred, [-1])  # flatten
#     phi_pred = phi_pred * weights / tf.math.reduce_mean(weights, axis=-1)
#     phi_pred = tf.math.reduce_mean(phi_pred, axis=-1)
#     phi_pred = tf.math.abs(phi_pred)
#     phi_pred = tf.repeat(tf.expand_dims(phi_pred, axis=0), repeats=[len(data['X_train'])], axis=0)
#         
#     print("phi_pred      : ", (phi_pred)[:10], tf.reduce_mean(phi_pred, axis=-1).numpy())
#     print("volk loss:      ", (fsmodel.compute_volk_loss(data['X_train'], weights, pred))[:10])
    # train model
    history = fsmodel.fit(data['X_train'], data['y_train'], epochs=args['n_epochs'], batch_size=args['batch_size'], verbose=2, callbacks=cb_list)
        
    # save trained model
    fsmodel.model.save(os.path.join(args['outdir'], 'model'))
    
    return history.history


def get_g(my_args):
    global mcy_logger
    mcy_logger.setLevel(logging.DEBUG)
    # don't process points to save time
    my_args = eval(my_args)
    pts = my_args['points']
    del my_args['points']
    
    # parse arguments
    args = to_numpy_arrays(my_args)

    # load toric data if exists/needed
    toric_data = None
    if args['model'] == 'PhiFSToric':
        if os.path.exists(args['toric_data_path']):
            toric_data = pickle.load(open(args['toric_data_path'], 'rb'))
        else:
            mcy_logger.error("Model set to {}, but {} with toric data not found.".format(args['model'], args['toric_data_path']))
        
    BASIS = prepare_tf_basis(pickle.load(open(os.path.join(args['outdir'], 'basis.pickle'), 'rb')))
    kappa = BASIS['KAPPA'].numpy()
    pts = tf.convert_to_tensor(pts, dtype=tf.float32)
    model = tfk.models.load_model(os.path.join(args['outdir'], 'model'))
    if args['model'] == 'PhiFS':
        fsmodel = PhiFSModel(model, BASIS)
    elif args['model'] == 'PhiFSToric':
        fsmodel = PhiFSModelToric(model, BASIS, toric_data=toric_data)
    elif args['model'] == 'MultFS':
        fsmodel = MultFSModel(model, BASIS)
    elif args['model'] == 'MatrixMultFS':
        fsmodel = MatrixFSModel(model, BASIS)
    elif args['model'] == 'MatrixMultFSToric':
        fsmodel = MatrixFSModelToric(model, BASIS, toric_data=toric_data)
    elif args['model'] == 'AddFS':
        fsmodel = AddFSModel(model, BASIS)
    elif args['model'] == 'Free':
        fsmodel = FreeModel(model, BASIS)
    else:
        mcy_logger.error("{} is not a recognized option for a model".format(args['model']))
        return []

    gs = fsmodel(pts)
    return gs.numpy()


def get_g_fs(my_args):
    def point_vec_to_complex(p):
        plen = len(p)//2
        return p[:plen] + 1.j*p[plen:]
    global mcy_logger
    mcy_logger.setLevel(logging.DEBUG)
    # don't process points to save time
    my_args = eval(my_args)
    pts = np.array([point_vec_to_complex(p) for p in np.array(my_args['points'])])
    del my_args['points']
    
    # parse arguments
    args = to_numpy_arrays(my_args)
            
    with open(os.path.join(os.path.abspath(args['outdir']), "point_gen.pickle"), 'rb') as hnd:
        point_gen = pickle.load(hnd)
    
    pbs = point_gen.pullbacks(pts)
    ts = args['ts'] if args['ts'] != [] else point_gen.kmoduli
    fs = point_gen.fubini_study_metrics(pts, vol_js=ts)
    fs_pbs = np.einsum('xai,xij,xbj->xab', pbs, fs, np.conj(pbs))
    return fs_pbs

    
def get_weights(my_args):
    def point_vec_to_complex(p):
        plen = len(p)//2
        return p[:plen] + 1.j*p[plen:]
    global mcy_logger
    mcy_logger.setLevel(logging.DEBUG)
    # don't process points to save time
    my_args = eval(my_args)
    pts = np.array([point_vec_to_complex(p) for p in np.array(my_args['points'])])
    del my_args['points']
    
    # parse arguments
    args = to_numpy_arrays(my_args)
            
    with open(os.path.join(os.path.abspath(args['outdir']), "point_gen.pickle"), 'rb') as hnd:
        point_gen = pickle.load(hnd)
    return point_gen.point_weight(pts, normalize_to_vol_j=True)


def get_omegas(my_args):
    def point_vec_to_complex(p):
        plen = len(p)//2
        return p[:plen] + 1.j*p[plen:]
    global mcy_logger
    mcy_logger.setLevel(logging.DEBUG)
    # don't process points to save time
    my_args = eval(my_args)
    pts = np.array([point_vec_to_complex(p) for p in np.array(my_args['points'])])
    del my_args['points']
    
    # parse arguments
    args = to_numpy_arrays(my_args)
            
    with open(os.path.join(os.path.abspath(args['outdir']), "point_gen.pickle"), 'rb') as hnd:
        point_gen = pickle.load(hnd)
    
    return point_gen.Omega_vec(pts)


def get_pullbacks(my_args):
    def point_vec_to_complex(p):
        plen = len(p)//2
        return p[:plen] + 1.j*p[plen:]
    global mcy_logger
    mcy_logger.setLevel(logging.DEBUG)
    # don't process points to save time
    my_args = eval(my_args)
    pts = [point_vec_to_complex(p) for p in np.array(my_args['points'])]
    del my_args['points']
    
    # parse arguments
    args = to_numpy_arrays(my_args)
            
    with open(os.path.join(os.path.abspath(args['outdir']), "point_gen.pickle"), 'rb') as hnd:
        point_gen = pickle.load(hnd)
    
    return point_gen.pullbacks(pts)
