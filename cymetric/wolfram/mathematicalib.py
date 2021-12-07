import numpy as np
import sys
import os
import re
import logging
import pickle
logging.basicConfig(stream=sys.stdout)
mcy_logger = logging.getLogger('mathematica')
point_gen = None

import cymetric
from cymetric.pointgen.pointgen import PointGenerator
from cymetric.pointgen.pointgen_mathematica import PointGeneratorMathematica, PointGeneratorToricMathematica
from cymetric.pointgen.nphelper import prepare_dataset, prepare_basis_pickle

import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras import initializers
	
tf.get_logger().setLevel('ERROR')

from cymetric.models.tfmodels import PhiFSModel, MultFSModel, FreeModel, MatrixFSModel, AddFSModel, PhiFSModelToric, MatrixFSModelToric
from cymetric.models.tfhelper import prepare_tf_basis
from cymetric.models.callbacks import RicciCallback, SigmaCallback, KaehlerCallback, AlphaCallback, VolkCallback, TransitionCallback
from cymetric.models.losses import sigma_loss
from cymetric.models.metrics import SigmaLoss, KaehlerLoss, TransitionLoss, VolkLoss
from cymetric.models.measures import ricci_measure, sigma_measure, transition_measure_loss


def to_numpy_arrays(my_args):
	for k, v in my_args.items():
		my_args[k] = np.array(v) if isinstance(v, list) else v
	
	return my_args

def generate_points_toric(my_args):
	global mcy_logger, point_gen
	args = to_numpy_arrays(eval(my_args))
	mcy_logger.setLevel(args['logger_level'])
	
	outdir = os.path.join(args['outdir'])
	mcy_logger.debug("Using output directory {}".format(os.path.abspath(args['outdir'])))
	
	# print ambient space
	args_str = re.sub('\],\n', '], ', str(args))
	args_str = re.sub(' +', ' ', str(args))
	mcy_logger.debug(args_str)
	
	point_gen = PointGeneratorToricMathematica(args['dim_cy'], [np.array(x) for x in args['monomials']], [np.array(x) for x in args['coeffs']], args['k_moduli'], args['ambient_dims'], args['sections'], args['patch_masks'], args['glsm_charges'], vol_j_norm=args['vol_j_norm'], precision=args['precision'], verbose=args['verbose'], point_file_path=args['point_file_path'])

	# save point generator to pickle
	mcy_logger.info("Saving point generator to {:}".format(os.path.join(os.path.abspath(args['outdir']), "point_gen.pickle")))
	with open(os.path.join(os.path.abspath(args['outdir']), "point_gen.pickle"), 'wb') as hnd:
		pickle.dump(point_gen, hnd)
	
	prepare_dataset(point_gen, args['num_pts'], args['outdir'])
	mcy_logger.info("Computing derivatives of J_FS, Omega, ...")
	prepare_basis_pickle(point_gen, args['outdir'])
	mcy_logger.debug("done")


def generate_points(my_args):
	global mcy_logger, point_gen
	args = to_numpy_arrays(eval(my_args))
	mcy_logger.setLevel(args['logger_level'])
	
	outdir = os.path.join(args['outdir'])
	mcy_logger.debug("Using output directory {}".format(os.path.abspath(args['outdir'])))
	
	# print ambient space
	amb_str = ""
	for d in args['ambient_dims']: amb_str += "P^{} x ".format(d)
	amb_str = amb_str[:-2]
	mcy_logger.debug("Ambient space: {}".format(amb_str))
	mcy_logger.debug("Kahler moduli: {}".format(args['k_moduli']))

	args_str = re.sub('\],\n', '], ', str(args))
	args_str = re.sub(' +', ' ', str(args))
	mcy_logger.debug(args_str)
	
	# need to specify monomials and their coefficients
	if args['monomials'] == [] or args['coeffs'] == []:
		raise ValueError("You need to specify both the monomials and their coefficients")

# 	if len(args['ambient_dims']) > 1:
# 		PG = PointGeneratorMathematica
# 		point_gen = PG([np.array(x) for x in args['monomials']], [np.array(x) for x in args['coeffs']], args['k_moduli'], args['ambient_dims'], precision=args['precision'], point_file_path=args['point_file_path'])
# 		args['monomials'], args['coeffs'] = [np.array(x) for x in args['monomials']], [np.array(x) for x in args['coeffs']]
# 	else:
# 		PG = PointGenerator
# 		point_gen = PG([np.array(x) for x in args['monomials']], [np.array(x) for x in args['coeffs']], args['k_moduli'], args['ambient_dims'])

	point_gen = PointGeneratorMathematica([np.array(x) for x in args['monomials']], [np.array(x) for x in args['coeffs']], args['k_moduli'], args['ambient_dims'], precision=args['precision'], vol_j_norm=args['vol_j_norm'], point_file_path=args['point_file_path'], selected_t=args['selected_t'])

	# save point generator to pickle
	mcy_logger.info("Saving point generator to {:}".format(os.path.join(os.path.abspath(args['outdir']), "point_gen.pickle")))
	with open(os.path.join(os.path.abspath(args['outdir']), "point_gen.pickle"), 'wb') as hnd:
		pickle.dump(point_gen, hnd)
	
	prepare_dataset(point_gen, args['num_pts'], args['outdir'], normalize_to_vol_j=True)
	mcy_logger.info("Computing derivatives of J_FS, Omega, ...")
	prepare_basis_pickle(point_gen, args['outdir'])
	mcy_logger.debug("done")


def train_NN(my_args):
	global mcy_logger
	
	args = to_numpy_arrays(eval(my_args))
	mcy_logger.setLevel(args['logger_level'])

	# get info of generated points
	data = np.load(os.path.join(args['outdir'], 'dataset.npz'))
	BASIS = prepare_tf_basis(pickle.load(open(os.path.join(args['outdir'], 'basis.pickle'), 'rb')))
	# print([key for key in data])
	# print([key for key in BASIS])
	# print(BASIS['DQDZB0'].shape, BASIS['DQDZF0'].shape, BASIS['DQDZB1'].shape, BASIS['DQDZF1'].shape)
	
	# load toric data if exists/needed
	if os.path.exists(args['toric_data_path']):
		toric_data = pickle.load(open(args['toric_data_path'], 'rb'))
	
	# if kappa is not provided, compute it
	if args['kappa'] == 0.:
		args['kappa'] = 1./np.mean(data['y_train'][:,-2])
	
	# check whether Keras is running on GPU or CPU:
	tf_devices = "GPU"
	if len(tf.config.list_physical_devices('GPU')) == 0:
		tf_devices = "CPU"
	mcy_logger.debug("Using {} for computation.".format(tf_devices))
	
	# extract architecture for NN
	amb	  = tf.cast(BASIS['AMBIENT'], dtype=tf.float32).numpy()
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
		rcb = RicciCallback((data['X_val'], data['y_val']))
		volkck = VolkCallback((data['X_val'], data['y_val']))
		cb_list = [scb, kcb, tcb, rcb, volkck]
	else:
		cb_list = []
	
	# metrics
	cmetrics = [SigmaLoss(), KaehlerLoss(), TransitionLoss(), VolkLoss()]
	
	# build model
	model = tf.keras.Sequential()
	model.add(tfk.Input(shape=(n_in,)))
	for n_hidden, act in zip(n_hiddens, acts):
		model.add(tfk.layers.Dense(n_hidden, activation=act))
	model.add(tfk.layers.Dense(n_out))
	
	mcy_logger.debug("Using model ", args['model'])
	if args['model'] == 'PhiFS':
		fsmodel = PhiFSModel(model, BASIS, alpha=args['alphas'], kappa=args['kappa'])
	elif args['model'] == 'PhiFSToric':
		fsmodel = PhiFSModelToric(model, BASIS, alpha=args['alphas'], kappa=args['kappa'], toric_data=toric_data)
	elif args['model'] == 'MultFS':
		fsmodel = MultFSModel(model, BASIS, alpha=args['alphas'], kappa=args['kappa'])
	elif args['model'] == 'MatrixMultFS':
		fsmodel = MatrixFSModel(model, BASIS, alpha=args['alphas'], kappa=args['kappa'])
	elif args['model'] == 'MatrixMultFSToric':
		fsmodel = MatrixFSModelToric(model, BASIS, alpha=args['alphas'], kappa=args['kappa'], toric_data=toric_data)
	elif args['model'] == 'AddFS':
		fsmodel = AddFSModel(model, BASIS, alpha=args['alphas'], kappa=args['kappa'])
	elif args['model'] == 'Free':
		fsmodel = FreeModel(model, BASIS, alpha=args['alphas'], kappa=args['kappa'])
	fsmodel.compile(custom_metrics=cmetrics, optimizer=tfk.optimizers.Adam(), loss=None)
	
	model.summary(print_fn=mcy_logger.debug)
	
	# train model
	history = fsmodel.fit(data['X_train'], data['y_train'], epochs=args['n_epochs'], batch_size=args['batch_size'], verbose=2, callbacks=cb_list)
		
	# save trained model
	fsmodel.model.save(os.path.join(args['outdir'], 'model'))
	
	return history.history


def get_g(my_args):
	def point_vec_to_complex(p):
		plen = len(p)//2
		return p[:plen] + 1.j*p[plen:]
	
	# don't process points to save time
	my_args = eval(my_args)
	pts = my_args['points']
	del my_args['points']
	
	# parse arguments
	args = to_numpy_arrays(my_args)
	
	# if kappa is not provided, compute it
	if args['kappa'] == 0.:
		data = np.load(os.path.join(args['outdir'], 'dataset.npz'))
		args['kappa'] = 1./np.mean(data['y_train'][:,-2])
		
	BASIS = prepare_tf_basis(pickle.load(open(os.path.join(args['outdir'], 'basis.pickle'), 'rb')))
	pts = tf.convert_to_tensor(pts, dtype=tf.float32)
	model = tfk.models.load_model(os.path.join(args['outdir'], 'model'))
	if args['model'] == 'PhiFS':
		fsmodel = PhiFSModel(model, BASIS, kappa=args['kappa'])
	elif args['model'] == 'MultFS':
		fsmodel = MultFSModel(model, BASIS, kappa=args['kappa'])
	elif args['model'] == 'MatrixMultFS':
		fsmodel = MatrixFSModel(model, BASIS, kappa=args['kappa'])
	elif args['model'] == 'AddFS':
		fsmodel = AddFSModel(model, BASIS, kappa=args['kappa'])
	elif args['model'] == 'Free':
		fsmodel = FreeModel(model, BASIS, kappa=args['kappa'])

	gs = fsmodel(pts)
	
	return [[point_vec_to_complex(x), g] for x, g in zip(pts.numpy(), gs.numpy())]
	
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
# 	return point_gen.point_weight_vec(pts, normalize_to_vol_j=True)
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
