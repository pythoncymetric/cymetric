"""
Mathematica interface for Cymetric - Framework Agnostic Version

This module provides a unified interface to Mathematica functionality
that works with PyTorch, TensorFlow, and JAX backends, automatically
selecting the available framework.

:Authors:
    Fabian Ruehle f.ruehle@northeastern.edu
"""
import numpy as np
import sys
import os
import re
import logging
import pickle
logging.basicConfig(stream=sys.stdout)
mcy_logger = logging.getLogger('mathematica')

# Import shared components
from ..pointgen.pointgen_mathematica import PointGeneratorMathematica, ToricPointGeneratorMathematica
from ..pointgen.pointgen_mc import CICYPointGeneratorMC, ToricCICYPointGeneratorMC
from ..pointgen.nphelper import prepare_dataset, prepare_basis_pickle

# Use the unified framework system
try:
    from cymetric import get_preferred_framework, TORCH_AVAILABLE, TENSORFLOW_AVAILABLE, JAX_AVAILABLE
    # Use the unified imports that automatically select the framework
    from cymetric.models.models import PhiFSModel, MultFSModel, FreeModel, MatrixFSModel, AddFSModel, PhiFSModelToric, MatrixFSModelToric
    from cymetric.models.helper import prepare_basis, train_model
    from cymetric.models.callbacks import SigmaCallback, KaehlerCallback, TransitionCallback, RicciCallback, VolkCallback, AlphaCallback
    from cymetric.models.metrics import SigmaLoss, KaehlerLoss, TransitionLoss, RicciLoss, VolkLoss
    
    # Get the current framework
    current_framework = get_preferred_framework()
    mcy_logger.info(f"Mathematica interface using {current_framework} backend")
    
except ImportError as e:
    mcy_logger.error(f"Could not import cymetric framework system: {e}")
    # Fallback to direct imports (legacy behavior)
    current_framework = None
    TORCH_AVAILABLE = False
    TENSORFLOW_AVAILABLE = False
    JAX_AVAILABLE = False

# Framework-specific utilities
def get_framework_modules():
    """Get framework-specific modules based on current framework."""
    framework = get_preferred_framework()
    
    if framework == 'tensorflow':
        try:
            import tensorflow as tf
            import tensorflow.keras as tfk
            tf.get_logger().setLevel('ERROR')
            return {
                'framework': 'tensorflow',
                'tf': tf,
                'tfk': tfk,
                'tensor_fn': tf.convert_to_tensor,
                'cast_fn': tf.cast,
                'float32': tf.float32,
                'device_check_fn': lambda: len(tf.config.list_physical_devices('GPU')) > 0,
                'model_ext': '.keras'
            }
        except ImportError:
            pass
    
    elif framework == 'torch':
        try:
            import torch
            return {
                'framework': 'torch', 
                'torch': torch,
                'tensor_fn': torch.tensor,
                'cast_fn': lambda x, dtype: x.to(dtype) if hasattr(x, 'to') else torch.tensor(x, dtype=dtype),
                'float32': torch.float32,
                'device_check_fn': lambda: torch.cuda.is_available(),
                'model_ext': '.pth'
            }
        except ImportError:
            pass
    
    elif framework == 'jax':
        try:
            import jax
            import jax.numpy as jnp
            import equinox as eqx
            import optax
            return {
                'framework': 'jax',
                'jax': jax,
                'jnp': jnp,
                'eqx': eqx,
                'optax': optax,
                'tensor_fn': jnp.array,
                'cast_fn': lambda x, dtype: jnp.array(x, dtype=dtype),
                'float32': jnp.float32,
                'device_check_fn': lambda: any(
                    'gpu' in d.device_kind.lower() for d in jax.devices()
                ),
                'model_ext': '.eqx'
            }
        except ImportError:
            pass
    
    raise ImportError(f"No framework ({framework}) available")

def build_model_architecture(n_in, n_hiddens, acts, n_out, framework_modules):
    """Build model architecture in framework-agnostic way."""
    framework = framework_modules['framework']
    
    if framework == 'tensorflow':
        tf = framework_modules['tf']
        tfk = framework_modules['tfk']
        
        model = tf.keras.Sequential()
        model.add(tfk.Input(shape=(n_in,)))
        for n_hidden, act in zip(n_hiddens, acts):
            model.add(tfk.layers.Dense(n_hidden, activation=act))
        model.add(tfk.layers.Dense(n_out, use_bias=False))
        return model
        
    elif framework == 'torch':
        torch = framework_modules['torch']
        
        layers = []
        prev_size = n_in
        
        for n_hidden, act in zip(n_hiddens, acts):
            layers.append(torch.nn.Linear(prev_size, n_hidden))
            # Map activation names to PyTorch activations
            if act == 'relu':
                layers.append(torch.nn.ReLU())
            elif act == 'tanh':
                layers.append(torch.nn.Tanh())
            elif act == 'sigmoid':
                layers.append(torch.nn.Sigmoid())
            elif act == 'swish' or act == 'silu':
                layers.append(torch.nn.SiLU())
            # Add more activations as needed
            prev_size = n_hidden
            
        # Final layer without bias (like TensorFlow version)
        layers.append(torch.nn.Linear(prev_size, n_out, bias=False))
        
        return torch.nn.Sequential(*layers)
    
    elif framework == 'jax':
        import jax
        import jax.nn as jnn
        eqx = framework_modules['eqx']
        
        layers = []
        prev_size = n_in
        key = jax.random.PRNGKey(0)
        
        for n_hidden, act in zip(n_hiddens, acts):
            key, subkey = jax.random.split(key)
            layers.append(eqx.nn.Linear(prev_size, n_hidden, key=subkey))
            if act == 'relu':
                layers.append(eqx.nn.Lambda(jnn.relu))
            elif act == 'tanh':
                layers.append(eqx.nn.Lambda(jnn.tanh))
            elif act == 'sigmoid':
                layers.append(eqx.nn.Lambda(jnn.sigmoid))
            elif act in ('swish', 'silu'):
                layers.append(eqx.nn.Lambda(jnn.swish))
            elif act == 'gelu':
                layers.append(eqx.nn.Lambda(jnn.gelu))
            prev_size = n_hidden
        
        key, subkey = jax.random.split(key)
        layers.append(eqx.nn.Linear(prev_size, n_out, use_bias=False, key=subkey))
        return eqx.nn.Sequential(layers)
    
    else:
        raise ValueError(f"Unknown framework: {framework}")

def create_optimizer(model, learning_rate, framework_modules):
    """Create optimizer in framework-agnostic way."""
    framework = framework_modules['framework']
    
    if framework == 'tensorflow':
        tfk = framework_modules['tfk']
        return tfk.optimizers.Adam(learning_rate=learning_rate)
        
    elif framework == 'torch':
        torch = framework_modules['torch']
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    elif framework == 'jax':
        optax = framework_modules['optax']
        return optax.adam(learning_rate)
    
    else:
        raise ValueError(f"Unknown framework: {framework}")

def save_model(model, model_path, framework_modules):
    """Save model in framework-agnostic way."""
    framework = framework_modules['framework']
    
    if framework == 'tensorflow':
        model.save(model_path)
        
    elif framework == 'torch':
        torch = framework_modules['torch']
        torch.save(model.state_dict(), model_path)
    
    elif framework == 'jax':
        eqx = framework_modules['eqx']
        eqx.tree_serialise_leaves(model_path, model)
    
    else:
        raise ValueError(f"Unknown framework: {framework}")

def load_model(model_path, model_architecture, framework_modules):
    """Load model in framework-agnostic way."""
    framework = framework_modules['framework']
    
    if framework == 'tensorflow':
        tfk = framework_modules['tfk']
        return tfk.models.load_model(model_path)
        
    elif framework == 'torch':
        torch = framework_modules['torch']
        model_architecture.load_state_dict(torch.load(model_path, map_location='cpu'))
        return model_architecture
    
    elif framework == 'jax':
        eqx = framework_modules['eqx']
        return eqx.tree_deserialise_leaves(model_path, model_architecture)
    
    else:
        raise ValueError(f"Unknown framework: {framework}")

from wolframclient.language import wl
from wolframclient.serializers import export as wlexport
from wolframclient.deserializers import WXFConsumer, binary_deserialize, WXFConsumerNumpy
Complex = complex


class wlConsumer(WXFConsumer):
    def build_function(self, head, args, **kwargs):
        # return a built in complex if head is Complex and argument length is 2.
        if head == wl.Complex and len(args) == 2:
            return complex(*args)
        elif head == wl.NumericArray:
            return [np.array(x) for x in args[0]]            
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
            if k == 'monomials' or k == 'coeffs':    
                args_dict[k] = [np.array(x) for x in v]
            else:
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

    args['monomials'] = [x.astype(int) for x in args['monomials']]
    args['coeffs'] = [x.astype(complex) for x in args['coeffs']]
    
    point_gen = PointGeneratorMathematica(args['monomials'], args['coeffs'], args['KahlerModuli'], args['ambient_dims'], precision=args['Precision'], point_file_path=args['point_file_path'], selected_t=args['selected_t'])

    # save point generator to pickle
    mcy_logger.info("Saving point generator to {:}".format(os.path.join(os.path.abspath(args['Dir']), "point_gen.pickle")))
    with open(os.path.join(os.path.abspath(args['Dir']), "point_gen.pickle"), 'wb') as hnd:
        pickle.dump(point_gen, hnd)
    
    kappa = prepare_dataset(point_gen, args['num_pts'], args['Dir'], normalize_to_vol_j=True, ltails=0)
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
    
    kappa = prepare_dataset(point_gen, args['num_pts'], args['Dir'], normalize_to_vol_j=True, ltails=0)
    mcy_logger.info("Computing derivatives of J_FS, Omega, ...")
    prepare_basis_pickle(point_gen, args['Dir'], kappa)
    mcy_logger.debug("done")


def generate_points_mc(my_args):
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

    args['monomials'] = [x.astype(int) for x in args['monomials']]
    args['coeffs'] = [x.astype(complex) for x in args['coeffs']]

    point_gen = CICYPointGeneratorMC(args['monomials'], args['coeffs'], args['KahlerModuli'], args['ambient_dims'])

    # save point generator to pickle
    mcy_logger.info("Saving point generator to {:}".format(os.path.join(os.path.abspath(args['Dir']), "point_gen.pickle")))
    with open(os.path.join(os.path.abspath(args['Dir']), "point_gen.pickle"), 'wb') as hnd:
        pickle.dump(point_gen, hnd)

    kappa = prepare_dataset(point_gen, args['num_pts'], args['Dir'], normalize_to_vol_j=True, ltails=0)
    mcy_logger.info("Computing derivatives of J_FS, Omega, ...")
    prepare_basis_pickle(point_gen, args['Dir'], kappa)
    mcy_logger.debug("done")


def generate_points_mc_toric(my_args):
    global mcy_logger
    args = to_numpy_arrays(my_args)
    mcy_logger.setLevel(args['logger_level'])
    mcy_logger.debug("Using output directory {}".format(os.path.abspath(args['Dir'])))

    args_str = re.sub('\], \n', '], ', str(args))
    args_str = re.sub(' +', ' ', str(args_str))
    mcy_logger.debug(args_str)

    with open(os.path.join(args['Dir'], 'toric_data.pickle'), 'rb') as f:
        toric_data = pickle.load(f)
    for key in toric_data:
        mcy_logger.debug(key)
        mcy_logger.debug(toric_data[key])

    point_gen = ToricCICYPointGeneratorMC(toric_data, args['KahlerModuli'])

    # save point generator to pickle
    mcy_logger.info("Saving point generator to {:}".format(os.path.join(os.path.abspath(args['Dir']), "point_gen.pickle")))
    with open(os.path.join(os.path.abspath(args['Dir']), "point_gen.pickle"), 'wb') as hnd:
        pickle.dump(point_gen, hnd)

    kappa = prepare_dataset(point_gen, args['num_pts'], args['Dir'], normalize_to_vol_j=True, ltails=0)
    mcy_logger.info("Computing derivatives of J_FS, Omega, ...")
    prepare_basis_pickle(point_gen, args['Dir'], kappa)
    mcy_logger.debug("done")


def train_NN(my_args):
    global mcy_logger
    
    args = to_numpy_arrays(my_args)
    mcy_logger.setLevel(args['logger_level'])
    mcy_logger.debug(args)
    
    # Get framework-specific modules
    try:
        framework_modules = get_framework_modules()
        mcy_logger.debug(f"Using {framework_modules['framework']} framework")
    except ImportError as e:
        mcy_logger.error(f"No framework available: {e}")
        return {}
    
    # get info of generated points
    data = np.load(os.path.join(args['Dir'], 'dataset.npz'))
    BASIS = prepare_basis(pickle.load(open(os.path.join(args['Dir'], 'basis.pickle'), 'rb')))
    kappa = BASIS['KAPPA']

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
    # check whether framework is running on GPU or CPU:
    device_available = framework_modules['device_check_fn']()
    device_str = "GPU" if device_available else "CPU"
    mcy_logger.debug("Using {} for computation.".format(device_str))
    
    # extract architecture for NN
    nfold_tensor = BASIS['NFOLD']
    if framework_modules['framework'] == 'tensorflow':
        nfold = framework_modules['cast_fn'](nfold_tensor, framework_modules['float32']).numpy()
    elif framework_modules['framework'] == 'jax':
        nfold = float(framework_modules['jnp'].array(nfold_tensor, dtype=framework_modules['float32']))
    else:  # torch
        if hasattr(nfold_tensor, 'numpy'):
            nfold = nfold_tensor.numpy()
        else:
            nfold = nfold_tensor
        nfold = float(nfold)
    
    n_in = data['X_train'].shape[1]
    n_hiddens, acts = args["HiddenLayers"], args["ActivationFunctions"]
    n_out = int(nfold**2)
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
    try:
        cmetrics = [SigmaLoss(), KaehlerLoss(), TransitionLoss(), RicciLoss(), VolkLoss()]
        cmetrics = [x for x, y in zip(cmetrics, args['PrintLosses']) if y]
    except:
        cmetrics = None  # Metrics not available
    cmetrics = None  # Don't use metrics anymore
    
    # build model using framework-agnostic function
    model = build_model_architecture(n_in, n_hiddens, acts, n_out, framework_modules)
    
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
    
    # Create optimizer using framework-agnostic function
    optimizer = create_optimizer(model, args['LearningRate'], framework_modules)
    
    # Print model summary if available
    if hasattr(model, 'summary'):
        model.summary(print_fn=mcy_logger.debug)
    else:
        mcy_logger.debug(f"Model architecture: {model}")

    # train model
    fsmodel, training_history = train_model(fsmodel, data, optimizer=optimizer, epochs=args['Epochs'], batch_sizes=args['BatchSizes'], verbose=2, custom_metrics=cmetrics, callbacks=cb_list)
        
    # save trained model using framework-agnostic function
    model_ext = framework_modules['model_ext']
    model_path = os.path.join(args['Dir'], f'model{model_ext}')
    mcy_logger.debug(f"Saving model to: {model_path}")
    mcy_logger.debug(f"Directory exists: {os.path.exists(args['Dir'])}")
    save_model(fsmodel.model, model_path, framework_modules)
    
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

    # Get framework-specific modules
    try:
        framework_modules = get_framework_modules()
    except ImportError as e:
        mcy_logger.error(f"No framework available: {e}")
        return []

    # load toric data if exists/needed
    toric_data = None
    if args['Model'] == 'PhiFSToric':
        if os.path.exists(args['toric_data_path']):
            toric_data = pickle.load(open(args['toric_data_path'], 'rb'))
        else:
            mcy_logger.error("Model set to {}, but {} with toric data not found.".format(args['Model'], args['toric_data_path']))
        
    BASIS = prepare_basis(pickle.load(open(os.path.join(args['Dir'], 'basis.pickle'), 'rb')))
    kappa = BASIS['KAPPA']
    
    # Convert points to tensor using framework-agnostic function
    pts_tensor = framework_modules['tensor_fn'](pts, dtype=framework_modules['float32'])
    
    # Load model using framework-agnostic function
    model_ext = framework_modules['model_ext']
    model_path = os.path.join(args['Dir'], f'model{model_ext}')
    
    if framework_modules['framework'] == 'tensorflow':
        # TensorFlow model loading
        tfk = framework_modules['tfk']
        model = tfk.models.load_model(model_path)
    elif framework_modules['framework'] == 'jax':
        # JAX/equinox model loading — reconstruct architecture then deserialise
        mcy_logger.warning("JAX model loading requires architecture reconstruction")
        eqx = framework_modules['eqx']
        nfold_tensor = BASIS['NFOLD']
        nfold = float(framework_modules['jnp'].array(nfold_tensor))
        n_in = pts.shape[1]
        n_out = int(nfold ** 2)
        if args['Model'] in ('PhiFS', 'PhiFSToric'):
            n_out = 1
        model_arch = build_model_architecture(n_in, [64, 64], ['relu', 'relu'], n_out, framework_modules)
        model = load_model(model_path, model_arch, framework_modules)
    else:
        # PyTorch model loading - need to reconstruct architecture
        # This is a limitation - we need to store model architecture info
        mcy_logger.warning("PyTorch model loading requires architecture reconstruction")
        # For now, create a simple model and load weights
        # In production, you'd want to save/load architecture metadata
        torch = framework_modules['torch']
        # Create a simple model (this should match the training architecture)
        model = torch.nn.Sequential(
            torch.nn.Linear(pts.shape[1], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64), 
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1, bias=False)
        )
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
    
    # Create appropriate fsmodel
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

    gs = fsmodel(pts_tensor)
    
    # Convert output to numpy
    if hasattr(gs, 'numpy'):
        return gs.numpy()
    elif hasattr(gs, 'detach'):
        return gs.detach().numpy()
    else:
        return np.array(gs)


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

    # Get framework-specific modules
    try:
        framework_modules = get_framework_modules()
    except ImportError as e:
        mcy_logger.error(f"No framework available: {e}")
        return []

    # load toric data if exists/needed
    toric_data = None
    if args['Model'] == 'PhiFSToric':
        if os.path.exists(args['toric_data_path']):
            toric_data = pickle.load(open(args['toric_data_path'], 'rb'))
        else:
            mcy_logger.error("Model set to {}, but {} with toric data not found.".format(args['Model'], args['toric_data_path']))
        
    BASIS = prepare_basis(pickle.load(open(os.path.join(args['Dir'], 'basis.pickle'), 'rb')))
    
    # Convert points to tensor using framework-agnostic function
    pts_tensor = framework_modules['tensor_fn'](pts, dtype=framework_modules['float32'])
    
    # Load model using framework-agnostic function
    model_ext = framework_modules['model_ext']
    model_path = os.path.join(args['Dir'], f'model{model_ext}')
    
    if framework_modules['framework'] == 'tensorflow':
        # TensorFlow model loading
        tfk = framework_modules['tfk']
        model = tfk.models.load_model(model_path)
    elif framework_modules['framework'] == 'jax':
        # JAX/equinox model loading — reconstruct architecture then deserialise
        mcy_logger.warning("JAX model loading requires architecture reconstruction")
        eqx = framework_modules['eqx']
        nfold_tensor = BASIS['NFOLD']
        nfold = float(framework_modules['jnp'].array(nfold_tensor))
        n_in = pts.shape[1]
        n_out = 1  # PhiFS models always output 1
        model_arch = build_model_architecture(n_in, [64, 64], ['relu', 'relu'], n_out, framework_modules)
        model = load_model(model_path, model_arch, framework_modules)
    else:
        # PyTorch model loading - similar limitation as get_g
        torch = framework_modules['torch']
        model = torch.nn.Sequential(
            torch.nn.Linear(pts.shape[1], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(), 
            torch.nn.Linear(64, 1, bias=False)
        )
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
    
    if args['Model'] == 'PhiFS':
        fsmodel = PhiFSModel(model, BASIS)
    elif args['Model'] == 'PhiFSToric':
        fsmodel = PhiFSModelToric(model, BASIS, toric_data=toric_data)
    else:
        mcy_logger.error("Calculating the Kahler potential for model {} is not supported".format(args['Model']))
        return []

    ks = fsmodel.get_kahler_potential(pts_tensor)
    
    # Convert output to numpy
    if hasattr(ks, 'numpy'):
        return ks.numpy()
    elif hasattr(ks, 'detach'):
        return ks.detach().numpy() 
    else:
        return np.array(ks)

    
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
