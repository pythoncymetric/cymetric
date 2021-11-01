import numpy as np

# ML packages
import tensorflow as tf
from cymetric.pointgen.nphelper import generate_monomials

@tf.function
def function_basis_tf(points, points_bar, ambient_coords,
        section_degrees = None, deg = 1):

    nProj = len(ambient_coords)-1
    npoints = len(points)
    if section_degrees == None:
        for i in range(nProj):
            ni_vars = ambient_coords[i+1] - ambient_coords[i]
            tmp_degrees = [tf.cast(np.array(list(
                generate_monomials(ni_vars, deg))), dtype=points.dtype)]
            if i == 0:
                section_degrees = tmp_degrees
            else:
                section_degrees += tmp_degrees 
    #print(section_degrees)
    for i in range(nProj):
        tmp_denom = tf.reduce_sum(
            (points[:,ambient_coords[i]:ambient_coords[i+1]]*\
             points_bar[:,ambient_coords[i]:ambient_coords[i+1]])**deg, axis=-1)
        if i == 0:
            denominator = tmp_denom
        else:
            denominator = tf.reduce_prod([denominator, tmp_denom], axis=-2)
        #print(i, denominator.shape, tmp_denom.shape)
    #print(denominator.shape)
    for i in range(nProj):
        tmp_num = tf.math.pow(
            points[:,tf.newaxis,ambient_coords[i]:ambient_coords[i+1]],
            section_degrees[i])
        tmp_num_bar = tf.math.pow(
            points_bar[:,tf.newaxis,ambient_coords[i]:ambient_coords[i+1]],
            section_degrees[i])
        tmp_num = tf.reduce_prod(tmp_num, axis=-1)
        tmp_num_bar = tf.reduce_prod(tmp_num_bar, axis=-1)
        if i == 0:
            numerator = tf.einsum('xi,xj->xij', tmp_num, tmp_num_bar)
            numerator = tf.reshape(numerator, (npoints, -1))
        else:
            tmp_numerator = tf.einsum('xi,xj->xij', tmp_num, tmp_num_bar)
            tmp_numerator = tf.reshape(tmp_numerator, (npoints, -1))
            numerator = tf.einsum('xi,xj->xij', numerator, tmp_numerator)
            numerator = tf.reshape(numerator, (npoints, -1))
    #print(numerator.shape)
    fbasis = numerator/denominator[:,np.newaxis]
    # the first ncoords basis will be identical
    return fbasis

def get_section_degrees(ambient_coords, deg):
    nProj = len(ambient_coords)-1
    deg = 1
    ambient_coords=[0,3,6]
    for i in range(nProj):
        ni_vars = ambient_coords[i+1] - ambient_coords[i]
        tmp_degrees = [tf.cast(np.array(list(
            generate_monomials(ni_vars, deg))), dtype=tf.complex64)]
        if i == 0:
            section_degrees = tmp_degrees
        else:
            section_degrees += tmp_degrees
    return section_degrees

def compute_df(points, ambient_coords, deg):
    section_degrees = get_section_degrees(ambient_coords, deg)
    points_bar = tf.math.conj(points)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(points)
        tape.watch(points_bar)
        f = function_basis_tf(points, points_bar,
                              section_degrees = section_degrees,
                              ambient_coords = ambient_coords,
                              deg = deg)
    df = tape.batch_jacobian(f, points)
    dbf = tape.batch_jacobian(f, points_bar)
    return df, dbf

def compute_laplacian_parts(model, data, deg, ambient_coords):
    weights = data['y_val'][:,-2]
    kappa = tf.cast(1/tf.math.reduce_mean(weights), dtype=tf.complex64)
    points = tf.cast(data['X_val'], tf.float32)
    cpoints = tf.complex(points[:, :model.ncoords],
                         points[:, model.ncoords:])
    predictions = model.predict(points)
    pullbacks = tf.cast(data['val_pullbacks'], dtype=cpoints.dtype)
    ginv = tf.linalg.inv(predictions)
    detg = tf.linalg.det(predictions)
    int_weight = tf.cast(detg*weights, dtype=tf.complex64)
    ginv_ambient = tf.einsum('xib,xij,xja->xba',
        pullbacks, ginv, tf.math.conj(pullbacks))
    df, dfb = compute_df(cpoints, ambient_coords, deg)
    #check transpose etc
    l1 = tf.einsum('xfi,xij,xgj->xfg', tf.math.conj(df), ginv_ambient, dfb)
    l2 = tf.einsum('xfi,xij,xgj->xfg', tf.math.conj(dfb), ginv_ambient, df)
    laplacian = kappa*tf.einsum('xfg,x->fg', l1+l2, int_weight)
    return laplacian, df, dfb
