# written for Sage 9f
from sage.all import *
import itertools as it
import numpy as np
import os
import pickle

def to_patch(fan, patch_index):
    # given a fan and a desired patch_index = 0, 1, ..., number(dual_cones), returns the exponents of the homogeneous coordinates in that patch such that the product of the hom coords with these exponents gives the affine coordinates
    top_cones = fan.cones(fan.dim())
    dual_cones_gens = [top_cone.dual().rays() for top_cone in top_cones]
    return [[v*w for v in fan.rays()] for w in dual_cones_gens[patch_index]]

def get_sections_of_line_bundle(fan, ks):
    # given a fan and coefficients ks, computes the sections of D=ks[i]*D[i] where D are the toric divisors
    tv = ToricVariety(fan)
    vertices = fan.rays()
    HH = tv.toric_divisor_group()
    divs = [HH(c) for c in tv.fan(dim=1)]
    section_vertices_dual_poly = (sum(k*d for k,d in zip(ks, divs))).sections()
    return [[w*fan.ray(i) + ks[i] for i in range(fan.nrays())] for w in section_vertices_dual_poly]

def get_rho(fan, kahler_cone_div):
    ks = [kahler_cone_div.coefficient(i) for i in range(fan.nrays())]
    return get_sections_of_line_bundle(fan, ks)

def get_w_of_nowhere_vanishing_monomial(fan, kahler_cone_div, patch_index):
    coord_exponents = to_patch(fan, patch_index)
    # get all sections in homogeneous coordinates
    ks = [kahler_cone_div.coefficient(i) for i in range(fan.nrays())]
    sections_homogeneous = get_sections_of_line_bundle(fan, ks)
    
    # identify nowhere-vanishing monomial. Remember that top_cone[patch_index] contains the vertices of the coordinates that can vanish
    top_cone_gens = fan.cones(fan.dim())[patch_index].rays()
    vertices = fan.rays()
    # find the index of the vertices (i.e. the toric coordinate), that can vanish
    vertex_mask = [0 for _ in range(len(fan.rays()))]
    for g in top_cone_gens:
        for i, v in enumerate(vertices):
            if v == g:
                vertex_mask[i] = 1
                break
    
    # now identify no-where vanishing monomial using this mask
    vertex_mask = vector(vertex_mask)
    for no_vanish_pos, s in enumerate(sections_homogeneous):
        if vertex_mask.dot_product(vector(s)) == 0:
            break

    section_vertices_dual_poly = kahler_cone_div.sections()
    w_star = section_vertices_dual_poly[no_vanish_pos]
    return w_star

def triple_intersection(TV):
    """ Takes TV returns triple intersection numbers  """
    HH = TV.cohomology_ring()
    c1 = HH(-TV.K())
    
    #generator of cohomology
    #VB = TV.toric_divisor_group().gens()
    #D = [HH(VB[i]) for i in range(len(VB))]

    #generator of kählercone
    J = []
    for D in TV.Kaehler_cone().rays():
        J += [HH(D.lift())]

        #return j

    triple = np.zeros((len(J), len(J), len(J)), dtype=np.int32)
    for i, j, k in it.product(range(len(J)), repeat=3):
        triple[i,j,k] += TV.integrate(c1*J[i]*J[j]*J[k])

    return triple

def get_homogeneous_coords_from_s_alphas(fan, kahler_cone_div, patch_index):
    vertices = fan.rays()
    section_vertices_dual_poly = kahler_cone_div.sections()
    w_star = get_w_of_nowhere_vanishing_monomial(fan, kahler_cone_div, patch_index)
    return [[(w-w_star)*v for v in vertices] for w in section_vertices_dual_poly]

def prepare_toric_cy_data(tv, out_dir, exp_aK=None, coeff_aK=None):
    tv_fan = tv.fan()
    KC = tv.Kaehler_cone()
    MC = tv.Mori_cone()
    
    # Complex dimension of CY
    dim_cy = tv.ambient_space().dimension() - 1
    
    # compute exponents of anti-canonical surface
    if exp_aK is None:
        exp_aK = get_sections_of_line_bundle(tv_fan, [1 for _ in range(tv_fan.nrays())])
    
    # compute random point in CS moduli space
    if coeff_aK is None:
        T = RealDistribution('gaussian', 1)
        res, ims = [T.get_random_element() for _ in range(len(exp_aK))], [T.get_random_element() for _ in range(len(exp_aK))]
        coeff_aK = [r+1.j*i for r,i in zip(res, ims)]
    
    # compute section monomials for each Kahler cone generator
    exps_sections = []
    for r in KC.rays():
        exps_sections.append(get_rho(tv_fan, r.lift()))
    
    # in each patch, compute the Kahler potential and the section mask indicating which coordinates are set to 1 (indicated by a 1 in the mask, all others have a 0)
    patch_masks = []
    exps_kappa = []
    for i in range(len(tv_fan.cones(tv_fan.dim()))):
        # Kahler metric (Fubini-Study equivalent) exponents
        tmp_kappas = []
        for r in KC.rays():
        	tmp_kappas.append(get_homogeneous_coords_from_s_alphas(tv_fan, r.lift(), i))
        exps_kappa.append(tmp_kappas)
        
        # patch masks
        hom_exps = to_patch(tv_fan, i)
        non_zero_vars = [0 for _ in range(len(hom_exps[0]))]
        for exps in hom_exps:
            for j, e in enumerate(exps):
                if e < 0:
                    non_zero_vars[j] = 1
        patch_masks.append(non_zero_vars)

    # compute volume normalization at (t_1,t_2,...) = (1,1,...)
    kahler_form = sum(r.lift().cohomology_class() for r in KC.rays())
    vol_j = tv.integrate(-tv.K().cohomology_class() * kahler_form * kahler_form * kahler_form)
    triple = triple_intersection(tv)
    res = {
        "dim_cy":        int(dim_cy), 
        "vol_j_norm":    int(vol_j), 
        "coeff_aK":      [complex(x) for x in coeff_aK], 
        "exp_aK":        [[int(x) for x in m] for m in exp_aK], 
        "exps_sections": [[[int(x) for x in m] for m in s] for s in exps_sections], 
        "patch_masks":   [[int(x) for x in m] for m in patch_masks], 
        "glsm_charges":	 [[int(x[i]) for i in range(len(x)-1)] for x in MC.rays()],
        "triple":        triple,
    #    "exps_kappa":    [[[[int(x) for x in m] for m in s] for s in ek] for ek in exps_kappa]  # n ot needed in the end
    }
    
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    with open(out_dir, 'wb') as hnd:
        pickle.dump(res, hnd)
    return res
