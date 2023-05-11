# tested with Sage 9.x
from sage.all import *
import itertools
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
    HH = tv.toric_divisor_group()
    divs = [HH(c) for c in tv.fan(dim=1)]
    section_vertices_dual_poly = (sum(k * d for k, d in zip(ks, divs))).sections()
    return [[w*fan.ray(i) + ks[i] for i in range(fan.nrays())] for w in section_vertices_dual_poly]


def get_rho(fan, kahler_cone_div):
    ks = [kahler_cone_div.coefficient(i) for i in range(fan.nrays())]
    return get_sections_of_line_bundle(fan, ks)


def get_w_of_nowhere_vanishing_monomial(fan, kahler_cone_div, patch_index):
    # coord_exponents = to_patch(fan, patch_index)
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
    no_vanish_pos = 0
    for no_vanish_pos, s in enumerate(sections_homogeneous):
        if vertex_mask.dot_product(vector(s)) == 0:
            break

    section_vertices_dual_poly = kahler_cone_div.sections()
    w_star = section_vertices_dual_poly[no_vanish_pos]
    return w_star

    
def intersection_numbers(tv):
    """ Takes TV, returns intersection numbers  """
    HH = tv.cohomology_ring()
    c1 = HH(-tv.K())
    dim_cy = tv.ambient_space().dimension() - 1

    # generator of Kahler cone
    J = []
    for D in tv.Kaehler_cone().rays():
        J += [HH(D.lift())]

    intersection_nums = np.zeros(tuple([len(J) for _ in range(int(dim_cy))]), dtype=int)
    for inds in itertools.product(range(len(J)), repeat=dim_cy):
        integrand = c1
        for i in inds:
            integrand *= J[i]
        intersection_nums[inds] += tv.integrate(integrand)

    return intersection_nums


def get_homogeneous_coords_from_s_alphas(fan, kahler_cone_div, patch_index):
    vertices = fan.rays()
    section_vertices_dual_poly = kahler_cone_div.sections()
    w_star = get_w_of_nowhere_vanishing_monomial(fan, kahler_cone_div, patch_index)
    return [[(w-w_star)*v for v in vertices] for w in section_vertices_dual_poly]


def get_section_relations(tv, k_monoms, exps_sections, use_groebner):
    # compute the relation among the sections
    if use_groebner:
        # to do so, work on the quotient ring of the coordinate ring by the KC sections and perform a primary decomposition of the defining ideal
        k_monoms_flat = [x for k in k_monoms for x in k]  # flattened list of sections of KC generators
        toric_coord_names = list(tv.coordinate_ring().variable_names())
        section_names = ['s' + str(i) + "_" + str(a) + "__" for i in range(len(exps_sections)) for a in range(len(exps_sections[i]))]  # construct names for sections: s_a_i is the i^th homogenous coordiante in the a^th projective ambient space
        all_var_names = section_names + toric_coord_names
        full_ring = PolynomialRing(QQ, all_var_names, order='invlex')  # this ordering ensures that the toric variables (z) are eliminated in favor of the section coordinates (s) when possible
        quot_ring = full_ring.quo([x-y for x, y in zip(full_ring.gens(), k_monoms_flat)])

        # Find relation among the s[a,i]
        prim_dec_ideals = quot_ring.defining_ideal().primary_decomposition()
        # get rid of stuff that contains z's
        section_relation_coeffs, section_relation_exps = [], []
        for pdi in prim_dec_ideals:
            for e in pdi.gens():
                coeffs, exps = e.coefficients(), e.exponents()
                if any([any(x[len(section_names):]) for x in exps]):
                    continue  # skip relations that involve a z
                section_relation_coeffs.append([x for x in coeffs])
                section_relation_exps.append([list(x[:len(section_names)]) for x in exps])
    else:
        # The individual sections are expressed as s_j= x^(e[a,j])
        # We are looking for a relation \prod_j s_j^alpha[j] = \prod_k s_k^beta[k] <=> \prod_r s_r^\gamma[r] = 1
        # Now, s_r = x^(e[a,j]), so the \gamma[r] will be in the kernel of the matrix e[a,j].
        exps_flat = [e for es in exps_sections for e in es]
        rels = matrix(exps_flat).kernel().matrix().rows()
        section_relation_coeffs, section_relation_exps = [], []
        for rel in rels:
            section_relation_coeffs.append([int(1), int(-1)])
            lhs, rhs = [], []
            for e in rel:
                if e > 0:
                    lhs.append(int(e))
                    rhs.append(int(0))
                elif e < 0:
                    lhs.append(int(0))
                    rhs.append(int(-e))
                else:
                    lhs.append(int(0))
                    rhs.append(int(0))
            section_relation_exps.append([lhs, rhs])

    # use numpy to convert from sage int to python int
    return np.array(section_relation_coeffs, dtype=int).tolist(), np.array(section_relation_exps, dtype=int).tolist()


def prepare_toric_cy_data(tv, out_dir, exp_aK=None, coeff_aK=None, use_groebner=False):
    tv_fan = tv.fan()
    KC = tv.Kaehler_cone()
    MC = tv.Mori_cone()
    KC_gens = tv.Kaehler_cone().rays()
    k_monoms = [[x for x in k.lift().sections_monomials()] for k in KC_gens]

    # Complex dimension of CY
    dim_cy = tv.ambient_space().dimension() - 1

    # compute exponents of anti-canonical surface
    if exp_aK is None:
        exp_aK = get_sections_of_line_bundle(tv_fan, [1 for _ in range(tv_fan.nrays())])

    # compute random point in CS moduli space
    if coeff_aK is None:
        T = RealDistribution('gaussian', 1)
        res, ims = [T.get_random_element() for _ in range(len(exp_aK))], [T.get_random_element() for _ in range(len(exp_aK))]
        coeff_aK = [r+1.j*i for r, i in zip(res, ims)]

    # compute section monomials for each Kahler cone generator
    exps_sections = []
    for r in KC_gens:
        exps_sections.append(get_rho(tv_fan, r.lift()))

    # in each patch, compute the Kahler potential and the section mask indicating which coordinates are set to 1 (indicated by a 1 in the mask, all others have a 0)
    patch_masks = []
    exps_kappa = []
    for i in range(len(tv_fan.cones(tv_fan.dim()))):
        # Kahler metric (Fubini-Study equivalent) exponents
        tmp_kappas = []
        for r in KC_gens:
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

    section_relation_coeffs, section_relation_exps = get_section_relations(tv, k_monoms, exps_sections, use_groebner)

    # compute volume normalization at (t_1,t_2,...) = (1,1,...)
    kahler_form = sum(r.lift().cohomology_class() for r in KC.rays())
    integrand = -tv.K().cohomology_class()
    for _ in range(int(dim_cy)):
        integrand *= kahler_form
    vol_j = tv.integrate(integrand)
    int_nums = intersection_numbers(tv)
    res = {
        "dim_cy":		 int(dim_cy),
        "vol_j_norm":	 int(vol_j),
        "coeff_aK":		 [complex(x) for x in coeff_aK],
        "exp_aK":		 [[int(x) for x in m] for m in exp_aK],
        "exps_sections": [[list(x.exponents()[0]) for x in y] for y in k_monoms],
        "patch_masks":	 [[int(x) for x in m] for m in patch_masks],
        "glsm_charges":	 [[int(x[i]) for i in range(len(x)-1)] for x in MC.rays()],
        "non_ci_coeffs": section_relation_coeffs,
        "non_ci_exps":	 section_relation_exps,
        "int_nums":		 int_nums,
    }

    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    with open(out_dir, 'wb') as hnd:
        pickle.dump(res, hnd)
    return res
