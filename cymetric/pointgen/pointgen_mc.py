"""
MCMC PointGenerator for CICY and toric CY manifolds.

:Authors:
    Fabian Ruehle f.ruehle@northeastern.edu

Two classes are provided:

  - :class:`CICYPointGeneratorMC`: tangent-space Metropolis-Hastings sampler
    for CICY manifolds defined by multiple hypersurfaces in products of
    projective spaces.

  - :class:`ToricPointGeneratorMC`: same sampler adapted to toric CY
    hypersurfaces, using the GLSM structure produced by
    ``sage_lib.prepare_toric_cy_data``.

Both classes target the measure :math:`|\Omega|^2` on the CY, where
:math:`\Omega` is the holomorphic volume form obtained from the Poincaré
residue theorem.

Point-generation recipe
-----------------------
1. Find an initial feasible point by Newton-projecting a random ambient-space
   point onto the constraint surface.
2. Run a short pilot chain to auto-tune the step size (targeting
   ``target_rate`` acceptance, default 30 %) and estimate the integrated
   autocorrelation time (IAC) for the thinning factor.
3. Burn-in and then collect one sample every ``thin`` steps.

MCMC step
---------
Given the current point :math:`z`:

1. Draw :math:`\omega \sim \mathcal{N}(0, I)` in :math:`\mathbb{C}^N`.
2. Project :math:`\omega` onto the holomorphic tangent space by subtracting
   all components along the constraint Jacobian rows:

   .. math::
       \omega \leftarrow \omega - J^\dagger (J J^\dagger)^{-1} J\,\omega

3. Propose :math:`z' = z + \text{step\_size}\cdot\omega`.
4. Newton-project :math:`z'` back onto the CY: minimum-norm step
   :math:`\delta = -J^\dagger (J J^\dagger)^{-1} Q(z')`.
5. Metropolis accept/reject with ratio
   :math:`|\Omega(z')|^2 / |\Omega(z)|^2`.

The proposal is symmetric (Gaussian in the tangent space), so the
Metropolis ratio reduces to the target density ratio up to an
:math:`O(\text{step\_size}^2)` Jacobian-of-projection correction that is
absorbed into the acceptance (standard practice for constrained MCMC).

Auto-tuning heuristics
----------------------
* **Step size**: a short pilot chain is run for ``n_rounds`` rounds of
  ``n_pilot`` steps each; after each round the step size is scaled by
  ``observed_rate / target_rate`` and clamped to ``[1e-6, 1e3]``.
* **Thinning**: the integrated autocorrelation time of :math:`|z|^2` is
  estimated from a pilot chain using Geyer's initial positive-sequence
  estimator (FFT-accelerated), and ``thin = ceil(IAC)`` is returned.
* **Burn-in**: defaults to ``max(100, 10 * thin)`` when not provided.
"""

import numpy as np
import logging
import sympy as sp
try:
    from tqdm import tqdm
except ImportError:
    class tqdm:  # noqa: N801
        """Minimal fallback used when tqdm is not installed.

        Prints '5% done', '10% done', … at 5-percentage-point intervals.
        Accepts the same positional/keyword arguments as ``tqdm.tqdm`` so
        the call-sites in :meth:`generate_points` need not change.
        """
        def __init__(self, iterable=None, *, total=None, desc='',
                     disable=False, **_kwargs):
            self._iter = iter(iterable) if iterable is not None else iter([])
            self._total = (total if total is not None
                           else (len(iterable) if hasattr(iterable, '__len__')
                                 else 0))
            self._done = 0
            self._desc = desc
            self._disable = disable
            self._next_report = 5  # next percentage milestone to print

        def __iter__(self):
            return self

        def __next__(self):
            val = next(self._iter)  # propagates StopIteration naturally
            self._done += 1
            if not self._disable and self._total > 0:
                pct = int(100 * self._done / self._total)
                while pct >= self._next_report:
                    print('{}: {}% done'.format(self._desc, self._next_report),
                          flush=True)
                    self._next_report += 5
            return val

        def set_description(self, desc, **_kwargs):
            self._desc = desc

        def set_postfix(self, *_args, **_kwargs):
            pass

        def close(self):
            pass

from cymetric.pointgen.pointgen_cicy import CICYPointGenerator
from cymetric.pointgen.nphelper import prepare_dataset, get_levicivita_tensor

logging.basicConfig(format='%(name)s:%(levelname)s:%(message)s')
logger = logging.getLogger('MCpointgen')


# ---------------------------------------------------------------------------
# CICYPointGeneratorMC
# ---------------------------------------------------------------------------

class CICYPointGeneratorMC(CICYPointGenerator):
    r"""MCMC-based point generator for CICY manifolds.

    Targets :math:`|\Omega|^2` on the complete-intersection CY via a
    tangent-space Metropolis-Hastings chain.  All geometry (pullback
    tensors, weight computation, etc.) is inherited from
    :class:`CICYPointGenerator`; only the point-generation routine is
    replaced.

    The expensive root-basis construction (only needed for the algebraic
    point generator) is skipped to speed up initialisation.

    Example:
        A simple example on a generic CY manifold of the family defined
        by the following configuration matrix:

        .. math::
            X \in [5|33]

        can be set up with

        >>> import numpy as np
        >>> from cymetric.pointgen.pointgen_mc import CICYPointGeneratorMC
        >>> from cymetric.pointgen.nphelper import generate_monomials
        >>> monomials = np.array(list(generate_monomials(6, 3)))
        >>> monomials_per_hyper = [monomials, monomials]
        >>> coeff = [np.random.randn(len(m)) for m in monomials_per_hyper]
        >>> kmoduli = np.ones(1)
        >>> ambient = np.array([5])
        >>> pg = CICYPointGeneratorMC(monomials_per_hyper, coeff,
        ...                           kmoduli, ambient)

        Generate a dataset with

        >>> pg.prepare_dataset(number_of_points, dir_name)
    """

    def __init__(self, monomials, coefficients, kmoduli, ambient,
                 step_size=None, burn_in=None, thin=None, seed=2021,
                 verbose=2, **kwargs):
        r"""Initializer.

        All positional and keyword arguments except those listed below are
        forwarded to :class:`CICYPointGenerator`.

        Args:
            monomials (list(ndarray[(nMonomials, ncoords), int])): monomials
                per hypersurface.
            coefficients (list(ndarray[(nMonomials)])): coefficients per
                hypersurface.
            kmoduli (ndarray[(nProj)]): Kaehler moduli.
            ambient (ndarray[(nProj), int]): ambient projective space
                dimensions.
            step_size (float, optional): MCMC proposal step size.  If None,
                auto-tuned from a pilot chain. Defaults to None.
            burn_in (int, optional): number of burn-in steps.  If None,
                set to ``max(100, 10 * thin)`` after thinning is determined.
                Defaults to None.
            thin (int, optional): thinning interval (keep every ``thin``-th
                step after burn-in).  If None, estimated from the integrated
                autocorrelation time. Defaults to None.
            seed (int, optional): RNG seed used by default in
                :meth:`generate_points`. Defaults to 2021.
            verbose (int, optional): Logging level. 1=Debug, 2=Info,
                else=Warning. Also controls the tqdm progress bar
                (disabled when verbose > 2). Defaults to 2.
            **kwargs: forwarded to :class:`CICYPointGenerator`.
        """
        if verbose == 1:
            level = logging.DEBUG
        elif verbose == 2:
            level = logging.INFO
        else:
            level = logging.WARNING
        logger.setLevel(level=level)

        super().__init__(monomials, coefficients, kmoduli, ambient,
                         verbose=verbose, **kwargs)
        self._mc_step_size = step_size
        self._mc_burn_in = burn_in
        self._mc_thin = thin
        self._mc_seed = seed
        self._verbose = verbose

    # ------------------------------------------------------------------
    # Override: skip the expensive root-basis (not needed for MCMC)
    # ------------------------------------------------------------------

    def _generate_all_bases(self):
        r"""Override to skip root-basis generation (not needed for MCMC).

        ``_generate_root_basis`` performs expensive symbolic substitutions
        that are only required by the algebraic point generator.  We skip it
        here and only build the dQdz basis used for gradient evaluation.
        """
        self.all_ts = self._generate_all_freets()
        self.selected_t = self._find_degrees()
        # Intentionally skip _generate_root_basis(): not needed for MCMC.
        self._generate_dQdz_basis()
        self.dzdz_generated = False
        self._generate_padded_basis()

    # ------------------------------------------------------------------
    # Low-level single-point helpers
    # ------------------------------------------------------------------

    def _eval_cy(self, z):
        r"""Evaluate all defining polynomials at a single point.

        Args:
            z (ndarray[(ncoords,), complex]): a single point.

        Returns:
            ndarray[(nhyper,), complex]: :math:`(Q_1(z), \ldots, Q_K(z))`.
        """
        vals = np.zeros(self.nhyper, dtype=complex)
        for i in range(self.nhyper):
            vals[i] = np.sum(
                self.coefficients[i] *
                np.prod(z[None, :] ** self.monomials[i], axis=-1))
        return vals

    def _jacobian_at_point(self, z):
        r"""Compute the Jacobian :math:`J_{ij} = \partial Q_i / \partial z_j`
        at a single point, using the precomputed monomial basis.

        Args:
            z (ndarray[(ncoords,), complex]): a single point.

        Returns:
            ndarray[(nhyper, ncoords), complex]: constraint Jacobian.
        """
        J = np.zeros((self.nhyper, self.ncoords), dtype=complex)
        for k in range(self.nhyper):
            J[k] = self._compute_dQdz(z[np.newaxis], k)[0]
        return J

    def _omega_density(self, z):
        r"""Target density :math:`|\Omega(z)|^2` at a single point (up to
        overall normalisation).

        Uses the Poincaré-residue holomorphic volume form from the parent.
        Returns 0 for singular or numerically degenerate points.

        Args:
            z (ndarray[(ncoords,), complex]): a point on the CY.

        Returns:
            float: :math:`|\Omega(z)|^2 \geq 0`.
        """
        omega = self.holomorphic_volume_form(z[np.newaxis])
        rho = float(abs(omega[0]) ** 2)
        return rho if np.isfinite(rho) else 0.0

    def _project_to_cy(self, z, max_iter=30, tol=1e-12):
        r"""Newton-project ``z`` onto the constraint surface
        :math:`\{Q_i = 0\}`.

        Uses the minimum-norm Newton step

        .. math::
            \delta = -J^\dagger (J J^\dagger)^{-1} Q(z)

        which lives in the row space of :math:`J` and therefore does not
        disturb the tangential component of :math:`z`.

        Args:
            z (ndarray[(ncoords,), complex]): starting point.
            max_iter (int, optional): maximum Newton iterations.
                Defaults to 30.
            tol (float, optional): convergence tolerance. Defaults to 1e-12.

        Returns:
            tuple(ndarray[(ncoords,), complex], bool):
                ``(projected_point, converged)``.
        """
        z = z.copy()
        for _ in range(max_iter):
            q = self._eval_cy(z)
            if np.max(np.abs(q)) < tol:
                return z, True
            J = self._jacobian_at_point(z)
            JJh = J @ J.conj().T  # (nhyper, nhyper)
            try:
                coeff = np.linalg.solve(JJh, q)
            except np.linalg.LinAlgError:
                return z, False
            z = z - J.conj().T @ coeff
        q = self._eval_cy(z)
        return z, bool(np.max(np.abs(q)) < 1e-6)

    def _tangent_step(self, z, step_size, rng):
        r"""Sample a Gaussian step in the holomorphic tangent space at ``z``.

        Draws :math:`\omega \sim \mathcal{N}(0, I)` in
        :math:`\mathbb{C}^N`, then projects out all components along the
        constraint Jacobian rows:

        .. math::
            \omega \leftarrow \omega
                - J^\dagger (J J^\dagger)^{-1} J\,\omega

        Args:
            z (ndarray[(ncoords,), complex]): current point.
            step_size (float): step scale.
            rng (np.random.Generator): random number generator.

        Returns:
            ndarray[(ncoords,), complex]: tangent-space displacement.
        """
        omega = (rng.standard_normal(self.ncoords) +
                 1j * rng.standard_normal(self.ncoords))
        J = self._jacobian_at_point(z)
        JJh = J @ J.conj().T
        try:
            omega = omega - J.conj().T @ np.linalg.solve(JJh, J @ omega)
        except np.linalg.LinAlgError:
            pass
        return step_size * omega

    def _normalize_point(self, z):
        r"""Put ``z`` in canonical projective patch.

        For each projective factor :math:`\mathbb{P}^n`, divides all
        coordinates in that factor by the one with the largest absolute value
        (so the maximum becomes :math:`1 + 0j`).

        Args:
            z (ndarray[(ncoords,), complex]): a point.

        Returns:
            ndarray[(ncoords,), complex]: normalised point.
        """
        return self._rescale_points(z[np.newaxis])[0]

    def _find_initial_point(self, rng, max_tries=500, acc=1e-8):
        r"""Find an initial feasible point on the CY.

        Repeatedly draws random points from the product of ambient spheres
        and Newton-projects them onto the constraint surface.

        Args:
            rng (np.random.Generator): random number generator.
            max_tries (int, optional): maximum attempts. Defaults to 500.
            acc (float, optional): required CY accuracy. Defaults to 1e-8.

        Returns:
            ndarray[(ncoords,), complex]: initial point on the CY.

        Raises:
            RuntimeError: if no feasible point is found within ``max_tries``.
        """
        for _ in range(max_tries):
            parts = []
            for n in self.ambient:
                raw = rng.standard_normal(2 * (int(n) + 1)).view(np.complex128)
                raw = raw / np.linalg.norm(raw)
                parts.append(raw)
            z = np.concatenate(parts)
            z, ok = self._project_to_cy(z)
            if ok:
                return self._normalize_point(z)
        raise RuntimeError(
            "Could not find an initial CICY point after %d tries." % max_tries)

    def _mcmc_step(self, z, step_size, rng):
        r"""One Metropolis-Hastings step targeting :math:`|\Omega|^2`.

        Args:
            z (ndarray[(ncoords,), complex]): current point.
            step_size (float): proposal step scale.
            rng (np.random.Generator): random number generator.

        Returns:
            tuple(ndarray[(ncoords,), complex], bool):
                ``(new_point, accepted)``.
        """
        pi_old = self._omega_density(z)
        delta = self._tangent_step(z, step_size, rng)
        z_prop, ok = self._project_to_cy(z + delta)
        if not ok:
            return z, False
        z_prop = self._normalize_point(z_prop)
        pi_new = self._omega_density(z_prop)

        # If the current point is degenerate, always accept the proposal.
        if pi_old < 1e-300:
            return z_prop, True

        log_ratio = np.log(max(pi_new, 1e-300)) - np.log(pi_old)
        if np.log(max(rng.random(), 1e-300)) < log_ratio:
            return z_prop, True
        return z, False

    # ------------------------------------------------------------------
    # Auto-tuning helpers
    # ------------------------------------------------------------------

    def _tune_step_size(self, z0, rng, n_pilot=300, target_rate=0.3,
                        n_rounds=6):
        r"""Adapt ``step_size`` via a pilot chain to approach
        ``target_rate`` acceptance.

        Runs ``n_rounds`` rounds of ``n_pilot`` steps each; after each
        round the step size is scaled by
        :math:`\text{observed\_rate} / \text{target\_rate}` and clamped
        to :math:`[10^{-6},\, 10^3]`.

        The initial step size is 10 % of the average absolute coordinate
        magnitude, which is a reasonable starting point for normalised
        projective coordinates.

        Args:
            z0 (ndarray): starting point.
            rng (np.random.Generator): random number generator.
            n_pilot (int, optional): steps per tuning round. Defaults to 300.
            target_rate (float, optional): target acceptance rate.
                Defaults to 0.3.
            n_rounds (int, optional): number of tuning rounds.
                Defaults to 6.

        Returns:
            tuple(float, ndarray):
                ``(tuned_step_size, last_point_of_pilot_chain)``.
        """
        step_size = 0.1 * float(np.mean(np.abs(z0)))
        z = z0.copy()
        rate = 0.0
        for _ in range(n_rounds):
            accepts = 0
            for _ in range(n_pilot):
                z, acc_flag = self._mcmc_step(z, step_size, rng)
                accepts += int(acc_flag)
            rate = accepts / n_pilot
            if rate < 1e-8:
                step_size *= 0.1
            else:
                step_size *= rate / target_rate
            step_size = float(np.clip(step_size, 1e-6, 1e3))
        logger.debug(
            "Step-size tuning: acceptance rate = {:.3f}, "
            "step_size = {:.4g}".format(rate, step_size))
        return step_size, z

    @staticmethod
    def _integrated_autocorrelation(samples):
        r"""Estimate the integrated autocorrelation time (IAC) of a 1-D
        sequence using Geyer's initial positive-sequence estimator
        (FFT-accelerated).

        The IAC is defined as

        .. math::
            \tau = \sum_{k=-\infty}^{+\infty} \rho(k)
                 = 1 + 2 \sum_{k=1}^\infty \rho(k)

        where :math:`\rho(k)` is the normalised autocorrelation at lag
        :math:`k`.  The estimator truncates the sum when consecutive pairs
        :math:`\rho(2k-1) + \rho(2k)` become non-positive.

        Args:
            samples (array_like): 1-D sequence of scalar samples.

        Returns:
            float: estimated IAC (:math:`\geq 1`).
        """
        s = np.asarray(samples, dtype=float)
        s = s - s.mean()
        n = len(s)
        var = float(np.var(s))
        if var < 1e-30 or n < 4:
            return 1.0
        # Normalised ACF via zero-padded FFT
        f = np.fft.rfft(s, n=2 * n)
        acf = np.fft.irfft(f * np.conj(f))[:n].real
        acf /= var * n
        # Geyer's initial positive-sequence estimator
        iac = acf[0]  # == 1.0 by normalisation
        for k in range(1, n // 2):
            gamma = acf[2 * k - 1] + acf[2 * k]
            if gamma <= 0.0:
                break
            iac += gamma
        return max(1.0, float(iac))

    def _estimate_thinning(self, z0, step_size, rng, n_pilot=600):
        r"""Estimate a thinning factor from a short pilot chain.

        Runs ``n_pilot`` steps and records all coordinate components
        (real and imaginary parts separately) at each step, then returns
        :math:`\lceil \max_k \text{IAC}_k \rceil` as the recommended
        thinning factor.

        Using only :math:`|z|^2` as the summary statistic (the previous
        approach) misses the *phase* degrees of freedom, which typically
        de-correlate much more slowly than the amplitudes — especially for
        low-dimensional manifolds such as elliptic curves.  Taking the
        maximum IAC over all components ensures we catch the slowest-mixing
        direction.

        Args:
            z0 (ndarray): starting point.
            step_size (float): step size.
            rng (np.random.Generator): random number generator.
            n_pilot (int, optional): pilot chain length. Defaults to 600.

        Returns:
            int: recommended thinning factor.
        """
        samples = []
        z = z0.copy()
        for _ in range(n_pilot):
            z, _ = self._mcmc_step(z, step_size, rng)
            samples.append(np.concatenate([z.real, z.imag]))
        samples = np.asarray(samples)  # shape (n_pilot, 2*ncoords)
        iac = max(
            self._integrated_autocorrelation(samples[:, k])
            for k in range(samples.shape[1])
        )
        thin = max(1, min(int(np.ceil(iac)), n_pilot // 5))
        logger.debug(
            "Thinning estimation: IAC = {:.2f}, thin = {}".format(iac, thin))
        return thin

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate_points(self, n_p, step_size=None, burn_in=None, thin=None,
                        seed=None, target_rate=0.3, **kwargs):
        r"""Generate ``n_p`` points on the CY using MCMC.

        If ``step_size``, ``burn_in``, or ``thin`` are not supplied here
        *and* were not set at construction time, they are determined
        automatically:

        * ``step_size`` is tuned via a pilot chain to achieve roughly
          ``target_rate`` acceptance.
        * ``thin`` is estimated from the IAC of :math:`|z|^2`.
        * ``burn_in`` defaults to ``max(100, 10 * thin)``.

        Args:
            n_p (int): number of points to return.
            step_size (float, optional): proposal step scale.
            burn_in (int, optional): burn-in length.
            thin (int, optional): thinning interval (keep every ``thin``-th
                step after burn-in).
            seed (int, optional): RNG seed; overrides the constructor seed.
            target_rate (float, optional): target acceptance rate for
                auto-tuning. Defaults to 0.3.
            **kwargs: ignored (for API compatibility).

        Returns:
            ndarray[(n_p, ncoords), np.complex128]: points on the CY.
        """
        rng = np.random.default_rng(
            seed if seed is not None else self._mc_seed)

        logger.info("Finding initial point...")
        z = self._find_initial_point(rng)
        logger.debug("|Q(z_0)| = {:.2e}".format(
            float(np.max(np.abs(self._eval_cy(z))))))

        # ---- step size ------------------------------------------------
        eff_step = step_size if step_size is not None else self._mc_step_size
        if eff_step is None:
            logger.info(
                "Tuning step size (target acceptance {:.0%})...".format(
                    target_rate))
            eff_step, z = self._tune_step_size(
                z, rng, target_rate=target_rate)
            logger.info("  -> step_size = {:.4g}".format(eff_step))

        # ---- thinning -------------------------------------------------
        eff_thin = thin if thin is not None else self._mc_thin
        if eff_thin is None:
            logger.info("Estimating thinning from autocorrelation...")
            eff_thin = self._estimate_thinning(z, eff_step, rng)
            logger.info("  -> thin = {}".format(eff_thin))

        # ---- burn-in --------------------------------------------------
        eff_burn = burn_in if burn_in is not None else self._mc_burn_in
        if eff_burn is None:
            eff_burn = max(100, 10 * eff_thin)
        logger.info(
            "Running chain: burn_in = {}, thin = {}, n_p = {}".format(
                eff_burn, eff_thin, n_p))

        # ---- main loop ------------------------------------------------
        total = eff_burn + n_p * eff_thin
        samples = []
        accepts = 0
        update_every = max(1, total // 200)
        pbar = tqdm(
            range(total),
            desc='MCMC (burn-in)',
            unit='step',
            mininterval=1.0,
            disable=self._verbose > 2,
        )
        for step in pbar:
            z, accepted = self._mcmc_step(z, eff_step, rng)
            if accepted:
                accepts += 1
            if step >= eff_burn:
                if step == eff_burn:
                    pbar.set_description('MCMC (sampling)')
                if (step - eff_burn) % eff_thin == 0:
                    samples.append(z.copy())
            if step % update_every == 0:
                pbar.set_postfix(
                    accept='{:.1%}'.format(accepts / (step + 1)),
                    pts=len(samples))
        pbar.set_postfix(
            accept='{:.1%}'.format(accepts / total),
            pts=len(samples))
        pbar.close()

        logger.info("Acceptance rate: {:.3f} ({}/{})".format(
            accepts / total, accepts, total))
        if len(samples) < n_p:
            logger.warning(
                "Collected {} samples instead of {}".format(
                    len(samples), n_p))
        return np.array(samples[:n_p])

    def generate_point_weights(self, n_pw, omega=False, normalize_to_vol_j=True):
        r"""Generate points with their integration weights.

        Because the MCMC chain samples *exactly* from
        :math:`|\Omega|^2`, every point carries the same weight in the
        target (CY-volume) measure.  Returning uniform weights here is
        therefore correct and avoids the spuriously large weights that
        would arise from the algebraic-sampler formula
        :math:`w = |\Omega|^2 / \det(g^{\rm FS})`.

        If ``normalize_to_vol_j`` is ``True`` the common weight is chosen
        so that

        .. math::

            \frac{1}{N}\sum_i w_i \det(g^{\rm FS}_{\rm pb})|_i
            = \text{vol\_j\_norm}

        which is the same convention as
        :func:`~cymetric.pointgen.pointgen.PointGenerator.point_weight`.
        This path is used by
        :func:`~cymetric.pointgen.nphelper.prepare_dataset`.

        Args:
            n_pw (int): number of point-weight pairs to generate.
            omega (bool, optional): if True, also return
                :math:`\Omega` at each point. Defaults to False.
            normalize_to_vol_j (bool, optional): normalise weights.
                Defaults to True.

        Returns:
            ndarray: structured array with fields ``point``, ``weight``
            (and ``omega`` if requested).
        """
        data_types = [
            ('point', np.complex128, self.ncoords),
            ('weight', np.float64),
        ]
        if omega:
            data_types += [('omega', np.complex128)]
        dtype = np.dtype(data_types)

        points = self.generate_points(n_pw)
        n_p = len(points)

        # Compute Omega once; re-use for both normalisation and output.
        omegas_hol = self.holomorphic_volume_form(points)

        # Uniform weights: MCMC samples exactly from |Omega|^2.
        weights = np.ones(n_p, dtype=np.float64)

        if normalize_to_vol_j:
            omega_sq = np.real(omegas_hol * np.conj(omegas_hol))
            pbs = self.pullbacks(points)
            fs_ref = self.fubini_study_metrics(
                points, vol_js=np.ones_like(self.kmoduli))
            fs_ref_pb = np.einsum(
                'xai,xij,xbj->xab', pbs, fs_ref, np.conj(pbs))
            # For MCMC sampling proportional to |Omega|^2:
            #   E_{MC}[det(g_pb)/|Omega|^2] ~ integral(det(g_pb) dA) / Z
            # so norm_fac * E_{MC}[det(g_pb)] = vol_j_norm.
            norm_fac = self.vol_j_norm / np.mean(
                np.real(np.linalg.det(fs_ref_pb)) / omega_sq)
            weights = norm_fac * np.ones(n_p, dtype=np.float64)

        point_weights = np.zeros(n_p, dtype=dtype)
        point_weights['point'] = points
        point_weights['weight'] = weights
        if omega:
            point_weights['omega'] = omegas_hol
        return point_weights

    def prepare_dataset(self, n_p, dirname, ltails=0., **kwargs):
        r"""Prepare training and validation data using MCMC points.

        Keyword arguments are forwarded to
        :func:`cymetric.pointgen.nphelper.prepare_dataset`.

        Args:
            n_p (int): number of points to generate.
            dirname (str): directory to save the dataset in.
            ltails (float, optional): fraction of points with small weights
                discarded from the left tail. Defaults to 0.

        Returns:
            int: 0
        """
        return prepare_dataset(self, n_p, dirname, ltails=ltails, **kwargs)


# ---------------------------------------------------------------------------
# ToricPointGeneratorMC
# ---------------------------------------------------------------------------

class ToricPointGeneratorMC(CICYPointGeneratorMC):
    r"""MCMC-based point generator for toric CY hypersurfaces.

    Mirrors :class:`~cymetric.pointgen.pointgen_mathematica.ToricPointGeneratorMathematica`
    in its setup (toric data from ``sage_lib.prepare_toric_cy_data``) but
    replaces Mathematica-based point generation with the tangent-space MCMC
    chain from :class:`CICYPointGeneratorMC`.

    The GLSM structure (``patch_masks``, ``glsm_charges``, ``sections``) is
    used for canonical normalisation after each accepted step via a
    log-linear C* rescaling.

    Example:
        We assume ``toric_data`` has been generated with
        :func:`~cymetric.sage.sagelib.prepare_toric_cy_data` beforehand.

        >>> import numpy as np
        >>> from cymetric.pointgen.pointgen_mc import ToricPointGeneratorMC
        >>> kmoduli = np.ones(len(toric_data['exps_sections']))
        >>> pg = ToricPointGeneratorMC(toric_data, kmoduli)
        >>> points = pg.generate_points(1000)

        Generate a dataset with

        >>> pg.prepare_dataset(number_of_points, dir_name)
    """

    def __init__(self, toric_data, kmoduli,
                 step_size=None, burn_in=None, thin=None,
                 seed=2021, verbose=2):
        r"""Initializer.

        Args:
            toric_data (dict): generated by
                ``sage_lib.prepare_toric_cy_data(TV, fname)``.  Required keys:
                ``dim_cy``, ``exp_aK``, ``coeff_aK``, ``exps_sections``,
                ``non_ci_coeffs``, ``non_ci_exps``, ``patch_masks``,
                ``glsm_charges``, ``vol_j_norm``, ``int_nums``.
            kmoduli (ndarray[(h^{1,1})]): Kaehler moduli.
            step_size (float, optional): MCMC proposal step size.
                Auto-tuned if None. Defaults to None.
            burn_in (int, optional): burn-in length.  Defaults to
                ``max(100, 10 * thin)`` if None.
            thin (int, optional): thinning interval.  Estimated from IAC
                if None. Defaults to None.
            seed (int, optional): RNG seed. Defaults to 2021.
            verbose (int, optional): logging verbosity. 1=Debug, 2=Info,
                else Warning. Defaults to 2.
        """
        # Set up logging first (the parent logger is shared).
        if verbose == 1:
            level = logging.DEBUG
        elif verbose == 2:
            level = logging.INFO
        else:
            level = logging.WARNING
        logger.setLevel(level=level)

        # ------------------------------------------------------------------
        # Toric structure (mirrors ToricPointGeneratorMathematica.__init__)
        # ------------------------------------------------------------------
        self.toric_data = toric_data
        self.nfold = int(toric_data['dim_cy'])
        self.sections = toric_data['exps_sections']
        self.non_CI_coeffs = toric_data['non_ci_coeffs']
        self.non_CI_exps = toric_data['non_ci_exps']
        self.patch_masks = np.array(toric_data['patch_masks'], dtype=bool)
        self.glsm_charges = np.array(toric_data['glsm_charges'])
        self.vol_j_norm = toric_data['vol_j_norm']
        self.intersection_tensor = np.array(toric_data['int_nums'])
        self.kmoduli = np.asarray(kmoduli, dtype=complex)
        self.ambient_dims = np.array(
            [len(s) + 1 for s in toric_data['exps_sections']])

        # Monomials and coefficients in list form (nhyper = 1).
        monomials = [np.array(toric_data['exp_aK'], dtype=np.int64)]
        coefficients = [np.array(toric_data['coeff_aK'], dtype=complex)]
        self.nhyper = 1
        self.nmonomials, self.ncoords = monomials[0].shape
        self.monomials = monomials
        self.coefficients = coefficients

        # Levi-Civita tensor for holomorphic volume form.
        self.lc = get_levicivita_tensor(self.nfold)

        # The 'ambient' attribute is a hack used by the inherited FS-metric
        # and weight routines.  Set it to the section dimensions as in the
        # existing toric generators (will be updated if needed).
        self.ambient = self.ambient_dims.copy()
        self.backend = 'multiprocessing'

        # Sympy setup needed by _generate_dQdz_basis.
        self.x = sp.var('x0:' + str(self.ncoords))
        self.poly = sum(
            self.coefficients[0] *
            np.multiply.reduce(
                np.power(self.x, self.monomials[0]), axis=-1))

        # MCMC hyper-parameters.
        self._mc_step_size = step_size
        self._mc_burn_in = burn_in
        self._mc_thin = thin
        self._mc_seed = seed
        self._verbose = verbose

        # Build dQdz basis (all we need for gradient evaluation).
        self._set_seed(seed)
        self._generate_dQdz_basis()
        self.dzdz_generated = False
        self._generate_padded_basis()

        # ------------------------------------------------------------------
        # Pre-compute per-patch GLSM inverse matrices for fast normalisation.
        # For each patch, the charge matrix Q_patch = glsm_charges[:, pc].T
        # has shape (|pc|, nP).  We store its (pseudo-)inverse so that
        #   log_lambda = Q_patch^{-1} @ (-log z_patch)
        # can be computed with a single matrix-vector product.
        # ------------------------------------------------------------------
        self._patch_coords = [np.where(pm)[0] for pm in self.patch_masks]
        self._patch_mat_inv = []
        for pc in self._patch_coords:
            Q_patch = self.glsm_charges[:, pc].T  # (|pc|, nP)
            try:
                if Q_patch.shape[0] == Q_patch.shape[1]:
                    self._patch_mat_inv.append(np.linalg.inv(Q_patch))
                else:
                    self._patch_mat_inv.append(np.linalg.pinv(Q_patch))
            except np.linalg.LinAlgError:
                self._patch_mat_inv.append(np.linalg.pinv(Q_patch))

    # ------------------------------------------------------------------
    # Toric overrides
    # ------------------------------------------------------------------

    def _normalize_point(self, z):
        r"""Apply GLSM C* rescaling to put ``z`` in the canonical toric patch.

        Iterates over toric patches.  For each patch with coordinate indices
        ``pc``, solves for the C* scaling parameters :math:`\lambda` via
        log-linear algebra:

        .. math::
            \log \lambda = -Q_{\rm patch}^{-1} \log z_{\rm patch}

        and accepts the patch if the resulting point satisfies
        :math:`\max_i |z_i| \leq 1 + \epsilon`.

        Falls back to plain max-abs normalisation if no patch succeeds
        (e.g. when a coordinate is numerically zero).

        Args:
            z (ndarray[(ncoords,), complex]): a point.

        Returns:
            ndarray[(ncoords,), complex]: GLSM-normalised point.
        """
        z_work = z.copy()
        for pc, Q_inv in zip(self._patch_coords, self._patch_mat_inv):
            z_patch = z_work[pc]
            if np.any(np.abs(z_patch) < 1e-300):
                continue
            try:
                log_lam = Q_inv @ (-np.log(z_patch))
                scalings = np.exp(log_lam @ self.glsm_charges)
                tmp = scalings * z_work
                if np.max(np.abs(tmp)) <= 1.0 + 1e-8:
                    return tmp
            except (np.linalg.LinAlgError, FloatingPointError):
                continue
        # Fallback: plain max-abs normalisation.
        max_abs = np.max(np.abs(z_work))
        if max_abs > 0:
            z_work = z_work / max_abs
        return z_work

    def _find_initial_point(self, rng, max_tries=500, acc=1e-8):
        r"""Find an initial toric CY point.

        Draws random points from :math:`(\mathbb{C}^*)^N` (standard complex
        normal, rescaled to keep coordinates order-1) and Newton-projects
        onto the CY equation.

        Args:
            rng (np.random.Generator): random number generator.
            max_tries (int, optional): maximum attempts. Defaults to 500.
            acc (float, optional): required CY accuracy. Defaults to 1e-8.

        Returns:
            ndarray[(ncoords,), complex]: initial GLSM-normalised point.

        Raises:
            RuntimeError: if no feasible point is found.
        """
        for _ in range(max_tries):
            z = (rng.standard_normal(self.ncoords) +
                 1j * rng.standard_normal(self.ncoords))
            # Avoid extremely large/small coordinates.
            max_abs = np.max(np.abs(z))
            if max_abs > 0:
                z = z / max_abs
            z, ok = self._project_to_cy(z)
            if ok:
                return self._normalize_point(z)
        raise RuntimeError(
            "Could not find an initial toric CY point after "
            "%d tries." % max_tries)

    def prepare_dataset(self, n_p, dirname,
                        val_split=0.1, ltails=0, rtails=0):
        r"""Prepare training and validation data.

        Args:
            n_p (int): number of points to generate.
            dirname (str): directory to save the dataset in.
            val_split (float, optional): train/validation split.
                Defaults to 0.1.
            ltails (float, optional): fraction discarded on the left tail
                of the weight distribution. Defaults to 0.
            rtails (float, optional): fraction discarded on the right tail
                of the weight distribution. Defaults to 0.

        Returns:
            float: kappa = vol_k / vol_cy.
        """
        return prepare_dataset(
            self, n_p, dirname,
            val_split=val_split, ltails=ltails, rtails=rtails,
            normalize_to_vol_j=True)

    def fubini_study_metrics(self, points, vol_js=None):
        return self._fubini_study_n_metrics(points, kfactors=vol_js)
   
    def _fubini_study_n_metrics(self, points, kfactors=None):
        r"""Computes the FS metric of points.

        Args:
            point (ndarray[(np, n), complex]): point
            kfactors (list): volume factor.

        Returns:
            ndarray[(len(points), n, n), complex]: g^FS
        """
        kfactors = self.kmoduli if kfactors is None else kfactors
        Js = np.zeros([len(points), len(self.sections[0][0]), len(self.sections[0][0])], dtype=np.complex128)
        for alpha in range(len(kfactors)):
            ms = np.transpose(np.prod([np.power(points, self.sections[alpha][a]) for a in range(len(self.sections[alpha]))], axis=-1), [1, 0])
            mss = ms*np.conj(ms)
            kappa_alphas = np.sum(mss, -1)
            J_alphas = 1/(points[:, :, np.newaxis] * np.conj(points[:, np.newaxis, :]))
            J_alphas = np.einsum('x,xab->xab', 1/(kappa_alphas**2), J_alphas)
            coeffs = np.einsum('xa,xb,ai,aj->xij', mss, mss, np.array(self.sections[alpha], dtype=np.complex128), np.array(self.sections[alpha], dtype=np.complex128)) - np.einsum('xa,xb,ai,bj->xij', mss, mss, np.array(self.sections[alpha], dtype=np.complex128), np.array(self.sections[alpha], dtype=np.complex128))
            Js += J_alphas * coeffs * complex(kfactors[alpha]) / complex(np.pi)
        return Js
