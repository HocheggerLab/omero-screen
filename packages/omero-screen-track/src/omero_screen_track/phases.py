"""PIP-FUCCI cell-cycle phases along tracked cells, with boundary uncertainty.

Each cell's track (background-subtracted PIP and geminin per frame) is
segmented into G1 → S → G2 → X by a left-to-right hidden semi-Markov model:

``G1``  PIP high, geminin at its floor.
``S``   PIP degraded (PCNA-dependent), geminin accumulating.
``G2``  PIP back, geminin high.
``E``   exiting without division: geminin being degraded (falling) in one
        continuing nucleus, PIP returning. Reached from S or G2.
``X``   after the exit: G1-like again. A division ends the track before E,
        so E and X appear only in exits. The ``exit`` boundary is the onset
        of E.

Method (see the vault note ``pip_fucci_phase_calling_plan``):

1. **Correct.** Geminin carries a PIP-proportional floor (bleed-through,
   ≈ 0.17 × PIP in 5054). The well's factor ``k`` is fitted on G1-like frames
   and ``k × PIP`` subtracted.
2. **Smooth** with a rolling median (resists one-frame segmentation spikes)
   then a short rolling mean (1 h); windows are set in hours, not frames.
   Longer smoothing shifts onset-timed boundaries by half its window; the
   model's minimum durations already suppress flicker.
3. **Normalise against the well**, not per track: PIP by the well's PIP-high
   reference, geminin by its high reference. A cell held in G1 for the whole
   window therefore stays G1 (per-track min–max scaling would stretch its
   noise into an S phase).
4. **Features:** normalised PIP, normalised geminin and Rob's ratio
   ``(gem − pip)/(gem + pip)``, which cancels the brightening of both
   channels when a nucleus compacts. Geminin's level is what keeps an exit
   from S (geminin falling, PIP returning) from being called G2.
5. **Model.** Each phase is a short chain of sub-states, which sets a minimum
   duration (a plain HMM would let a one-frame S flicker through); the last
   sub-state has a self-loop set from the expected duration. A track may start
   in any sub-state (cells are already mid-cycle at the window start) and end
   anywhere; X is entered only by a transition. Emission means and variances are fitted per well by EM, pooled
   over the well's cells, with X tied to G1.
6. **Posteriors by forward–backward** over the whole track (smoothing, not
   filtering, which would place every boundary late).

Because the model is left-to-right, ``P(phase ≥ G2 at t)`` only rises along a
track and is the cumulative distribution of the S→G2 transition time. The
boundary is therefore reported as its **median** (first frame with P ≥ 0.5,
also the frame from which G2 is called) and a **90 % interval** (P from 0.05
to 0.95). Rob's rule, the frame where the smoothed ratio's derivative turns
negative after its S-phase peak, is reported beside it as a cross-check.
Phase durations come from paths sampled from the posterior (forward
filtering, backward sampling), so their uncertainty is carried, not a
difference of two medians.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

PHASES = ("G1", "S", "G2", "E", "X")
_G1, _S, _G2, _E, _X = range(5)
_N = len(PHASES)
#: Features per frame: normalised PIP, ratio, normalised geminin, geminin's
#: slope, and PIP's slope split into its rising and falling parts (per hour).
#: The split makes the slope one-sided per phase: S may contain falling PIP
#: (S entry) but not rising PIP, so S/G2 lands at the onset of PIP
#: re-accumulation (Rob's convention, the end of replication), not where PIP
#: is already back up.
FEATURES = ("pip_n", "ratio", "gem_n", "gem_slope", "pip_rise", "pip_fall")
#: Emission entries (phase, feature) that EM does not refit: they encode the
#: S/G2 convention (Rob's: S ends at the first return of PIP), so a well's
#: data must not loosen or tighten them. Refitted, S's PIP spread shrank until
#: noise alone ended S at its minimum length.
_FIXED = ((1, 0), (1, 4))  # (S, pip_n), (S, pip_rise)


@dataclass(frozen=True)
class PhaseParams:
    """Settings for :func:`annotate_phases`.

    Attributes:
        interval_minutes: Time between frames.
        median_hours: Rolling-median window.
        mean_hours: Rolling-mean window applied after the median.
        min_hours: Minimum duration of G1, S, G2, E, X (sets the sub-state
            chain).
        mean_hours_phase: Expected duration of G1, S, G2, E, X (sets the
            self-loop of each chain's last sub-state).
        exit_from_s: Of the cells leaving S, the fraction that exit (to E)
            instead of entering G2.
        bleed_pip_quantile: Frames with PIP above this well quantile are the
            candidates for fitting the bleed-through factor ...
        bleed_quantile: ... which is this quantile of geminin/PIP among them
            (the G1 floor, since S/G2 frames have more geminin).
        pip_ref_quantile: Well PIP-high reference (G1/G2 level).
        gem_ref_quantile: Well geminin-high reference (late S / G2 level).
        em_iterations: EM iterations for the emission parameters.
        prior_frames: Weight, in frames, of each phase's starting emission
            values in the EM update (MAP estimate). It keeps EM from swapping
            phase meanings (G2 sliding onto G1 values, say) while letting a
            well's data set the scales.
        min_sd: Floor on emission standard deviations.
        t_dof: Degrees of freedom of the Student-t emissions. Heavy tails let
            transition frames (PIP falling fast at S entry, say), which no
            single phase describes, stay on the path through their neighbours
            instead of forcing a different path.
        n_samples: Paths sampled per cell for duration distributions.
        wide_hours: An S/G2 interval wider than this flags the cell.
        seed: Seed for path sampling.
    """

    interval_minutes: float = 20.0
    median_hours: float = 1.0
    mean_hours: float = 1.0
    min_hours: tuple[float, ...] = (1.0, 4.0, 1.5, 1.0, 1.0)
    mean_hours_phase: tuple[float, ...] = (8.0, 8.0, 4.0, 3.0, 24.0)
    exit_from_s: float = 0.1
    bleed_pip_quantile: float = 0.6
    bleed_quantile: float = 0.25
    pip_ref_quantile: float = 0.75
    gem_ref_quantile: float = 0.95
    em_iterations: int = 10
    prior_frames: float = 300.0
    min_sd: float = 0.05
    t_dof: float = 3.0
    n_samples: int = 300
    wide_hours: float = 2.0
    seed: int = 0

    def frames(self, hours: float) -> int:
        """Hours to a whole number of frames (at least 1)."""
        return max(1, int(round(hours * 60 / self.interval_minutes)))


@dataclass(frozen=True)
class WellReference:
    """Well-level corrections and scales.

    Attributes:
        bleed: Geminin bleed-through per unit PIP.
        pip_high: PIP-high reference (G1/G2 level), after smoothing.
        gem_high: Geminin-high reference, after correction and smoothing.
    """

    bleed: float
    pip_high: float
    gem_high: float


@dataclass
class _Model:
    """Sub-state chain model: transition matrix, start weights, emissions."""

    phase_of: np.ndarray  # sub-state -> phase index
    log_a: np.ndarray  # (n, n) log transition matrix
    log_pi: np.ndarray  # (n,) log start weights
    mean: np.ndarray  # (phases, features) emission means
    sd: np.ndarray  # (phases, features) emission scales
    dof: float = 3.0  # Student-t degrees of freedom
    first: dict[int, int] = field(
        default_factory=dict
    )  # phase -> first sub-state


# -- preprocessing ------------------------------------------------------------


def _rolling(x: pd.Series, window: int, how: str) -> pd.Series:
    r = x.rolling(window, center=True, min_periods=1)
    return r.median() if how == "median" else r.mean()


def _smooth(x: pd.Series, params: PhaseParams) -> pd.Series:
    """Rolling median then mean, windows in hours; gaps stay NaN."""
    med = _rolling(x, params.frames(params.median_hours) | 1, "median")
    out = _rolling(med, params.frames(params.mean_hours) | 1, "mean")
    return out.where(x.notna())


def fit_reference(states: pd.DataFrame, params: PhaseParams) -> WellReference:
    """Fit a well's bleed-through factor and PIP / geminin scales.

    Args:
        states: Per-frame rows with ``pip`` and ``geminin``
            (background-subtracted) for one well.
        params: Settings.
    """
    pip = states["pip"].to_numpy(dtype=float)
    gem = states["geminin"].to_numpy(dtype=float)
    ok = np.isfinite(pip) & np.isfinite(gem) & (pip > 0)
    pip, gem = pip[ok], gem[ok]
    high = pip >= np.quantile(pip, params.bleed_pip_quantile)
    bleed = float(np.quantile(gem[high] / pip[high], params.bleed_quantile))
    bleed = max(bleed, 0.0)
    corrected = np.clip(gem - bleed * pip, 0, None)
    return WellReference(
        bleed=bleed,
        pip_high=float(np.quantile(pip, params.pip_ref_quantile)),
        gem_high=max(
            float(np.quantile(corrected, params.gem_ref_quantile)), 1e-9
        ),
    )


def features(
    cell: pd.DataFrame, ref: WellReference, params: PhaseParams
) -> pd.DataFrame:
    """Corrected, smoothed and normalised signals for one cell, by frame.

    Returns :data:`FEATURES` as columns (NaN where the cell has no mask),
    indexed like ``cell``. ``gem_slope`` is the change of ``gem_n`` per hour.
    """
    pip = cell["pip"].astype(float)
    gem = (cell["geminin"].astype(float) - ref.bleed * pip).clip(lower=0)
    pip_n = (_smooth(pip, params) / ref.pip_high).clip(lower=0)
    gem_n = (_smooth(gem, params) / ref.gem_high).clip(lower=0)
    ratio = (gem_n - pip_n) / (gem_n + pip_n + 1e-9)
    return pd.DataFrame(
        {
            "pip_n": pip_n,
            "ratio": ratio,
            "gem_n": gem_n,
            "gem_slope": _slope(gem_n, params),
            "pip_rise": _slope(pip_n, params).clip(lower=0),
            "pip_fall": _slope(pip_n, params).clip(upper=0),
        }
    )


def _slope(x: pd.Series, params: PhaseParams) -> pd.Series:
    """Change per hour of a smoothed series; NaN where ``x`` is NaN."""
    filled = x.interpolate(limit_direction="both").to_numpy(dtype=float)
    per_hour = 60 / params.interval_minutes
    d = (
        np.gradient(filled) * per_hour
        if len(filled) > 1
        else np.zeros(len(filled))
    )
    return pd.Series(d, index=x.index).where(x.notna())


# -- model --------------------------------------------------------------------


def _build_model(params: PhaseParams) -> _Model:
    """Sub-state chains G1 → S → (G2 | E), G2 → E → X; X absorbing."""
    chains = [params.frames(h) for h in params.min_hours]
    phase_of = np.concatenate([np.full(k, p) for p, k in enumerate(chains)])
    n = len(phase_of)
    first = {p: int(np.argmax(phase_of == p)) for p in range(_N)}
    last = {p: first[p] + chains[p] - 1 for p in range(_N)}
    a = np.zeros((n, n))
    for p in range(_N):
        for i in range(first[p], last[p]):
            a[i, i + 1] = 1.0  # forced march through the minimum duration
        if p == _X:
            a[last[p], last[p]] = 1.0
            continue
        extra = max(
            params.frames(params.mean_hours_phase[p]) - chains[p] + 1, 2
        )
        stay = 1.0 - 1.0 / extra
        leave = 1.0 - stay
        a[last[p], last[p]] = stay
        if p == _G1:
            a[last[p], first[_S]] = leave
        elif p == _S:
            a[last[p], first[_G2]] = leave * (1 - params.exit_from_s)
            a[last[p], first[_E]] = leave * params.exit_from_s
        elif p == _G2:
            a[last[p], first[_E]] = leave
        else:  # E
            a[last[p], first[_X]] = leave
    # A track may begin in G1, S or G2 (uniform over those, then over
    # sub-states). E and X are entered only by a transition: X shares G1's
    # emission, so as a start state it would absorb every long G1 track. A
    # cell that had already exited before the window starts is called G1.
    pi = np.zeros(n)
    for p in (_G1, _S, _G2):
        idx = np.flatnonzero(phase_of == p)
        pi[idx] = 1.0 / (3 * len(idx))
    with np.errstate(divide="ignore"):
        log_a, log_pi = np.log(a), np.log(pi)
    # S/G2 follows Rob's convention (decided 2026-10-23): S ends at the first
    # return of PIP, the switch of the ratio's derivative. S's PIP emission is
    # therefore tight at the floor and allows no rise; on 5054 C2 the model's
    # G2 entry then sits a median 1 frame from Rob's rule (was ~7 frames
    # with a loose S emission).
    # Starting values, refined by EM. Columns follow FEATURES:
    # pip_n, ratio, gem_n, gem_slope, pip_rise, pip_fall.
    # fmt: off
    mean = np.array([
        [1.0, -0.8, 0.03, 0.0, 0.0, 0.0],     # G1
        [0.02, 0.9, 0.4, 0.08, 0.0, -0.05],   # S: PIP degraded, geminin rising
        [0.4, 0.4, 0.9, 0.0, 0.08, 0.0],      # G2: PIP climbing back
        [0.6, 0.0, 0.4, -0.15, 0.1, 0.0],     # E: geminin degraded, PIP back
        [1.0, -0.8, 0.03, 0.0, 0.0, 0.0],     # X: as G1
    ])
    sd = np.array([
        [0.3, 0.25, 0.1, 0.04, 0.03, 0.03],
        [0.03, 0.2, 0.25, 0.06, 0.005, 0.2],  # S: PIP at floor, no rise
        [0.3, 0.35, 0.3, 0.08, 0.08, 0.03],
        [0.35, 0.5, 0.3, 0.1, 0.1, 0.05],
        [0.3, 0.25, 0.1, 0.04, 0.03, 0.03],
    ])
    # fmt: on
    return _Model(
        phase_of=phase_of,
        log_a=log_a,
        log_pi=log_pi,
        mean=mean,
        sd=sd,
        dof=params.t_dof,
        first=first,
    )


def _log_emission(obs: np.ndarray, model: _Model) -> np.ndarray:
    """(T, n) Student-t log-likelihood per frame and sub-state; NaN frames → 0."""
    from math import lgamma, log, pi

    nu = model.dof
    const = lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * log(nu * pi)
    mu, sd = model.mean[model.phase_of], model.sd[model.phase_of]
    z = (obs[:, None, :] - mu[None]) / sd[None]
    ll = const - np.log(sd[None]) - (nu + 1) / 2 * np.log1p(z**2 / nu)
    ll = np.where(np.isfinite(obs)[:, None, :], ll, 0.0)
    return ll.sum(axis=2)


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):  # all -inf → -inf, as intended
        out = np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m
    return np.squeeze(out, axis=axis)


def _forward_backward(
    log_e: np.ndarray, model: _Model
) -> tuple[np.ndarray, np.ndarray, float]:
    """Log forward variables, posterior over sub-states, log-likelihood."""
    t_len, n = log_e.shape
    la = np.empty((t_len, n))
    la[0] = model.log_pi + log_e[0]
    for t in range(1, t_len):
        la[t] = _logsumexp(la[t - 1][:, None] + model.log_a, axis=0) + log_e[t]
    lb = np.zeros((t_len, n))
    for t in range(t_len - 2, -1, -1):
        lb[t] = _logsumexp(
            model.log_a + (log_e[t + 1] + lb[t + 1])[None], axis=1
        )
    ll = float(_logsumexp(la[-1], axis=0))
    post = np.exp(la + lb - ll)
    post /= post.sum(axis=1, keepdims=True)
    return la, post, ll


def _sample_paths(
    la: np.ndarray, model: _Model, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Forward-filtering backward-sampling: (n, T) phase paths."""
    t_len = la.shape[0]
    out = np.empty((n, t_len), dtype=int)

    def draw(logw: np.ndarray) -> int:
        w = np.exp(logw - np.max(logw))
        return int(rng.choice(len(w), p=w / w.sum()))

    for k in range(n):
        s = draw(la[-1])
        out[k, -1] = model.phase_of[s]
        for t in range(t_len - 2, -1, -1):
            s = draw(la[t] + model.log_a[:, s])
            out[k, t] = model.phase_of[s]
    return out


def fit_emissions(
    observations: list[np.ndarray], model: _Model, params: PhaseParams
) -> _Model:
    """EM over a well's cells for the per-phase emission means and SDs.

    X is tied to G1. Updates are MAP estimates shrunk towards the starting
    values (``prior_frames``), so a phase with little posterior mass (E in a
    control well, say) stays near them, and phases cannot swap meanings.
    """
    n_feat = model.mean.shape[1]
    fitted = (_G1, _S, _G2, _E)
    mu0, sd0 = model.mean.copy(), model.sd.copy()
    k = params.prior_frames
    for _ in range(params.em_iterations):
        num = np.zeros((_N, n_feat))
        sq = np.zeros((_N, n_feat))
        den = np.zeros((_N, n_feat))
        for obs in observations:
            if len(obs) == 0:
                continue
            _, post, _ = _forward_backward(_log_emission(obs, model), model)
            g = _phase_posterior(post, model)
            g[:, _G1] += g[:, _X]  # X shares G1's emission
            ok = np.isfinite(obs)
            x = np.where(ok, obs, 0.0)
            for p in fitted:
                w = g[:, p][:, None] * ok
                num[p] += (w * x).sum(axis=0)
                sq[p] += (w * x**2).sum(axis=0)
                den[p] += w.sum(axis=0)
        for p in fitted:
            # MAP update with a conjugate prior centred on the starting values.
            m = (num[p] + k * mu0[p]) / (den[p] + k)
            scatter = sq[p] - 2 * m * num[p] + den[p] * m**2  # Σ w (x - m)²
            v = (scatter + k * (sd0[p] ** 2 + (mu0[p] - m) ** 2)) / (
                den[p] + k
            )
            keep = [f for (q, f) in _FIXED if q == p]
            new_sd = np.sqrt(np.clip(v, params.min_sd**2, None))
            m[keep], new_sd[keep] = mu0[p, keep], sd0[p, keep]
            model.mean[p] = m
            model.sd[p] = new_sd
        model.mean[_X], model.sd[_X] = model.mean[_G1], model.sd[_G1]
    return model


def _phase_posterior(post: np.ndarray, model: _Model) -> np.ndarray:
    """(T, phases) posterior from the (T, sub-states) one."""
    return np.stack(
        [post[:, model.phase_of == p].sum(axis=1) for p in range(_N)], axis=1
    )


# -- boundaries -----------------------------------------------------------------


def _crossing(cdf: np.ndarray, frames: np.ndarray, level: float) -> float:
    hit = np.flatnonzero(cdf >= level)
    return float(frames[hit[0]]) if len(hit) else np.nan


def _boundary(cdf: np.ndarray, frames: np.ndarray) -> dict[str, Any]:
    """Median and 90 % interval of a transition whose CDF along the track is ``cdf``.

    ``state`` says whether the transition happened before the track
    (``before``), inside it (``inside``) or not by its end (``after``).
    """
    if cdf[0] >= 0.5:
        return _no_boundary("before")
    if cdf[-1] < 0.5:
        return _no_boundary("after")
    return {
        "median": _crossing(cdf, frames, 0.5),
        "lo": _crossing(cdf, frames, 0.05),
        "hi": _crossing(cdf, frames, 0.95),
        "state": "inside",
    }


def _no_boundary(state: str) -> dict[str, Any]:
    return {"median": np.nan, "lo": np.nan, "hi": np.nan, "state": state}


def _prefixed(name: str, b: dict[str, Any]) -> dict[str, Any]:
    """``{"median", "lo", "hi", "state"}`` as ``name``, ``name_lo`` ..."""
    return {
        name: b["median"],
        f"{name}_lo": b["lo"],
        f"{name}_hi": b["hi"],
        f"{name}_state": b["state"],
    }


def _rob_g2(
    ratio: np.ndarray,
    frames: np.ndarray,
    s_entry: float,
    end: float,
    params: PhaseParams,
    fall: float = 0.1,
    plateau: float = 0.02,
) -> float:
    """Rob's S/G2 rule: where the smoothed ratio's derivative switches.

    The ratio rises as PIP is degraded at S entry, plateaus while PIP is
    absent (near +1, so its maximum can sit anywhere on the plateau), and
    falls once PIP returns. Rob read the switch on a twice-smoothed ratio,
    i.e. the end of the plateau before the main fall. Here: smooth again over
    ``1.5 × mean_hours``; find the first frame after S entry where the ratio
    has fallen ``fall`` below its running maximum; the switch is the last
    frame before it within ``plateau`` of that maximum.
    """
    if not np.isfinite(s_entry):
        return np.nan
    r = pd.Series(ratio).interpolate(limit_direction="both")
    r = (
        r.rolling(
            params.frames(1.5 * params.mean_hours) | 1,
            center=True,
            min_periods=1,
        )
        .mean()
        .to_numpy()
    )
    idx = np.flatnonzero((frames >= s_entry) & (frames <= end))
    if len(idx) < 3:
        return np.nan
    run_max = np.maximum.accumulate(r[idx])
    fallen = np.flatnonzero(r[idx] < run_max - fall)
    if len(fallen) == 0:
        return np.nan  # no sustained fall before the end
    stop = fallen[0]
    near = np.flatnonzero(r[idx[:stop]] >= run_max[stop] - plateau)
    return float(frames[idx[near[-1]]]) if len(near) else np.nan


def _durations(paths: np.ndarray, hours: float) -> dict[str, float]:
    """Median and 90 % interval of S and G2 durations over sampled paths.

    Only paths in which the phase both starts and ends inside the track count;
    ``*_n`` is the fraction of paths that do.
    """
    out: dict[str, float] = {}
    for name, p in (("s", _S), ("g2", _G2), ("e", _E)):
        lengths = []
        for path in paths:
            idx = np.flatnonzero(path == p)
            if len(idx) and idx[0] > 0 and idx[-1] < len(path) - 1:
                lengths.append(len(idx) * hours)
        lengths_arr = np.array(lengths)
        out[f"{name}_hours"] = (
            float(np.median(lengths_arr)) if len(lengths_arr) else np.nan
        )
        out[f"{name}_hours_lo"] = (
            float(np.quantile(lengths_arr, 0.05))
            if len(lengths_arr)
            else np.nan
        )
        out[f"{name}_hours_hi"] = (
            float(np.quantile(lengths_arr, 0.95))
            if len(lengths_arr)
            else np.nan
        )
        out[f"{name}_hours_n"] = (
            len(lengths_arr) / len(paths) if len(paths) else 0.0
        )
    return out


# -- public -------------------------------------------------------------------


def annotate_phases(
    cells: pd.DataFrame,
    states: pd.DataFrame,
    params: PhaseParams | None = None,
    reference: WellReference | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, WellReference]:
    """Call phases for every cell in ``states``.

    **Call it once per plate, with every well's cells together** (cell ids
    must be unique across wells). The reference scales and the emission model
    are then fitted on the pooled cells, so a phase means the same reporter
    state in every well. Fitted on one well alone, a knockdown well with few
    S/G2 cells gets a low geminin reference that inflates noise into
    spurious S, G2 or exit calls.

    Args:
        cells: One row per cell (``cell`` column), e.g. from
            :func:`~omero_screen_track.fate.follow_cells`.
        states: One row per cell and frame with ``cell``, ``timepoint``,
            ``pip``, ``geminin`` (background-subtracted).
        params: Settings; defaults to :class:`PhaseParams`.
        reference: Scales to use instead of fitting them on ``states``.

    Returns:
        ``(cells, states, reference)``. ``states`` gains ``phase`` (the
        posterior-median phase: G2 from P(≥ G2) ≥ 0.5), ``p_g1`` … ``p_x``
        and the :data:`FEATURES`. ``cells`` gains, per boundary
        (``s_entry``, ``g2_entry``, ``exit`` = onset of E), the median frame,
        ``*_lo`` / ``*_hi`` (90 % interval) and ``*_state`` (before / inside
        / after the track); ``g2_entry_rob`` (Rob's derivative rule); S, G2
        and E durations with intervals; ``start_phase`` / ``end_phase``; and
        ``phase_flags``.
    """
    params = params or PhaseParams()
    ref = reference or fit_reference(states, params)
    hours = params.interval_minutes / 60
    st = states.sort_values(["cell", "timepoint"]).copy()
    feats = {}
    obs = {}
    for cid, g in st.groupby("cell", sort=False):
        # Reindex to every frame so gaps are NaN, not skipped.
        full = g.set_index("timepoint").reindex(
            range(int(g["timepoint"].min()), int(g["timepoint"].max()) + 1)
        )
        f = features(full, ref, params)
        feats[cid] = f
        obs[cid] = f[list(FEATURES)].to_numpy(dtype=float)
    model = fit_emissions(list(obs.values()), _build_model(params), params)
    rng = np.random.default_rng(params.seed)

    frame_rows, cell_rows = [], []
    for cid, f in feats.items():
        frames = f.index.to_numpy()
        la, post, _ = _forward_backward(_log_emission(obs[cid], model), model)
        p = _phase_posterior(post, model)
        cum = np.cumsum(p, axis=1)
        # Called phase: the posterior median along the cycle order, i.e. the
        # first phase whose cumulative probability reaches 0.5. Between S and
        # G2 this is exactly the rule "G2 once P(≥ G2) ≥ 0.5".
        called = np.argmax(cum >= 0.5, axis=1)
        ge_s = 1 - cum[:, _G1]  # P(≥ S): CDF of S entry
        ge_g2 = 1 - cum[:, _S]  # P(≥ G2): CDF of leaving S
        ge_e = 1 - cum[:, _G2]  # P(≥ E): CDF of the exit onset
        fr = pd.DataFrame(
            {
                "cell": cid,
                "timepoint": frames,
                "phase": [PHASES[k] for k in called],
                **{f"p_{ph.lower()}": p[:, k] for k, ph in enumerate(PHASES)},
                **{c: f[c].to_numpy() for c in FEATURES},
            }
        )
        frame_rows.append(fr)

        s_b = _boundary(ge_s, frames)
        # G2 entry is reported only for cells called G2 somewhere; a cell
        # going S → E (exit from S) has no G2 entry.
        g2_b = (
            _boundary(ge_g2, frames)
            if p[:, _G2].max() >= 0.5
            else _no_boundary("after" if ge_g2[0] < 0.5 else "before")
        )
        x_b = _boundary(ge_e, frames)
        end = x_b["median"] if np.isfinite(x_b["median"]) else frames[-1]
        rob = _rob_g2(
            f["ratio"].to_numpy(), frames, s_b["median"], end, params
        )
        paths = _sample_paths(la, model, params.n_samples, rng)
        flags = []
        if g2_b["state"] == "inside":
            width = (g2_b["hi"] - g2_b["lo"]) * hours
            if width > params.wide_hours:
                flags.append("s_g2_wide")
            if np.isfinite(rob) and not (
                g2_b["lo"] - 1 <= rob <= g2_b["hi"] + 1
            ):
                flags.append("rob_disagrees")
        seen = np.flatnonzero(np.isfinite(obs[cid][:, 0]))
        cell_rows.append(
            {
                "cell": cid,
                **_prefixed("s_entry", s_b),
                **_prefixed("g2_entry", g2_b),
                **_prefixed("exit", x_b),
                "g2_entry_rob": rob,
                **_durations(paths, hours),
                "start_phase": PHASES[int(called[seen[0]])]
                if len(seen)
                else "",
                "end_phase": PHASES[int(called[seen[-1]])]
                if len(seen)
                else "",
                "phase_flags": " ".join(flags),
            }
        )

    per_frame = pd.concat(frame_rows, ignore_index=True)
    keep = [
        c
        for c in states.columns
        if c not in per_frame.columns or c in ("cell", "timepoint")
    ]
    out_states = states[keep].merge(
        per_frame, on=["cell", "timepoint"], how="left"
    )
    per_cell = pd.DataFrame(cell_rows)
    keep_c = [
        c for c in cells.columns if c not in per_cell.columns or c == "cell"
    ]
    out_cells = cells[keep_c].merge(per_cell, on="cell", how="left")
    return out_cells, out_states, ref
