#!/usr/bin/env python3
"""
mcmc_results.py — analyse a run_noda_mcmc NPZ.

Prints medians / 68% credible intervals / R-hat / ESS, the NO-vs-IO
posterior odds (both-ordering runs), and the TI evidence; writes a corner
plot, the dm2_31 marginal posterior, and trace plots.

Usage:
  noda_mcmc_results --input mcmc_osc.npz --outdir Plots/mcmc
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .ptmcmc import PTSampler

LABELS = {
    'sin2_th12': r'$\sin^2\theta_{12}$',
    'sin2_th13': r'$\sin^2\theta_{13}$',
    'dm2_21':    r'$\Delta m^2_{21}\,[10^{-5}\,\mathrm{eV}^2]$',
    'dm2_31':    r'$\Delta m^2_{31}\,[10^{-3}\,\mathrm{eV}^2]$',
    'me_rho':    r'$\rho_\mathrm{matter}$',
}


def _label(p):
    return LABELS.get(p, p)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, nargs='+',
                   help='one or more NPZ files from run_noda_mcmc (e.g. '
                        'independent CNAF chains with different seeds, or '
                        '--resume segments); they are merged, with each '
                        'file x walker treated as an independent chain')
    p.add_argument('--outdir', default='Plots/mcmc')
    p.add_argument('--burn-frac', type=float, default=0.0,
                   help='extra fraction of the stored chain to discard')
    p.add_argument('--corner-params', default='',
                   help='comma list of params for the corner plot '
                        '(default: shared osc params present, else first 8)')
    p.add_argument('--prefix', default='')
    return p.parse_args()


def credible(x, level=0.68):
    lo = np.percentile(x, 50 * (1 - level))
    hi = np.percentile(x, 50 * (1 + level))
    return np.median(x), lo, hi


def main():
    cli = parse()
    os.makedirs(cli.outdir, exist_ok=True)
    pre = cli.prefix or os.path.splitext(os.path.basename(cli.input[0]))[0]

    # ── load & merge (multiple files = independent chains / segments) ────
    names = j31 = both = None
    chains, logls, ti_list = [], [], []
    for path in cli.input:
        z = np.load(path, allow_pickle=True)
        f_names = [str(s) for s in z['param_names']]
        if names is None:
            names, j31, both = f_names, int(z['j_dm31']), bool(z['both_mode'])
        elif f_names != names or int(z['j_dm31']) != j31 \
                or bool(z['both_mode']) != both:
            raise RuntimeError(f"{path} has a different param set / mode "
                               f"than {cli.input[0]} — cannot merge")
        c, l = z['chain'], z['logl']
        nburn = int(cli.burn_frac * c.shape[0])
        chains.append(c[nburn:])
        logls.append(l[nburn:])
        if 'logZ_TI' in z:
            ti_list.append((os.path.basename(path), float(z['logZ_TI'])))

    # crop all segments to the shortest so (file x walker) stack as chains
    n_min = min(c.shape[0] for c in chains)
    chain = np.concatenate([c[-n_min:] for c in chains], axis=1)
    logl  = np.concatenate([l[-n_min:] for l in logls], axis=1)
    n, w, d = chain.shape
    flat = chain.reshape(-1, d)
    print(f"loaded {len(cli.input)} file(s): {n} sweeps x {w} "
          f"(file x walker) chains x {d} params "
          f"({flat.shape[0]} samples after discard)")

    # ── convergence diagnostics ──────────────────────────────────────────
    try:
        import arviz as az
        idata = az.from_dict(posterior={p: chain[:, :, i].T
                                        for i, p in enumerate(names)})
        rhat = az.rhat(idata)
        ess  = az.ess(idata)
        diag = {p: (float(rhat[p].values), float(ess[p].values))
                for p in names}
    except Exception as e:                  # pragma: no cover
        print(f"(arviz diagnostics unavailable: {e})")
        diag = {p: (np.nan, np.nan) for p in names}

    print(f"\n{'param':<24s}{'median':>14s}{'-68%':>12s}{'+68%':>12s}"
          f"{'R-hat':>8s}{'ESS':>10s}")
    for i, p in enumerate(names):
        med, lo, hi = credible(flat[:, i])
        r, e = diag[p]
        print(f"{p:<24s}{med:>14.6g}{lo - med:>12.3g}{hi - med:>+12.3g}"
              f"{r:>8.3f}{e:>10.0f}")
    # NOTE for both-mode: R-hat/ESS treat walkers as chains; a bimodal
    # dm2_31 legitimately shows R-hat > 1 unless every walker crosses
    # orderings many times — judge dm2_31 with the per-ordering table below.

    # ── NMO posterior odds ───────────────────────────────────────────────
    if both:
        sign = flat[:, j31] > 0
        f_no = float(sign.mean())
        # Clopper-Pearson (beta) interval on the NO fraction, evaluated on
        # the EFFECTIVE number of independent samples (raw MCMC samples are
        # autocorrelated) — same convention as ORSA's calculate_BF_unc but
        # ESS-scaled.
        from scipy.stats import beta as _beta
        n_eff = max(diag.get('dm2_31', (np.nan, flat.shape[0]))[1], 1.0)
        if not np.isfinite(n_eff):
            n_eff = flat.shape[0]
        k_eff = f_no * n_eff
        p_lo = _beta.ppf(0.15865, k_eff, n_eff - k_eff + 1) if k_eff > 0 else 0.0
        p_hi = (_beta.ppf(1 - 0.15865, k_eff + 1, n_eff - k_eff)
                if k_eff < n_eff else 1.0)
        err = 0.5 * (p_hi - p_lo)
        print(f"\nP(NO|data) = {f_no:.4f} (68% CI [{p_lo:.4f}, {p_hi:.4f}], "
              f"ESS-based)    P(IO|data) = {1 - f_no:.4f}")
        if 0 < f_no < 1:
            bf = f_no / (1 - f_no)
            # equal prior volume per ordering -> posterior odds = Bayes
            # factor B. 2 ln B is directly comparable to the frequentist
            # dchi2(IO-NO); Kass & Raftery: 2-6 positive, 6-10 strong,
            # >10 very strong.
            twolnB = 2 * np.log(bf)
            err_2lnB = 2 * err / max(f_no * (1 - f_no), 1e-12)
            kr = ('barely worth mentioning' if abs(twolnB) < 2 else
                  'positive' if abs(twolnB) < 6 else
                  'strong' if abs(twolnB) < 10 else 'very strong')
            print(f"posterior odds NO/IO = {bf:.4g}   "
                  f"(log10 = {np.log10(bf):+.2f})")
            print(f"2 ln(B) = {twolnB:+.2f} +- {err_2lnB:.2f}   "
                  f"[Kass-Raftery: {kr}; compare frequentist dchi2]")
        signs = chain[:, :, j31] > 0                 # (n, walkers)
        flips = int(np.sum(signs[1:] != signs[:-1]))
        print(f"cold-chain NO<->IO transitions: {flips} total "
              f"({flips / w:.1f}/walker) — want >~100 total for a stable "
              f"Bayes factor")
        for lab, msk in (('NO', sign), ('IO', ~sign)):
            if msk.sum() < 10:
                print(f"  {lab}: <10 samples")
                continue
            med, lo, hi = credible(np.abs(flat[msk, j31]))
            print(f"  {lab}: |dm2_31| = {med:.4f} {lo - med:+.4f} "
                  f"{hi - med:+.4f}  ({msk.sum()} samples)")

    if ti_list:
        print()
        for fname, ti in ti_list:
            print(f"TI log-evidence [{fname}] = {ti:.3f} "
                  f"(accurate only with a deep ladder, beta_min ~ 1e-3)")
    if 'acc_swap' in z:
        tag = f" [{os.path.basename(cli.input[-1])}]" if len(cli.input) > 1 else ""
        print(f"swap acceptance per pair{tag}: "
              f"{np.array2string(z['acc_swap'], precision=2)}")
        if both and 'acc_jump' in z:
            print(f"NO<->IO jump acceptance per temp{tag}: "
                  f"{np.array2string(z['acc_jump'], precision=2)}")

    # ── corner plot ──────────────────────────────────────────────────────
    if cli.corner_params:
        cp = [s.strip() for s in cli.corner_params.split(',') if s.strip()]
    else:
        osc = ['sin2_th12', 'sin2_th13', 'dm2_21', 'dm2_31', 'me_rho']
        cp = [p for p in osc if p in names] or names[:8]
    ci = [names.index(p) for p in cp]
    try:
        import corner
        fig = corner.corner(flat[:, ci], labels=[_label(p) for p in cp],
                            bins=50, quantiles=[0.16, 0.5, 0.84],
                            show_titles=True, title_fmt='.4g',
                            plot_datapoints=False, fill_contours=True)
        path = os.path.join(cli.outdir, f"{pre}_corner.png")
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"\nwrote {path}")
    except Exception as e:                  # pragma: no cover
        print(f"(corner plot skipped: {e})")

    # ── dm2_31 marginal ──────────────────────────────────────────────────
    x31 = flat[:, j31]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(x31, bins=300, density=True, histtype='stepfilled', alpha=0.7)
    ax.set_xlabel(_label('dm2_31'), fontsize=13)
    ax.set_ylabel('posterior density', fontsize=13)
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    if both:
        ax.axvline(0, color='k', lw=0.5)
    path = os.path.join(cli.outdir, f"{pre}_dm2_31_marginal.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {path}")

    # ── traces (dm2_31 + logL) ───────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    nshow = min(8, w)
    for k in range(nshow):
        axes[0].plot(chain[:, k, j31], lw=0.4, alpha=0.8)
        axes[1].plot(logl[:, k], lw=0.4, alpha=0.8)
    axes[0].set_ylabel(_label('dm2_31'))
    axes[1].set_ylabel(r'$\log L = -\chi^2/2$')
    axes[1].set_xlabel('sweep')
    for a in axes:
        a.grid(alpha=0.3)
    path = os.path.join(cli.outdir, f"{pre}_trace.png")
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
