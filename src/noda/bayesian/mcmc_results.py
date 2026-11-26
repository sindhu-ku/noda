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
    p.add_argument('--annotate', default='',
                   help='optional text drawn top-right of the dm2_31 '
                        'marginal (e.g. "w/ DYB and NuFit-LBL const.")')
    p.add_argument('--add-pull', default='',
                   help='comma list from {dyb,lbl}: reweight the (pull-less) '
                        'chain with these dm2_31 pulls posthumously via '
                        'importance weights w = exp(-chi2_pull/2). Exact — '
                        'quote the printed weighted ESS with the result')
    p.add_argument('--pdg-opt', default='',
                   help='PDG_opt (e.g. NuFitLBL) used to reconstruct the '
                        'per-ordering DYB pull centres when the npz has no '
                        'pull_meta (chains recorded before July 2026)')
    p.add_argument('--lbl-spec', default='',
                   help='"cNO,cIO,err[,errIO]" in 1e-3 eV^2 (cIO signed): '
                        'manual LBL pull centres when the npz has no '
                        'pull_meta (e.g. "2.517,-2.408,0.0265,0.0275")')
    p.add_argument('--bins', type=int, default=80,
                   help='number of bins per panel of the dm2_31 marginal '
                        '(use e.g. 200 for well-sampled JUNO-only chains)')
    p.add_argument('--profile-root', default='', nargs='*',
                   help='ROOT file(s) with frequentist dm2_31 chi2 '
                        'profile(s) to overlay on the marginal (e.g. the NO '
                        'and IO Postfit_* files; combine_profile output '
                        'also works)')
    p.add_argument('--profile-graphs', default='profiles/chi2_profile',
                   help='comma list of TGraph names read from EACH '
                        '--profile-root file (x = dm2_31 in 1e-3 eV^2 or '
                        'eV^2, signed by ordering; y = chi2 or delta-chi2)')
    return p.parse_args()


def _load_profiles(root_paths, graph_names):
    """Read frequentist chi2-profile TGraphs -> (x [1e-3 eV^2], dchi2).
    All graphs from all files are concatenated; the global minimum over
    all of them is subtracted, so NO/IO panels share one normalisation
    (like the posterior areas do)."""
    import uproot
    xs, ys = [], []
    for root_path in root_paths:
        with uproot.open(root_path) as f:
            for name in graph_names:
                g = f[name.strip()]
                xs.append(np.asarray(g.member('fX'), float))
                ys.append(np.asarray(g.member('fY'), float))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    if np.nanmax(np.abs(x)) < 0.1:          # graph stored in eV^2
        x = x * 1e3                          # -> 1e-3 eV^2 (chain units)
    order = np.argsort(x)
    x, y = x[order], y[order]
    return x, y - np.nanmin(y)


def credible(x, level=0.68, weights=None):
    if weights is None:
        lo = np.percentile(x, 50 * (1 - level))
        hi = np.percentile(x, 50 * (1 + level))
        return np.median(x), lo, hi
    o = np.argsort(x)
    cw = np.cumsum(weights[o])
    cw /= cw[-1]
    xq = lambda q: x[o][min(np.searchsorted(cw, q), len(x) - 1)]
    return xq(0.5), xq(0.5 * (1 - level)), xq(0.5 * (1 + level))


def pull_chi2(flat, names, j31, add, meta, pdg_opt, lbl_spec):
    """chi2 of the requested dm2_31 pulls at each sample, matching the
    fit-time convention (juno_fit pulls / minuit.lbl_pull_target)."""
    dm31 = flat[:, j31] * 1e-3               # Minuit 1e-3 units -> eV^2
    sign = dm31 > 0
    chi2 = np.zeros(len(flat))
    ops = (meta or {}).get('op')
    if ops is None and pdg_opt:
        from ..physics import nuosc
        ops = {}
        for lab, no in (('NO', True), ('IO', False)):
            nuosc.SetOscillationParameters(opt=pdg_opt, NO=no)
            ops[lab] = dict(nuosc.op_nom)
    if 'dyb' in add:
        if not ops:
            raise SystemExit("--add-pull dyb: npz has no pull_meta — pass "
                             "--pdg-opt (the config's PDG_opt, e.g. NuFitLBL)")
        for lab, msk in (('NO', sign), ('IO', ~sign)):
            c, e = ops[lab]['dm2_31_dyb'], ops[lab]['dm2_31_dyb_err']
            chi2[msk] += ((dm31[msk] - c) / e) ** 2
    if 'lbl' in add:
        if lbl_spec:
            vals = [float(v) * 1e-3 for v in lbl_spec.split(',')]
            if len(vals) == 3:
                (c_no, c_io, e_no), e_io = vals, vals[2]
            else:
                c_no, c_io, e_no, e_io = vals
        elif meta and meta.get('lbl', {}).get('dm2_31_lbl') is not None:
            L = meta['lbl']
            c_no, e_no = L['dm2_31_lbl'], L['dm2_31_lbl_err']
            if L.get('dm2_31_lbl_IO') is not None:
                c_io = -abs(L['dm2_31_lbl_IO'])
                e_io = L.get('dm2_31_lbl_err_IO') or e_no
            else:                    # legacy: project NO central via f_lbl
                from ..fit.minuit import f_lbl
                sp = meta['solar_pin']
                s13 = flat[~sign, names.index('sin2_th13')]
                c_io = f_lbl(c_no, sp['sin2_th12'], s13, sp['dm2_21'],
                             L['theta23'], L['dcp'], 'IO')
                e_io = e_no
        else:
            raise SystemExit("--add-pull lbl: npz has no pull_meta — pass "
                             '--lbl-spec "cNO,cIO,err" (1e-3 eV^2, cIO '
                             "signed, e.g. \"2.517,-2.408,0.0265\")")
        chi2[sign]  += ((dm31[sign] - c_no) / e_no) ** 2
        chi2[~sign] += ((dm31[~sign] - c_io) / e_io) ** 2
    return chi2


def _dm31_marginal(x31, path, both_mode, f_no, annotate, weights=None,
                   bins=80, profile=None):
    """dm2_31 marginal. Both orderings populated -> broken-axis style (as in
    the ReactorNuPWG Bayesian comparison): IO panel left (red), NO panel
    right (blue), common normalisation so the two areas are the posterior
    ordering probabilities. Optional per-sample importance weights."""
    if weights is None:
        weights = np.ones(len(x31))
    w_tot = weights.sum()
    m_io, m_no = x31 < 0, x31 > 0
    if both_mode and m_io.sum() >= 10 and m_no.sum() >= 10:
        fig, (axL, axR) = plt.subplots(
            1, 2, sharey=True, figsize=(9, 4.5),
            gridspec_kw={'wspace': 0.06})
        h_max = 0.0
        for ax, msk, color, lab in ((axL, m_io, '#c1554d', 'MCMC IO'),
                                    (axR, m_no, '#4c72b0', 'MCMC NO')):
            xs, ws = x31[msk], weights[msk]
            p_lo, p_hi = np.percentile(xs, [0.05, 99.95])
            pad = 0.25 * (p_hi - p_lo)
            edges = np.linspace(p_lo - pad, p_hi + pad, bins + 1)
            bw = edges[1] - edges[0]
            h, _, _ = ax.hist(xs, bins=edges, weights=ws / (w_tot * bw),
                              histtype='stepfilled', color=color, alpha=0.85,
                              edgecolor=color, label=lab)
            h_max = max(h_max, h.max())
            ax.set_xlim(edges[0], edges[-1])
            ax.set_xlabel(_label('dm2_31'), fontsize=13)
            ax.tick_params(direction='in', top=True)
        if profile is not None:
            px, pdc = profile
            like = np.exp(-0.5 * pdc)        # global-min subtracted ->
            like *= h_max / like.max()       # one scale for both panels
            for ax, msk in ((axL, px < 0), (axR, px > 0)):
                if msk.any():
                    ax.plot(px[msk], like[msk], 'k--', lw=1.4,
                            label=r'Frequentist $e^{-\Delta\chi^2/2}$')
        for ax, loc in ((axL, 'upper left'), (axR, 'upper right')):
            ax.legend(frameon=False, fontsize=11, loc=loc)
        axL.set_ylabel('Posterior Prob.', fontsize=13)
        axL.set_ylim(bottom=0)
        # axis-break marks
        axL.spines['right'].set_visible(False)
        axR.spines['left'].set_visible(False)
        axR.tick_params(left=False)
        d = 0.02
        kw = dict(transform=axL.transAxes, color='k', clip_on=False, lw=1)
        axL.plot((1 - d, 1 + d), (-d, +d), **kw)
        axL.plot((1 - d, 1 + d), (1 - d, 1 + d), **kw)
        kw['transform'] = axR.transAxes
        axR.plot((-d, +d), (-d, +d), **kw)
        axR.plot((-d, +d), (1 - d, 1 + d), **kw)
        if f_no is not None and 0 < f_no < 1:
            two_lnk = 2 * np.log(f_no / (1 - f_no))
            fig.text(0.5, 0.62,
                     "Bayes factor\n"
                     r"$2\ln\kappa = 2\ln\frac{P(NO)}{P(IO)}"
                     rf" = {two_lnk:.2f}$",
                     ha='center', va='center', fontsize=14)
        if annotate:
            fig.text(0.985, 0.97, annotate, ha='right', va='top', fontsize=12)
    else:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        h, edges, _ = ax.hist(x31, bins=max(bins, 300), density=True,
                              weights=weights,
                              histtype='stepfilled', alpha=0.7)
        if profile is not None:
            px, pdc = profile
            like = np.exp(-0.5 * pdc)
            msk = (px >= edges[0]) & (px <= edges[-1])
            if msk.any():
                ax.plot(px[msk], like[msk] * h.max() / like[msk].max(),
                        'k--', lw=1.4,
                        label=r'Frequentist $e^{-\Delta\chi^2/2}$')
                ax.legend(frameon=False, fontsize=11)
        ax.set_xlabel(_label('dm2_31'), fontsize=13)
        ax.set_ylabel('posterior density', fontsize=13)
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        if both_mode:
            ax.axvline(0, color='k', lw=0.5)
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def main():
    cli = parse()
    os.makedirs(cli.outdir, exist_ok=True)
    pre = cli.prefix or os.path.splitext(os.path.basename(cli.input[0]))[0]

    # ── load & merge (multiple files = independent chains / segments) ────
    names = j31 = both = pull_meta = None
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
        if pull_meta is None and 'pull_meta' in z:
            pull_meta = z['pull_meta'][0]
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

    # ── posthumous pull reweighting ──────────────────────────────────────
    wts, f_no_w = None, None
    if cli.add_pull:
        add = [s.strip() for s in cli.add_pull.split(',') if s.strip()]
        bad = set(add) - {'dyb', 'lbl'}
        if bad:
            raise SystemExit(f"--add-pull: unknown pull(s) {sorted(bad)}")
        chi2p = pull_chi2(flat, names, j31, add, pull_meta,
                          cli.pdg_opt, cli.lbl_spec)
        wts = np.exp(-0.5 * (chi2p - chi2p.min()))
        ess_k = wts.sum() ** 2 / (wts ** 2).sum()
        print(f"\n════ reweighted with {'+'.join(add)} pull(s) "
              f"(importance weights, exact) ════")
        print(f"weighted ESS (Kish) = {ess_k:,.0f} of {len(wts):,} samples "
              f"({100 * ess_k / len(wts):.1f}%)")
        if both:
            sgn = flat[:, j31] > 0
            f_no_w = float(wts[sgn].sum() / wts.sum())
            err = np.sqrt(max(f_no_w * (1 - f_no_w), 1e-12) / ess_k)
            print(f"P(NO|data) = {f_no_w:.4f} +- {err:.4f}    "
                  f"P(IO|data) = {1 - f_no_w:.4f}")
            if 0 < f_no_w < 1:
                bfw = f_no_w / (1 - f_no_w)
                e2 = 2 * err / max(f_no_w * (1 - f_no_w), 1e-12)
                print(f"posterior odds NO/IO = {bfw:.3g}   "
                      f"2 ln(B) = {2 * np.log(bfw):.2f} +- {e2:.2f}")
            for lab, msk in (('NO', sgn), ('IO', ~sgn)):
                if msk.sum() < 10:
                    continue
                med, lo, hi = credible(np.abs(flat[msk, j31]),
                                       weights=wts[msk])
                print(f"  {lab}: |dm2_31| = {med:.4f} {lo - med:+.4f} "
                      f"{hi - med:+.4f}")

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
    profile = None
    if cli.profile_root:
        try:
            profile = _load_profiles(cli.profile_root,
                                     cli.profile_graphs.split(','))
            print("overlaying frequentist profile from "
                  f"{', '.join(cli.profile_root)} "
                  f"({cli.profile_graphs}, {len(profile[0])} points)")
        except Exception as e:
            print(f"(profile overlay skipped: {e})")
    path = os.path.join(cli.outdir, f"{pre}_dm2_31_marginal.png")
    _dm31_marginal(x31, path, both, f_no if both else None, cli.annotate,
                   bins=cli.bins, profile=profile)
    print(f"wrote {path}")
    if wts is not None:
        path = os.path.join(cli.outdir, f"{pre}_dm2_31_marginal_reweighted.png")
        note = (cli.annotate + '  ' if cli.annotate else '') \
            + f"reweighted: +{'+'.join(add)} pull"
        _dm31_marginal(x31, path, both, f_no_w, note, weights=wts,
                       bins=cli.bins, profile=profile)
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
