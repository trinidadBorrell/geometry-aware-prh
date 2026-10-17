# exp-011: geometry estimators on synthetic manifolds (WP0)

**Status:** stage 1 done · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-011-synthetic-geometry/`

WP0 of the [proposal](../proposal-geometry-aware-prh.md). Stage 1 of WP0: the geometry
estimators alone. Stage 2 (compatible vs incompatible *pairs*, scored by the WP3/WP5 alignment
measures) follows once those measures exist.

## Idea

Before trusting ID, geodesic or curvature estimates of model representations (ID 20-60, n = 1k-10k,
ambient 1k-5k), check them on manifolds whose geometry is known, at the same n, ID and ambient
dimension.

## Hypothesis

- ID (Levina-Bickel, TwoNN) is biased low at ID 40 but orders the manifolds correctly.
- Graph geodesics recover true geodesics well at n = 10k for ID 5, and degrade at ID 20-40
  (neighbourhoods too coarse); on the Swiss roll they beat Euclidean distances.
- Delta-hyperbolicity ranks tree < flat < sphere when computed from graph geodesics.
- Ollivier-Ricci curvature of kNN graphs is biased negative in high ID and gives negative values
  for flat clusters too (curvature confused with clustering). Falsified if its mean separates
  sphere > flat = swiss > tree at ID 20-40 and clusters are not negative.

## Method

- Manifolds (`geoprh.synthetic`): flat cube, sphere S^d, d-dimensional Swiss roll (intrinsically
  flat), thick binary tree, 10 Gaussian clusters. Intrinsic dimension d = 5, 20, 40; n = 1000,
  10000; random isometric embedding into 1024 dimensions; noise 0 or 0.1 x the median
  nearest-neighbour distance (isotropic, all 1024 dimensions).
- Estimators (`geoprh.geometry`, `geoprh.intrinsic_dim`):
  - ID: Levina-Bickel (k = 10..20), TwoNN, TwoNN by decimation (ID vs scale).
  - Local PCA dimension (PCs for 90% variance, skdim `lPCA` rule) of kNN neighbourhoods,
    k = 10..500: the flatness profile.
  - Isomap graph geodesics (k = 10, 30, 100) vs the true geodesic, from 200 sources: Spearman
    and median relative error, against Euclidean distances; connected components.
  - Gromov delta (four-point, 200k random quadruples) on 1000 points, relative to the diameter:
    from graph geodesics (k = 30), Euclidean distances and true geodesics.
  - Ollivier-Ricci curvature (GraphRicciCurvature, alpha = 0.5, uniform neighbour measure, exact
    OT) of the kNN graph, k = 10 and 30, on the first 2000 points. Linux only (the package
    needs the `fork` start method).

## Commands

```bash
ssh root@<ip> -p <port> 'BRANCH=oddharak bash -s' < infra/runpod/setup_pod.sh
ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && THREADS=16 AUTO_TERMINATE=1 nohup bash infra/runpod/run_exp011_012.sh > /workspace/results/oddharak/exp-011-012.log 2>&1 < /dev/null &"
```

## Results

Run 2026-10-17, commit `e91fabf`, RTX PRO 4500 pod (28 vCPU), ~40 min for the 60 datasets (run in
parallel with exp-012). Outputs in `/workspace/results/oddharak/exp-011-synthetic-geometry/`
(`synthetic.jsonl`), copied to `results/exp-011-synthetic-geometry/` with `summary.txt` and
fig1-fig4 from `plot.py`. Noise at 0.1 x the NN distance changes almost nothing, so the numbers
below are noise-free, n = 10000 unless stated.

**1. ID estimators: biased low, order preserved.** Levina-Bickel / TwoNN give 4.8-5.3 at ID 5,
15.8-18.0 at ID 20 and 27-30 at ID 40 (flat, sphere, Swiss roll), i.e. 0.7-0.9 of the truth at
our IDs, the same for all three manifolds. TwoNN by decimation is flat across scales for these
manifolds. Relative comparisons of ID (exp-010, exp-012) are therefore sound; absolute values
are underestimates, more so at higher ID. The thick tree is scale dependent (local ID 20, tree
ID 1): LB gives 11 and TwoNN 14 at ID 20.

**2. Graph geodesics fail at ID >= 20.** Median relative error of Isomap graph distances
against the true geodesic:

| manifold | ID 5, k = 10 / 30 / 100 | ID 20 | ID 40 |
|---|---|---|---|
| flat | 0.40 / 0.20 / 0.10 | 1.20 / 0.75 / 0.50 | 1.52 / 1.01 / 0.59 |
| sphere | 0.38 / 0.18 / 0.08 | 1.26 / 0.77 / 0.50 | 1.64 / 0.99 / 0.67 |
| Swiss roll | 0.37 / 0.22 / 0.13 | 0.94 / 0.56 / 0.36 | 1.22 / 0.79 / 0.48 |

At ID 5 the graph recovers geodesics and, on the Swiss roll with k = 10, beats Euclidean distance
(Spearman 0.68 vs 0.35). At ID 20-40, graph paths zig-zag and overestimate distances by 50-150%,
and on the Swiss roll they correlate *less* with the truth than Euclidean distance does (Spearman
0.25-0.33 vs 0.42-0.44). With n = 1000 it is the same or worse. Neighbourhoods at n = 10k are
far too coarse for kNN-graph geodesics in 20+ dimensions.

**3. Delta-hyperbolicity detects tree-like structure robustly, not positive curvature.**
delta_rel from graph geodesics (k = 30): tree 0.03 at every ID, against flat 0.47-0.67, sphere
0.62-0.73, Swiss 0.60-0.66, clusters 0.26-0.37. Flat vs sphere are not separated at ID 20-40
(0.63 vs 0.62 at ID 20), although the true-geodesic values are (0.38 vs 0.58): the graph errors
of point 2 inflate the flat value.

**4. Ollivier-Ricci curvature of the kNN graph has the wrong sign.** Mean edge curvature (k = 10):
flat -0.11 / -0.34 / -0.38 at ID 5 / 20 / 40, sphere -0.16 / -0.44 / -0.52 (*more* negative
than flat), thick tree +0.19 to +0.20 (positive), clusters -0.03 to -0.09. k = 30 shifts
everything up but keeps the order. So the estimator is dominated by a negative high-dimension
bias, ranks the positively curved sphere below flat space, and calls the tree positively curved
(dense neighbourhoods along its edges). Hypothesis 4 confirmed: ORC on kNN graphs is not usable as
a curvature estimate at our n and ID.

**5. The local PCA profile does not see curvature here.** At k = 500 the local dimension is
0.85-0.9 of the ID for flat, sphere and Swiss roll alike (17 / 18 / 18 at ID 20, 34 / 34 / 34 at
ID 40); at k = 10 it saturates at 7-8 whatever the ID (11 points cannot span more). Neighbourhoods
of up to 500 points are too small relative to the curvature radius to show bending.

**Implications for the plan (Gate A).**
- Pass: ID estimates (relative use), delta-hyperbolicity from graph geodesics as a test of
  hierarchical / tree-like structure.
- Fail: Ollivier-Ricci curvature (dropped from WP5), the local PCA profile as a curvature
  probe at k <= 500.
- Graph-geodesic rungs (R3, R4 of WP3) and geodesic-ball scales (WP4) are unreliable at ID 20-40
  with n = 10k: their results would mostly measure graph artefacts. WP3 should lead with the
  diffusion / heat-kernel rung, keep graph rungs as exploratory, and test whether larger n
  (100k) or larger k fixes them on these synthetic sets before trusting them on models.
- None of this measures pairs yet: stage 2 (compatible vs incompatible pairs) follows with the
  WP3 / WP5 measures.
