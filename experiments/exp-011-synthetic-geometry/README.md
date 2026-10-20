# exp-011: geometry estimators on synthetic manifolds (WP0)

**Status:** stage 1 done, PersonaManifold replication done (addendum) · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-011-synthetic-geometry/`

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

## Addendum: PersonaManifold's curvature pipeline and density-decoupling test

**Status:** done · Run 2026-10-20, commit `b48b0b2`, RTX PRO 4500 pod (28 vCPU), ~25 min.
Outputs in `/workspace/results/oddharak/exp-011-synthetic-geometry/personamanifold/`, copied to
`results/exp-011-synthetic-geometry/personamanifold/` (`personamanifold_orc.jsonl`,
`personamanifold_summary.txt`, `fig6_pm_decoupling`).

### Question

PersonaManifold (Xu et al. 2026, Section 3.2, Section 4.2, Appendix E) reports that node Ollivier-Ricci curvature
of persona representations (ID d* ~ 15-23, n ~ 2k) is positive for professional personas
("dense clusters", read as sphere-like) and negative for fictional ones ("isolated
neighbourhoods", saddle-like), and that this is intrinsic geometry, not density, because the
type differences persist within density quintiles (Kruskal-Wallis p < 0.001) and density barely
correlates with curvature (Spearman rho = 0.08). Our Euclidean-graph ORC above (point 4) had the
wrong sign at ID 20-40. Does their exact construction behave differently, and does their
density test separate curvature from other structure?

### Method

`personamanifold_orc.py`, with their own package (`personamanifold`, pinned to commit
`33e181c`, the code reviewed in full):
- **Graph:** their `PersonaManifold.fit`: PCA to 99% of the variance; local metric
  g_i = sum_a v_a v_a^T / (lambda_a + 1e-4) on the top-d local PCA directions (whitening);
  edge weight (|delta|_{g_i} + |delta|_{g_j}) / 2; union-symmetrised kNN graph with k = 5d
  (Table A.1); d = the true ID (capped at the retained PCA dimension, which binds for the thick
  tree at ID 20 and 40, where it is 15 and 25, and for the clusters at ID 40, 39).
- **Curvature:** their `edge_curvature` convention for every edge: uniform measure on the graph
  neighbours, no mass on the node (idleness 0, their default), exact optimal transport with
  graph shortest-path costs. Computed with GraphRicciCurvature ("OTD", alpha 0, exp_power 0);
  `tests/test_personamanifold_orc.py` checks it equals their `edge_curvature` edge by edge (run on
  the pod: passed). Node curvature = mean over the node's own k kNN edges (Section 3.2).
- **Comparison:** the same union-kNN topology with plain Euclidean edge lengths.
- **Their Appendix E test:** density = 1 / distance to the k-th neighbour (k = 5d; also k = 10)
  in the PCA space; density quintiles; Kruskal-Wallis across types within each quintile;
  Spearman(density, node curvature).
- **Data:** n = 2000 (their scale), ID 5, 20, 40, random isometry into 1024 dimensions, noise 0.1
  x the NN distance. Scenarios: each exp-011 manifold alone; a *mixture* of four components
  (flat, sphere, tree, clusters; 500 points each, separate graph components) with the manifold
  as "type"; *grouped*: one flat cube where half the points are uniform ("spread") and half
  come in 100 tight groups of 10 ("grouped"), all intrinsically flat; *null*: a flat cube with
  random labels.

### Results

**1. Their construction gives the same answers as the Euclidean graph.** Mean node ORC, whitened
graph (Euclidean twin within 0.03-0.08 everywhere):

| manifold (true curvature) | ID 5, k = 25 | ID 20, k = 100 | ID 40, k = 200 |
|---|---|---|---|
| flat (0) | -0.056 | +0.172 | +0.265 |
| Swiss roll (0) | -0.076 | +0.175 | +0.263 |
| sphere (+) | -0.145 | +0.100 | +0.230 |
| thick tree (- at large scale) | +0.461 | +0.582 | +0.650 |
| clusters (0 within pieces) | +0.031 | +0.602 | +0.867 |

- The sign is set by k and the ID, not by the geometry: with k = 5d everything is negative at
  ID 5 and positive at ID 20-40 (exp-011 point 4, with k = 10-30, was negative throughout).
- The order is inverted relative to the truth at every ID: the sphere is always the *lowest*,
  below flat space; the tree and the clusters are the *highest*. Whitening changes nothing that
  matters.

**2. Density and curvature are strongly coupled, not weakly.** On a single, uniformly sampled
flat cube (the null scenario, no curvature anywhere) Spearman(density, node ORC) is -0.50 at ID 5
and -0.95 at ID 20 and 40: ORC is almost a deterministic function of the sampling fluctuations of
density. The sign of the coupling depends on the geometry (flat -0.95, sphere +0.32 or -0.31,
clusters +0.30 or +0.48 at ID 20 / 40), so in a mixture of regions it can partly cancel, and a
small overall rho (theirs: 0.08) does not show that curvature is independent of density.
(Separately, with n ~ 2000, rho = 0.08 has p ~ 3e-4 and rho = 0.11 has p ~ 1e-6, so "n.s." in
their Appendix E is only possible if the correlation was computed on far fewer points.)

**3. Their density-decoupling test passes for differences that are not curvature.**
(`fig6_pm_decoupling`, which redraws their Fig. E.1(b) on our scenarios):
- *grouped* (both types intrinsically flat): Kruskal-Wallis p < 0.001 in 3/5 quintiles at ID 5
  and **5/5 at ID 20 and 40**; at ID 20 and 40 spread points sit above grouped ones by 0.08-0.26
  in every quintile (at ID 5 the gap changes sign across quintiles). The difference comes from the grouping: edges between members of a tight group are
  very short, and kappa = 1 - W1 / d(x, y) divides by that short distance.
- *mixture*: p < 0.001 in every quintile (4/4, 4/4, 5/5), with the type order tree > clusters >
  flat > sphere, the reverse of the true curvature (sphere +, flat 0, tree -).
- *null*: no quintile significant (p = 0.05-1) at any ID, so the test is not trivially
  significant; it detects *some* structural difference between groups, but nothing in it ties
  that difference to curvature.

### What this means for the PersonaManifold claims

- "Professional personas tend toward positive curvature; fictional characters exhibit negative
  curvature" describes differences in node ORC that our synthetic data reproduces from grouping,
  cluster structure and density alone, with the sign set by k and the ID. With their k = 5d and
  d* ~ 20, flat data is already positive (+0.17), so positive values cannot be read as
  sphere-like, nor negative ones as saddle-like.
- Their density-matched test cannot certify that the heterogeneity is intrinsic geometry: it
  passes on intrinsically flat data that differs only in grouping, and on a mixture whose
  curvature order it reverses.
- Their public code does not include the node-curvature, density or persona-scale pipeline, so
  the paper's own numbers cannot be rerun; this addendum replicates the construction as specified
  in Section 3.2, Table A.1 and the released `curvature.py`.
- Their behavioural results (geodesic distances vs triplet consistency, steering quality) do not
  depend on ORC meaning curvature and are not addressed here.
- For this project: graph Ollivier-Ricci curvature is dropped (as in Gate A above); any curvature
  claim about model representations needs controls of the kind used here.
