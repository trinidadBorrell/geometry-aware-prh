# exp-011: geometry estimators on synthetic manifolds (WP0)

**Status:** running · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-011-synthetic-geometry/`

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

Pending.
