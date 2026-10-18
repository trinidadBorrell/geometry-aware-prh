# exp-013: intrinsic vs extrinsic curvature from neighbourhood moments

**Status:** done (v2) · **Owner:** oddharak · **Outputs:** `/workspace/results/oddharak/exp-013-curvature-moments/`

Part of WP0 (estimator validation) and WP5 (invariants) of the
[proposal](../proposal-geometry-aware-prh.md). Follows exp-011, where Ollivier-Ricci curvature had
the wrong sign and local PCA could not see curvature at our IDs.

## Why this experiment

### The question behind it

exp-010 found that LLM representations need 10-25x more principal components than their
intrinsic dimension (PCA90 / ID = 10-25) at every layer, and ViTs reach 7-26x in late layers. We
read that as "the data is not flat", and it is the main reason to doubt metrics built on
straight-line distance. But "not flat" has two very different meanings:

- **Extrinsic curvature**: how the manifold sits in the ambient space. A sheet of paper rolled
  into a tube is extrinsically curved; its straight-line (chord) distances differ from distances
  along the sheet, and it occupies more linear directions than its dimension.
- **Intrinsic curvature**: what an inhabitant of the manifold can measure without leaving it:
  the angles of triangles, the area of discs, how fast nearby geodesics diverge. A rolled sheet
  is intrinsically *flat*: unroll it and nothing is stretched. A sphere is intrinsically curved:
  no map of the Earth is distance-preserving.

Gauss's *Theorema Egregium* says the intrinsic curvature is unchanged by any bending that does not
stretch (any isometry). So if two models build the "same" manifold but embed it differently
(rolled differently, wound through different directions), their intrinsic curvature agrees and
their extrinsic curvature does not. That is exactly the distinction PRH-type questions need:

- Linear / Euclidean metrics (CKA, Procrustes, and to a lesser degree Euclidean kNN) see the
  embedding, i.e. extrinsic structure.
- A "shared model of reality" is a statement about intrinsic structure: which samples are near
  each other *along* the representation, and how the space curves.

If the LLM nonlinearity (PCA90 >> ID) is mostly extrinsic winding of an intrinsically flat (or
mildly curved) manifold, then the disagreement between linear and local metrics, and between
PRH and ARH, is a measurement artefact of the kind H1 in the proposal predicts. If models differ
in *intrinsic* curvature, that is an invariant no choice of ruler can remove (H2).

### Why the earlier tools could not answer it

exp-011 showed:
- Ollivier-Ricci curvature on kNN graphs is dominated by a high-dimensional negative bias and
  ranks the sphere below flat space.
- Local PCA cannot see a sphere's curvature: a d-sphere lives in d + 1 dimensions, so bending
  adds at most one direction, invisible to a 90% variance count. It also cannot tell an
  intrinsically flat manifold that winds through many directions from an intrinsically curved
  one.
- Graph geodesics are off by 50-150% at ID 20-40 with n = 10k, so methods built on them
  (Isomap, the geodesic-ball volume growth of Hickok & Blumberg 2023, which, if I read it
  correctly, estimates distances on a graph) inherit that error.

A local quadratic description does not need geodesics or graphs, only the positions of a
point's neighbours.

## Method: second fundamental form from two moments

Near a point p, a d-dimensional manifold M in R^D is the graph of a quadratic map. In tangent
coordinates u (the first-order, flat part) it reads

$$x = p + u + \tfrac12 \, \mathrm{II}(u,u) + O(|u|^3),$$

where II, the **second fundamental form**, is a symmetric bilinear map from tangent vectors to
normal vectors. It is the curvature of M as seen from R^D. Two contractions of it matter:

- the **mean-curvature vector** H = sum_i II(e_i, e_i) (the trace);
- the squared norm |II|^2 = sum_ij |II(e_i, e_j)|^2.

The **Gauss equation** turns these extrinsic quantities into the intrinsic **scalar curvature**:

$$S = |H|^2 - |\mathrm{II}|^2 .$$

Checks: the unit sphere S^d has II(u,v) = <u,v> n (one normal direction), so |H|^2 = d^2,
|II|^2 = d and S = d(d-1), the textbook value; a plane or a rolled sheet has S = 0.

**Estimating II without fitting it.** A direct quadratic regression needs d(d+1)/2 coefficients
per normal direction (210 at ID 20, 820 at ID 40) and thousands of neighbours. Instead we use
two moments of the neighbourhood in the normal space, which need only averages. If the tangent
offsets u of the k nearest neighbours are spread isotropically (for a uniform ball of radius r:
E[u_i u_j] = r^2/(d+2) delta_ij and E[u_i u_j u_k u_l] = r^4/((d+2)(d+4)) (three delta pairs)),
then with T = E|u|^2 the variance in the tangent space:

- **First moment.** The neighbourhood's centroid sits off p in the normal direction by
  m = 1/2 E[II(u,u)] = 1/2 (T/d) H. A sphere's neighbours all lie "below" p, so the centroid is
  displaced toward the centre; a rolled sheet's centroid is displaced toward the roll's axis.
- **Second moment.** The neighbours spread in the normal space with variance
  V_N = 1/4 (T/d)^2 [ (d+2)/(d+4) (|H|^2 + 2|II|^2) - |H|^2 ].

Both grow like T^2, i.e. like r^4. Noise does not: noise in p adds a constant to |m|^2, noise in
the neighbours adds a constant to V_N. So over several neighbourhood sizes k we regress
|m|^2 = a + b T^2 and V_N = c + e T^2 and read the curvature from the slopes:

- |H|^2 = 4 d^2 b
- |II|^2 = [ (4 d^2 e + |H|^2)(d+4)/(d+2) - |H|^2 ] / 2
- S = |H|^2 - |II|^2.

The tangent space at each point is the top-d principal directions of its neighbourhood. We
report S / (d(d-1)), which equals 1/R^2 for a sphere of radius R (1 for the unit sphere, 0 for
anything intrinsically flat), and |H| / d, the mean extrinsic curvature per direction.

The relation |m| proportional to |H| is the discrete form of a classical identity: the
Laplace-Beltrami operator applied to the embedding coordinates gives the mean-curvature vector,
Delta_M x = H, which is also what graph-Laplacian methods converge to. Sritharan, Wang & Hormoz
(PNAS 2021) estimate curvature through II and the Gauss equation with a quadratic fit on
single-cell data; the moment version here trades their full fit for robustness at high ID.

**Finite-sample correction of the first moment.** The centroid of k neighbours is itself
noisy: its normal offset is the mean of k normal displacements, so |m|^2 is overestimated by
V_N / k (the variance of a mean). That term shrinks as k grows and, left in, it biases the slope
b, and so |H|, downward (run v1 below had this bias). We subtract V_N / (k-1) per neighbourhood
and use unbiased 1/(k-1) variances.

**Choosing d from the data.** The estimator needs the tangent dimension d, and it is very
sensitive to underestimating it. If d is too small by m, m true tangent directions are counted as
normal; their variance grows like T (like r^2), not like T^2, and the regression on T^2 returns
large values of either sign. TwoNN gives 0.7-0.9 of the true ID at ID 20-40 (exp-011), so it
cannot supply d. Instead we use the scaling itself, the idea behind multiscale SVD (Little,
Maggioni & Rosasco): across scales, tangent variance grows like r^2, curvature-driven normal
variance like r^4, noise not at all. For each candidate d (0.7x to 2x the TwoNN ID) we fit
V_N = c + a T + e T^2 over the neighbourhood sizes; a d is roughly the number of tangent
directions that leaked into the normal part. We take the smallest d with a d < 0.5, i.e. less than
half a leaked direction. Overestimating d is much less harmful: extra "tangent" directions only
remove a little normal variance (checked in the d-sensitivity figure).

**Pooled estimate and scale profile.** The expansion holds only while neighbourhoods are small
compared with the curvature radius, and exp-011 showed that at ID 20-40 even 10 neighbours are
far apart. So we report two things: a pooled estimate (regression over the three smallest k)
and a **scale profile**, the estimate from each pair of consecutive k (10, 20, 40, ..., 640).
Where the expansion holds the profile is flat; where it drifts, higher-order terms or density
effects take over.

**Assumptions to keep in mind.** Uniform density and isotropic neighbourhoods (kNN in a region
of varying density shifts the centroid along the tangent space, which is projected out, and
skews the second moment, which is not); d known (we use both the true ID and the TwoNN estimate,
which is 0.7-0.9 of the truth at ID 20-40, exp-011); a single curvature summary per dataset
(curvature that varies over the manifold is averaged).

## Test manifolds

The exp-011 suite (flat, sphere, Swiss roll, thick tree, clusters) plus two manifolds that are
curved through *many* ambient directions, built to reproduce PCA90 >> ID with known intrinsic
curvature:

- **Flat torus by Fourier features (`torus_rff`).** theta uniform in [0, 2 pi)^d; each of ~256
  random integer frequency vectors w_j gives two coordinates a_j (cos <w_j, theta>,
  sin <w_j, theta>), with a_j = |w_j|^(-1/2). The pulled-back metric sum_j a_j^2 w_j w_j^T is the
  same at every point, so the manifold is **intrinsically flat** (S = 0) while it winds through
  ~500 dimensions (PCA90 / ID about 21 at ID 20, like an LLM layer).
- **Veronese sphere (`veronese`).** The unit sphere mapped by z -> z z^T / sqrt(2) into symmetric
  matrices. The map preserves the sphere's metric locally (tested), so the **intrinsic curvature
  is that of the unit sphere** (S / d(d-1) = 1), but the embedding uses (d+1)(d+2)/2 dimensions
  (PCA90 / ID about 10 at ID 20). It identifies z with -z (the projective space RP^d), so it is
  also a 2-to-1 quotient of the sphere: locally the same, globally different.

These two give clean controls for the whole programme:
- sphere vs Veronese: same intrinsic geometry, different embedding and global topology;
- flat vs Swiss roll vs flat torus: intrinsically flat, increasingly wound embeddings.

## Grid

Synthetic: 7 manifolds x ID 2, 5, 20, 40 x n = 1k, 10k, 100k x noise 0 / 0.1 (as exp-011),
random isometric embedding into 1024 dimensions; 300 random centres, k = 10, 20, ..., 640;
the estimate with the true ID and with the TwoNN ID. Global PCA90 and participation ratio.

Models: the 8 models with 10240 cached WIT-1M samples (BLOOMZ 560m, 1b1, 1b7; ViT IN21k tiny and
small, MAE base, DINOv2 small, CLIP base), every layer, PRH features (q = 0.95 clamp, l2 norm),
d = TwoNN ID (rounded). Note: l2 normalisation puts every point on the unit sphere of R^D, which
by itself contributes positive intrinsic curvature to any manifold spread over many directions of
that sphere; compare layers and models with each other, not with zero.

## Hypotheses

- The estimator recovers S / d(d-1) = 0 for flat, Swiss roll and flat torus and 1 for sphere and
  Veronese at ID 2-5, and keeps the 0 vs 1 ordering at ID 20-40 for n >= 10k (falsified if the
  intrinsically flat torus comes out as curved as the sphere).
- More samples help (the expansion needs small neighbourhoods), slowly.
- Model layers with PCA90 / ID >> 1 have much lower intrinsic curvature than their linear
  dimension suggests (mostly extrinsic winding).

## Commands

```bash
ssh root@<ip> -p <port> 'BRANCH=oddharak bash -s' < infra/runpod/setup_pod.sh
ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && AUTO_TERMINATE=1 nohup bash infra/runpod/run_exp013.sh > /workspace/results/oddharak/exp-013.log 2>&1 < /dev/null &"
uv run python experiments/exp-013-curvature-moments/plot.py --root results/exp-013-curvature-moments/v2
```

## Results

Two runs on RTX PRO 4500 pods (28-32 vCPU, used as CPU machines), 2026-10-18. Outputs in
`/workspace/results/oddharak/exp-013-curvature-moments/` (v1 at the top level, v2 in `v2/`),
copied to `results/exp-013-curvature-moments/{v1,v2}/` with `summary.txt` and the figures.
- **v1** (commit `7e95c33`, ~25 min): 168 synthetic sets (ID 2-40, n = 1k-100k) and the 8
  models, d = true ID or TwoNN ID. Had the centroid bias described in the method.
- **v2** (commit `431abba`, ~50 min): bias corrected, data-driven d, S for every candidate d;
  84 synthetic sets (ID 5, 20, 40 x n = 10k, 100k x noise 0, 0.1) and the 8 models (135 layers).
  The numbers below are v2 unless stated.

### 1. With the true ID, the estimator separates intrinsically flat from curved, and is exact at low ID

S / d(d-1), pooled over the three smallest usable k, noise 0.1 (true value 0 for flat, Swiss,
torus, clusters; 1 for sphere and Veronese):

| manifold | ID 5, n = 10k / 100k | ID 20 | ID 40 |
|---|---|---|---|
| flat | -0.01 / -0.03 | -0.03 / -0.03 | -0.04 / -0.04 |
| Swiss roll | +0.01 / +0.00 | +0.01 / +0.01 | +0.01 / +0.01 |
| flat torus (PCA90/ID 10-83) | +0.08 / +0.03 | +0.10 / +0.13 | +0.10 / +0.12 |
| sphere | +1.16 / +1.03 | +2.70 / +2.00 | +5.57 / +3.66 |
| Veronese sphere | +1.44 / +1.07 | -1.23 / -1.76 | +1.34 / +0.73 |

- At ID 2-5 (and ID 2 in v1) the estimate converges to the truth as n grows: sphere 1.16 -> 1.03,
  Veronese 1.44 -> 1.07.
- At every ID, the intrinsically flat manifolds give |S| <= 0.13, however they are embedded: the
  flat torus winds through ~500 dimensions with PCA90 / ID of 10-83 and still comes out flat.
- At ID 20-40 the curved manifolds come out clearly non-zero, but the size is off by up to 5x
  and the Veronese sphere even has the wrong sign at ID 20. Neighbourhoods at n <= 100k are not
  small compared with the curvature radius (exp-011: the nearest neighbour on S^20 is ~39 degrees
  away), so higher-order terms dominate. More samples help only slowly (sphere ID 20: 2.70 at 10k,
  2.00 at 100k).

### 2. d cannot be determined from the data at ID >= 20, and errors in d dominate

- **TwoNN** is 0.7-0.9 of the true ID for the ordinary manifolds and wildly *over* for the wound
  torus (150 at true ID 20 and 40): at the sampling scale, a manifold that oscillates through
  hundreds of directions looks high-dimensional. Intrinsic-dimension estimates of strongly
  wound manifolds are therefore not trustworthy either.
- **The scaling rule** (smallest d with less than half a leaked tangent direction) recovers d at
  ID 5 but fails at ID 20-40 (it picks 11-35 for ID 20 and ~20 for ID 40): with r^2 and r^4
  growth over only a factor of ~2 in radius, the linear and quadratic terms are not separable.
- **Sensitivity** (`fig5_d_sensitivity`, ID 20): underestimating d by 10-40% gives flat manifolds
  a spurious S of +1 to +3, because leaked tangent variance grows like T. For d at or above the
  true ID, the flat manifolds sit at 0 for every d; the sphere settles at a constant -0.7 and the
  Veronese drifts from -1 to +2.4. So the estimate is a **one-sided flatness test**: an
  intrinsically flat manifold gives S ~ 0 for all d >= ID; a non-zero S that is stable over a wide
  range of d >= ID says "not flat", without a reliable size or sign.

### 3. Models: LLM layers and late ViT layers fail the flatness test; early ViT layers are out of range

S / d(d-1) on PRH features (n = 10240) over the candidate range 0.7x-2x the TwoNN ID:

| layers | S over the d range | reading |
|---|---|---|
| BLOOMZ layer 0 (mean-pooled embeddings) | +3.4 to +23.6, falling with d | far from flat |
| BLOOMZ middle and last layers | +2.9 to +5.9, smooth, nearly constant in d | not flat |
| ViT last block (IN21k, DINOv2, CLIP) | +1.8 to +6.2, nearly constant in d | not flat |
| ViT first and middle blocks, all of MAE | +4 to -2900, erratic, changing sign with d | estimator out of range |

- In the map (`fig3_map`), LLM layers sit at PCA90 / d of 15-45, the same range as the flat
  torus, but at S ~ 4-8 instead of 0. Their large linear dimension is therefore not *only* the
  winding of an intrinsically flat sheet.
- Early and middle ViT blocks are nearly linear (PCA90 4-88, about the ID or below): there is
  almost no variance outside the tangent directions, the moments are dominated by density
  variation and clusters, and the estimate is meaningless (MAE stays out of range at every depth).
- Caveat on the positive values: PRH features are l2-normalised, so they lie on the unit sphere of
  R^D, and a manifold spread over many directions of that sphere inherits positive curvature
  from it (S / d(d-1) = 1 for a great subsphere). Together with the upward bias seen on the
  synthetic sphere, the magnitudes (+2 to +6) cannot be read as a curvature radius. The robust
  statement is the one-sided one: not consistent with an intrinsically flat manifold.

### What this means

- **PCA90 >> ID does not by itself mean intrinsic curvature.** The flat torus reproduces the
  LLM pattern of exp-010 (PCA90 / ID 10-80) with zero intrinsic curvature. exp-010's
  "nonlinearity" could have been pure embedding; for LLM layers the flatness test says it is not
  only that.
- **Local intrinsic curvature is not measurable to useful precision at ID 20-40 with n <= 100k**,
  by this estimator or by the ones in exp-011. Model-to-model comparisons of curvature values
  (WP5) would compare biases. Curvature can serve only as the one-sided flatness check above.
- **The intrinsic/extrinsic split is still the right question for PRH**, but it has to be asked
  through comparisons that do not need local differential geometry: whether a smooth map aligns
  two models (WP7), Gromov-Wasserstein lower bounds on distances (WP5), and spectral comparisons
  (WP6). The flat torus vs the Swiss roll and the Veronese vs the sphere are ready-made positive
  and negative controls for those (same intrinsic geometry, different embeddings).
- On Hickok & Blumberg (2023): their scalar-curvature estimator uses the volume growth of geodesic
  balls, which (as far as I recall) are built from graph distances. exp-011 found graph geodesics
  off by 50-150% at ID 20-40, so I would test it on this synthetic suite before investing in it;
  the harness here makes that a small addition.
