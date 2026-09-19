# geometry-aware-prh

Do the Platonic and Aristotelian representation hypotheses disagree only because we
measure with the wrong geometry?

## Layout

```
Aristotelian/        upstream: Gröger et al. 2026, null-calibrated similarity (+ calibrated_similarity pkg)
platonic-rep/        upstream: Huh et al. 2024, original PRH feature extraction + alignment
experiments/         our experiment drivers (ruff-enforced)
  prh_small/         small-model PRH replication (both pipelines, same models)
infra/runpod/        push code to a pod, set it up, run a job that terminates its own pod
results/             local outputs (git-ignored)
```

Upstream code is kept close to the original. The only changes are:
- an added `small` modelset (in `platonic-rep/tasks.py` and `Aristotelian/aristotelian/prh/prh_models.py`)
  and the matching `--modelset` choice in the platonic-rep CLIs;
- `concrete_args=VIT_FX_CONCRETE_ARGS` on the ViT `create_feature_extractor` calls. timm>=1.0.26
  ViTs take `is_causal` in `forward()`, and without this hint torch.fx turns it into a Proxy
  that SDPA rejects. Features are unchanged (checked against forward hooks and timm 1.0.25);
- the path to platonic-rep in `Aristotelian/scripts/validation/prh_pipeline.py`.

Ruff skips those two directories.

## Setup (once)

Requires [uv](https://docs.astral.sh/uv/). One environment at the repo root serves both
codebases (uv workspace; `Aristotelian` is a member, so `uv run` works from any subfolder).

```bash
uv sync --all-groups          # Python 3.11, torch 2.8 + CUDA 12.8, dev tools
uv run pre-commit install     # ruff check/format on commit
```

Useful commands:

```bash
uv run ruff check . && uv run ruff format .                 # lint/format our code
cd Aristotelian && uv run pytest -q --no-cov                # upstream test suite (~6 min)
uv add <pkg>                                                # add a dependency (updates uv.lock)
```

## The two pipelines in one paragraph

Both pipelines do the same thing: embed the 1024 WIT image–caption pairs (`minhuh/prh`,
revision `wit_1024`), taking every layer of each language model (mean-pooled over tokens) and
every block of each ViT (CLS token). Then, for each (LLM, ViT) pair, they score every layer
pair with a similarity metric and report the **max over layer pairs**. platonic-rep reports
that raw max. Aristotelian also builds a permutation null for the *max statistic*, which gives
a p-value and a calibrated ("gated") score, with BH-FDR applied across the model grid.

## Running locally (small models, RTX 3060 6 GB is enough)

```bash
# 1. platonic-rep: extract features (downloads ~8 GB of weights to the HF cache)
cd platonic-rep
uv run python extract_features.py --dataset minhuh/prh --subset wit_1024 --modelset small \
    --modality language --pool avg --batch_size 8 --output_dir ../results/prh_small/platonic/features
uv run python extract_features.py --dataset minhuh/prh --subset wit_1024 --modelset small \
    --modality vision --pool cls --batch_size 32 --output_dir ../results/prh_small/platonic/features

# 2. platonic-rep: alignment (needs CUDA; metrics: mutual_knn, cka, cknna, unbiased_cka, svcca, ...)
uv run python measure_alignment.py --dataset minhuh/prh --subset wit_1024 --modelset small \
    --modality_x language --pool_x avg --modality_y vision --pool_y cls --metric mutual_knn \
    --input_dir ../results/prh_small/platonic/features --output_dir ../results/prh_small/platonic/alignment
cd ..

# 3. Aristotelian: extract (its own cache) + null-calibrated alignment
uv run python experiments/prh_small/run_aristotelian.py --device cuda --metrics mutual_knn,cka_lin

# 4. Side-by-side table
uv run python experiments/prh_small/compare.py
```

## Running on Runpod

Read `CLAUDE.md` for the team rules first (EU-RO-1 only, one pod at a time, state the price,
always **terminate**). Outputs go to `/workspace/results/<person>/`.

1. **Create a GPU pod** (with `runpodctl` or by asking Claude, which uses the Runpod MCP tools) (e.g. `NVIDIA L4` or `NVIDIA RTX PRO 4000 Blackwell` for small models)
   from template `sw0wyyvy63` with volume `7mcpsbxer7` mounted at `/workspace`:
   ```bash
   runpodctl pod create --name oddharak-prh-small --template-id sw0wyyvy63 \
     --network-volume-id 7mcpsbxer7 --data-center-ids EU-RO-1 --gpu-id "NVIDIA L4" --wait
   runpodctl pod get <pod-id>        # note the public IP and the port mapped to 22
   ```
2. **Push code and set up the environment** (from Git Bash, at the repo root):
   ```bash
   infra/runpod/push_code.sh <ip> <port>
   ssh -p <port> root@<ip> "bash /root/geometry-aware-prh/infra/runpod/setup_pod.sh"
   ```
3. **Launch detached.** The job terminates its own pod when it succeeds:
   ```bash
   ssh -p <port> root@<ip> "mkdir -p /workspace/results/oddharak && cd /root/geometry-aware-prh && \
     nohup bash infra/runpod/run_prh_small.sh > /workspace/results/oddharak/prh_small.log 2>&1 &"
   ```
4. **Monitor** with `ssh -p <port> root@<ip> tail -f /workspace/results/oddharak/prh_small.log`.
   After it finishes, check `runpodctl pod list` to confirm nothing is left running. If the job
   failed, the pod stays up for debugging, so terminate it yourself (`runpodctl remove pod <id>`, or ask Claude to use the Runpod MCP `delete-pod`).
5. **Get the results.** The pod is gone, but `/workspace` persists. Either start a cheap CPU pod
   (template `10tnena836`, `--compute-type CPU`) and `scp` from it, or run the analysis on
   that pod.
