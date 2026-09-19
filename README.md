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

## Activation cache (never repeat a forward pass)

Activations live once on the team volume, per model and layer
(see `/workspace/README.md`):

```
/workspace/activations/<model>/layer_XX/<dataset>.pt          # [n_samples, dim] tensor
/workspace/activations/<model>/layer_XX/<dataset>.meta.json   # model, layer, dataset, seed, dtype, shape, ...
```

`<model>` is the HF/timm name with `/` replaced by `__`; `<dataset>` encodes dataset, subset,
pooling and caption index (e.g. `minhuh_prh_wit_1024_pool-avg_cid-0`). `geoprh.activation_cache`
bridges it to the upstream per-model feature files without touching upstream code:

```bash
uv run python -m geoprh.activation_cache ls   --modelset small                 # what is cached
uv run python -m geoprh.activation_cache pull --modelset small --layout platonic --dir <features dir>
uv run python -m geoprh.activation_cache push --modelset small --layout platonic --dir <features dir>
```

`pull` rebuilds upstream files from the cache, so the upstream extraction skips them; `push`
stores new ones (it never overwrites). Locally, set `ACTIVATIONS_DIR` to a local folder.

## Running on Runpod

Read `CLAUDE.md` for the team rules first (EU-RO-1 only, one pod at a time, state the price,
always **terminate**). Outputs go to `/workspace/results/<person>/`. Pods can be created with
`runpodctl` or by asking Claude (Runpod MCP tools). From Git Bash at the repo root:

1. **Create a GPU pod** from template `sw0wyyvy63` with volume `7mcpsbxer7` at `/workspace`
   (small models: `NVIDIA RTX PRO 4500 Blackwell`, $0.72/h):
   ```bash
   runpodctl pod create --name oddharak-prh-small --template-id sw0wyyvy63      --network-volume-id 7mcpsbxer7 --data-center-ids EU-RO-1      --gpu-id "NVIDIA RTX PRO 4500 Blackwell" --wait
   ```
   Wait until the pod lists a public port for `22/tcp` (`runpodctl pod get <id>`). That is the
   **direct** SSH address, `ssh root@<ip> -p <port>`. The `ssh.runpod.io` proxy only gives
   interactive shells (no commands, no scp).
2. **Look around.** `ssh root@<ip> -p <port>` opens a shell. Useful commands:
   `nvidia-smi`, `df -h /workspace`, `cat /workspace/README.md`, `ls /workspace/activations`.
   Non-interactive commands (`ssh host "cmd"`) do not see `HF_HOME`/`RUNPOD_*`, so start them
   with `. /etc/rp_environment;`.
3. **Push code and set up the environment:**
   ```bash
   infra/runpod/push_code.sh <ip> <port>     # tar of the working tree -> /root/geometry-aware-prh
   ssh root@<ip> -p <port> "bash /root/geometry-aware-prh/infra/runpod/setup_pod.sh"
   ```
4. **Launch detached** (survives SSH disconnects). The job pulls from the activation cache,
   extracts only what is missing, pushes new activations back, then computes alignment:
   ```bash
   ssh root@<ip> -p <port> "mkdir -p /workspace/results/oddharak && cd /root/geometry-aware-prh &&      AUTO_TERMINATE=1 nohup bash infra/runpod/run_prh_small.sh > /workspace/results/oddharak/prh_small.log 2>&1 &"
   ```
   With `AUTO_TERMINATE=1` a successful run removes its own pod; a failed run leaves it up.
5. **Monitor:** `ssh root@<ip> -p <port> "tail -f /workspace/results/oddharak/prh_small.log"`.
6. **Copy results home** while the pod is up:
   `scp -P <port> -r root@<ip>:/workspace/results/oddharak/prh_small/*/alignment results/pod/`
   (the volume persists after the pod is gone).
7. **Terminate** (`runpodctl remove pod <id>`, or ask Claude) and confirm with
   `runpodctl pod list` that nothing is left running.

## PRH paper figures (val set, CPU)

`geoprh.prh_alignment` reproduces platonic-rep's `measure_alignment` (q=0.95 clamp, l2 norm,
max over layer pairs) on CPU with per-layer caching; `tests/test_prh_alignment.py` checks it
against `metrics.AlignmentMetrics`. It reads activations from the shared cache.

```bash
ssh root@<ip> -p <port> 'BRANCH=oddharak bash -s' < infra/runpod/setup_pod.sh
ssh root@<ip> -p <port> "cd /root/geometry-aware-prh && WORKERS=16 nohup bash infra/runpod/run_prh_val.sh > /workspace/results/oddharak/prh_val.log 2>&1 &"
```

Outputs (`/workspace/results/oddharak/prh_val/`): `cross_modal.npz/json` (10 LLMs x 17 ViTs, 15
metrics), `vision_vision.npz/json`, `figures/` (PRH Fig. 3, 10, 12, 13, 14). Each worker needs
~3 GB RAM; size `WORKERS` by memory, not `nproc` (pods report host cores).
