# Launch

How to set up a machine, start the runs of `runs/`, watch them and restart them. The main study runs on a TPU v7
slice: 64 chips on 16 hosts, 4 chips and 8 JAX devices per host (each v7 chip is two devices of ~95 GiB), 128
devices in all. Every command, flag, file and environment variable named here exists in this repository; if one
does not, this document is wrong, so stop and report it.

## 0. Quick start

```bash
uv sync --extra tpu                          # on every host (on GH200: --extra cuda)
# data/tokens -> the token streams (a local copy on every host), ckpt -> a directory all hosts share
# runs/ is set for one node of 8 GPUs (section 5); the slice overrides the batch and the mesh
# on host i (0..15), all 16 at the same time (section 3 starts them from host 0 over ssh):
mkdir -p log/trunk_8k && TTT_COORDINATOR=<host 0>:8476 TTT_NUM_PROCESSES=16 TTT_PROCESS_ID=<i> \
    .venv/bin/python -u -m ttt.train --config runs/trunk_8k.json --batch_size 128 --grad_accum 1 --fsdp_size 0 \
    > log/trunk_8k/train_$(hostname).log 2>&1
```

The same command, run again after any interruption, resumes from the newest checkpoint. The next runs of the chain
use the same command with another `--config` ([config.md](config.md)).

## 1. Environment

- Python 3.12 (`.python-version`); `uv` installs it if the host has another one.
- `uv sync --extra tpu` on a TPU VM; `uv sync --extra cuda` on GH200 / H100 (CUDA 13). Never both: they conflict, and
  `uv.lock` pins every version, jax 0.11.2 with libtpu 0.0.48.1 on TPU.
- On H100 nodes the same environment also comes as a Docker image (section 5).
- The package is installed editable, so `python -m ttt.train` works from any directory. The tools in `tools/`
  import `src.ttt` and must run from the repository root.
- `JAX_PLATFORMS`: unset or `tpu` on TPU, `cuda` on GPU, `cpu` for the tests.
- `TTT_ATTENTION` (default `auto`): Pallas flash attention on TPU, cuDNN on GPU, XLA elsewhere.
- `TTT_SUFFIX_LAYER_REMAT` (default `1`): `0` keeps the suffix layers' activations instead of recomputing them,
  4.5 % faster for about 10 GiB more per device (GH200, 1B at 8K, 2 sequences per GPU). The plain-TTT runs on GH200
  used `0`, the method's run `1`.
- `JAX_COMPILATION_CACHE_DIR` defaults to `~/.jax_cache`, so a restart skips most of the compilation.
- `WANDB_API_KEY`: only when `wandb_project` is set. Only process 0 logs to W&B.

## 2. Before a launch

1. **Code.** The same commit on every host: rsync the repository from host 0 without `.venv`, `.git`, `log` and the
   data (as the v4 pod's launcher did), or check out the commit on every host. Without `.git` on a host, set
   `TTT_GIT_COMMIT` (and `TTT_GIT_BRANCH`, `TTT_GIT_DIRTY=0`) so the checkpoints record the code.
2. **Environment.** `uv sync --extra tpu` on every host.
3. **Data.** `data/tokens` on every host, a local copy, never through gcsfuse ([data.md](data.md), section 6).
4. **Checkpoints.** `ckpt` must be ONE directory that every host reads and writes, for example a GCS bucket mounted
   with gcsfuse at the same path on all hosts.
   - Process 0 writes each checkpoint (`model.eqx`, `opt.eqx`, the configs, `state.json`, then `DONE`) and deletes
     old ones beyond `keep_checkpoints`. A 1B checkpoint is 15 GB.
   - On a restart every process lists the directory itself and reads the whole checkpoint, so every host must see
     the same newest `DONE`. Keep the gcsfuse metadata caches short: the v4 pod mounted with
     `--implicit-dirs --stat-cache-ttl 1h --type-cache-ttl 1h --max-conns-per-host 100`, and a 1 h stat cache can
     hide a new checkpoint from another host.
   - gcsfuse 0.35 on the v4 pod staged every written file in full on the local disk before uploading it: check the
     gcsfuse version and the free local disk.
   - A gcsfuse mount can die silently under heavy reads: the directory looks empty, then "Checkpoint file not found"
     on that host while the others wait at the barrier, or SIGBUS on a memory-mapped read. Before every launch check
     each host with `mountpoint -q` and a real read; never scan the bucket while a job runs.

## 3. On the TPU v7 slice

The v4 pod's method: it needs no TPU management API, only ssh from host 0 to the other hosts. Put the 16 hostnames
in `hosts.txt`, host 0 first, check that host 0 reaches every host by ssh without a password, then start the same
command on all of them at once, each with its own number:

```bash
C=$(head -1 hosts.txt):8476; i=0
for h in $(cat hosts.txt); do
  ssh $h "cd <repo> && mkdir -p log/trunk_8k && TTT_COORDINATOR=$C TTT_NUM_PROCESSES=16 TTT_PROCESS_ID=$i \
    setsid -f .venv/bin/python -u -m ttt.train --config runs/trunk_8k.json --batch_size 128 --grad_accum 1 --fsdp_size 0 \
    > log/trunk_8k/train_\$(hostname).log 2>&1 < /dev/null"
  i=$((i + 1))
done
```

- `setsid -f ... < /dev/null`: started with `nohup ... &`, the ssh session hung on the v4 pod.
- Write empty values as `--flag=`: a bare `""` is lost on the way through ssh.
- `TTT_COORDINATOR` makes every process call `jax.distributed.initialize` with this coordinator and these ids.
  Without the `TTT_*` variables, `NUM_PROCESSES=16` lets JAX find the hosts itself; that depends on what the
  platform provides, so use it only after it worked on the first day.
- `runs/` is set for one node of 8 GPUs (section 5): on the slice, at every length, `--batch_size 128 --grad_accum 1
  --fsdp_size 0`. `batch_size` must be a multiple of the 128 devices, and at most about 4 sequences of 8K fit on a
  device. There is no tensor or sequence parallelism, so 128 sequences per step is the minimum at every length.
- A host that does not start blocks all others in the initialisation. If one fails, stop all and start again.
- A cold compile of the 1B step took 15–26 min on one v4 node and 1–2 min on the pod.
- **Logs.** Every process prints `[ttt] process i/16 on <host>: main - training log follows` or `worker - quiet`.
  Process 0 is assigned by the TPU runtime, not by `TTT_PROCESS_ID`, and can be on any host (on the v4 pod it was
  w-2): find it with `grep -l "main - training log follows" log/<run>/train_*.log` and link or copy that file to
  `log/<run>/train_main.log` on that host. Process 0 also writes the evaluation files there (`log_dir`); copy that
  directory to the bucket from time to time.

## 4. On GH200 (SLURM, DeltaAI)

- One process per node, 4 GPUs each: `TTT_COORDINATOR=<first node's hsn0 IP>:<port>`,
  `TTT_NUM_PROCESSES=<nodes>`, `TTT_PROCESS_ID=$SLURM_PROCID`, `TTT_LOCAL_DEVICE_IDS=0,1,2,3`; on one node none of
  them.
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.94` and
  `XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_all_reduce_combine_threshold_bytes=268435456 --xla_gpu_all_gather_combine_threshold_bytes=268435456 --xla_gpu_reduce_scatter_combine_threshold_bytes=268435456"`;
  across nodes `NCCL_CROSS_NIC=1 NCCL_NET_GDR_LEVEL=3 NCCL_SOCKET_IFNAME=hsn0 FI_PROVIDER=cxi`.
- `TTT_HEARTBEAT_TIMEOUT_S=900`: a process frozen by the shared file system otherwise kills the run after 100 s.
- 128 sequences on one node: `--batch_size 16 --grad_accum 8 --fsdp_size 4`. The data order depends only on the step
  batch, so such a run can resume a TPU checkpoint.
- Stage the data, the environment and the checkpoint on the nodes' local NVMe (`/tmp`): the training must not
  depend on `/taiga`.

## 5. On H100 (Docker)

The image (`Dockerfile`) holds Python and the locked dependencies of `uv sync --extra cuda`. The repository is
mounted at run time: a code change needs no new image, a change of `uv.lock` does.

**The image.**

- Build it on any x86_64 machine with Docker 23 or newer (BuildKit) and access to docker.io, ghcr.io and PyPI:
  `docker build -t ttt-h100:latest .`. No GPU is needed: the build installs prebuilt wheels (every locked package
  has one for x86_64), and XLA compiles for the GPU at run time. An image runs only on the architecture it was built
  for: H100 hosts are x86_64, GH200 aarch64; on an Apple-silicon Mac add `--platform linux/amd64`.
- Copy it to every node: `docker save ttt-h100:latest | gzip > ttt-h100.tar.gz` once, then
  `docker load -i ttt-h100.tar.gz` on each node. The driver must support CUDA 13 (`nvidia-smi`: 580 or newer).

**`tools/docker_run.sh`.**

- `tools/docker_run.sh <command>` runs a command in it (no command: a shell) as the calling user, with all their
  groups: the files it writes are the user's, the checkpoints record the commit, and section 7 applies unchanged. It
  mounts the repository at `/workspace`, over it `ckpt`, `log` and `data/tokens` by their real paths (links to other
  file systems work; `data/tokens` read-only, so prepare data outside the container), and `~/.jax_cache`.
  `CODE_DIR` (the repository), `CKPT_DIR`, `LOG_DIR`, `DATA_DIR`, `CACHE_DIR` and `IMAGE_NAME` override the defaults.
- It passes `TTT_*`, `XLA_*`, `JAX_*`, `NCCL_*`, `WANDB_*` and `CUDA_VISIBLE_DEVICES` into the container, by name,
  except `JAX_COMPILATION_CACHE_DIR` (the image points it at the mounted cache); also `USER`, and `HOME=/tmp`. No
  other variable.
- The container sees every GPU of the node (`--gpus all`): Docker does not see a scheduler's allocation, so on a
  node shared with other jobs restrict it with `CUDA_VISIBLE_DEVICES`.
- `runs/` is set for one node of 8 GPUs: a step batch of 128 sequences as 16 × 8 at 8K and 8 × 16 at 32K and
  128K, with FSDP over the node's 8 GPUs (`fsdp_size 8`, [config.md](config.md)). On one node no `TTT_*` variable
  is needed, and its only process is process 0:

```bash
mkdir -p log/trunk_8k && tools/docker_run.sh python -u -m ttt.train --config runs/trunk_8k.json \
    > log/trunk_8k/train_main.log 2>&1
```

**Before the first run.**

1. On a GPU node:
   - `docker run --rm --gpus all ttt-h100:latest nvidia-smi`: Docker, the NVIDIA Container Toolkit and the driver
     (580 or newer).
   - `cat /sys/class/infiniband/*/ports/*/link_layer`: `InfiniBand` or `Ethernet` (RoCE) work as they are; no
     device means TCP sockets, and AWS EFA (`efa*`) or HPE Slingshot (`/dev/cxi*`) need an aws-ofi-nccl plugin that
     the image does not have.
   - `env | grep -E '^(NCCL|XLA|JAX)_'`: these variables are forwarded into the container. Unset those a site's
     modules set for another network (on DeltaAI `NCCL_NET_PLUGIN` names a host path and `NCCL_IB_DISABLE=1` turns
     InfiniBand off).
2. On every node: the repository at the same path and commit; `data/tokens` as a local copy (the 12 streams of
   `runs/trunk_8k.json` with their `windows_8192_4194304.npy`, `pg19_test` and `unigram_llama3.i64`: 221 GB);
   `ckpt` a directory that all nodes share. Commit first: every checkpoint records the commit and whether the tree
   was dirty.
3. On one node:
   - `tools/docker_run.sh python -c "import jax; print(jax.devices())"` lists 8 CUDA devices.
   - `tools/docker_run.sh python -u -m ttt.train --config runs/tiny_smoke.json`, run twice (section 6, step 2).
   - `tools/docker_run.sh python -u -m ttt.train --config runs/trunk_8k.json --probe_memory true`: the memory per
     GPU of the warm-up below, 2 sequences per GPU. The step donates its inputs, so the peak is `argument` plus
     `temp`, not the printed total: measured on GH200 and scaled to 8 GPUs, about 42 GiB, and 51 GiB with
     `TTT_SUFFIX_LAYER_REMAT=0`.

**The warm-up on 8 nodes** (`runs/trunk_8k_warmup.json`).

- The trunk's first 5000 steps, to the end of the warm-up: 128 × 1 on 64 GPUs (2 sequences per GPU) on a
  `dp 8 × fsdp 8` mesh, in `ckpt/trunk_8k`. Its last step, 4999, is checkpointed and evaluated.
- Its environment: `TTT_SUFFIX_LAYER_REMAT=0` (2 per GPU fits without the recompute, 6 % faster on GH200),
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.94` (JAX takes 75 % of the memory otherwise), `TTT_HEARTBEAT_TIMEOUT_S=900`,
  the `XLA_FLAGS` of section 4, and on the first launch `NCCL_DEBUG=INFO`: the log then names the network,
  `Using network IB` or `Using network Socket` (TCP, far slower).
- One container per node, all 8 started together, node `i` with `TTT_PROCESS_ID=i`. Node 0 is the coordinator and
  runs process 0: link its `train_<host>.log` to `log/trunk_8k/train_main.log`.

```bash
mkdir -p log/trunk_8k && TTT_COORDINATOR=<node 0>:8476 TTT_NUM_PROCESSES=8 TTT_PROCESS_ID=<i> \
    TTT_SUFFIX_LAYER_REMAT=0 TTT_HEARTBEAT_TIMEOUT_S=900 XLA_PYTHON_CLIENT_MEM_FRACTION=0.94 \
    tools/docker_run.sh python -u -m ttt.train --config runs/trunk_8k_warmup.json \
    > log/trunk_8k/train_$(hostname).log 2>&1
```

- Under SLURM, all 8 from one command:

```bash
srun --nodes=8 --ntasks-per-node=1 bash -c 'mkdir -p log/trunk_8k
  export TTT_COORDINATOR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1):8476
  export TTT_NUM_PROCESSES=$SLURM_NNODES TTT_PROCESS_ID=$SLURM_PROCID
  export TTT_SUFFIX_LAYER_REMAT=0 TTT_HEARTBEAT_TIMEOUT_S=900 XLA_PYTHON_CLIENT_MEM_FRACTION=0.94
  tools/docker_run.sh python -u -m ttt.train --config runs/trunk_8k_warmup.json \
      > log/trunk_8k/train_$(hostname).log 2>&1'
```

- Expect about 3.4 s per step (GH200, one micro-batch of 2 sequences per GPU) plus the all-reduces across nodes: 5–6
  hours for the 5000 steps with a 400 Gb/s NIC per GPU. The backward of the chunk scan all-reduces weight-sized
  gradients in every iteration, about 46 GiB per micro-batch, so a step much slower than about 4 s means the network
  is the limit.
- With one micro-batch per step the log has no `noise`; `tools/grad_noise.py` measures it on a checkpoint.

**After the warm-up.**

- `--config runs/trunk_8k.json` resumes `step_00004999` (`resume: auto`): on 8 nodes with `--batch_size 128
  --grad_accum 1` and the same environment, on one node with the file's 16 × 8. Only the step batch fixes the data
  order, so the micro-batch split may change at any restart.
- Sequences per GPU per micro-batch, measured on GH200 (4 GPUs, the trunk), memory per GPU scaled to an 80 GB H100
  with `fsdp 8`:

| per GPU | tokens/s per GPU | memory |
|:--|--:|--:|
| 1 | 3.8k | |
| 2 | 4.5k | 42 GiB |
| 2, `TTT_SUFFIX_LAYER_REMAT=0` | 4.8k | 51 GiB |
| 4 | 4.8k | 69 GiB |
| 4, `TTT_SUFFIX_LAYER_REMAT=0` | | about 88 GiB: does not fit |

  The method's arm is about 4 % slower. 3 per GPU without the recompute (5.0k) does not divide a step batch of 128.

## 6. First day on a new machine

On H100, the steps of section 5 ("Before the first run") come first; steps 2–4 run in `tools/docker_run.sh`.

1. On one host, confined to its own chips ([evaluation.md](evaluation.md), section 2):
   `JAX_PLATFORMS=tpu python tests/test_flash_attention.py`. The Pallas kernels are TPU-only, so the CPU tests
   cannot cover them.
2. `--config runs/tiny_smoke.json`: 20 steps of a tiny model, checkpoints at steps 9 and 19. The same command again
   resumes, finds nothing left to do and exits.
3. `--config runs/trunk_8k.json --probe_memory true` compiles the 1B step and prints XLA's memory estimate per
   device without training. The same for `runs/plain_32k.json` and `runs/plain_128k.json`, adding
   `--continue_from= --doc_windows false` (a random model and contiguous slices are enough for the estimate).
4. Start the trunk and replace the throughput estimate with the measured `s/step`: about 0.5–0.75M tokens/s for the
   slice, 1.4–2 s per step at 8K.

## 7. Watching and controlling a run

- A step line: `[ 123450] loss 2.4567  inner 3.1234  drift 1.23e-02  gnorm 0.456  lr 1.00e-03  1.523s/step  688k tok/s`.
  The method's runs add `write[...]`; gradient norms by group (`gn[e/p/s/w/o]`) appear when the norm exceeds 2.
- An evaluation line: `[eval 124000] loss ... no-ttt ... gain ...` ([evaluation.md](evaluation.md)).
- `touch log/TRIGGER_SAVE` on any one host: a checkpoint at the end of the current step. `touch log/TRIGGER_STOP`:
  a checkpoint, then a clean exit. `kill -USR1 <pid>` and `kill -USR2 <pid>` do the same. The directory is
  `$TTT_TRIGGER_DIR`, by default the repository's `log/`.
- After any stop or crash, run the same command on all hosts again.
- To kill a run: `pkill -9 -f 'python -u -m ttt[.]train'` in an ssh command of its own. A `pkill -f` whose pattern
  matches its own shell's command line (a launch in the same command) kills that shell first.
- Signs of trouble: `loss nan`; `gnorm` far above its usual range for many steps; a host's log that stops. The
  method's write amplifies gradients early in training, one reason it is switched on late.

## 8. Checklist before starting a run

| check | why |
|:--|:--|
| The same commit and `uv.lock` on every host | the hosts must compile the same program |
| `batch_size × grad_accum` is the intended step batch; `batch_size` is a multiple of the device count | otherwise the run refuses to start |
| The streams on local disks at `data_root` on every host, `num_tokens` as in [data.md](data.md) | memory-mapped streams over gcsfuse crashed the v4 pod |
| Every source has `windows_<seq_len>_4194304.npy` | `doc_windows` refuses to start without it; 128K must be built first |
| Only the `*_dc` web streams; `eval_sources` is PG-19 test | the other web streams contain PG-19 test books |
| `ckpt` is the same shared directory on every host, and every host sees the same newest `DONE` | every process resolves `resume: auto` itself |
| Every gcsfuse mount alive (`mountpoint -q` and a real read on each host); no bucket scan during a run | a dead mount stalls the barrier or crashes reads |
| `TTT_COORDINATOR`, `TTT_NUM_PROCESSES=16` and a distinct `TTT_PROCESS_ID` 0..15 on every host, all started together | one missing host blocks the rest |
| `continue_from` points to the previous run's last `step_*` with a `DONE` | the chain of runs |
| `steps`, `decay_start`, `decay_end` are global steps | the schedule continues across runs |
| The main log is `log/<run>/train_main.log` on process 0's host | `tools/round_compare.py` reads the mix from it |
