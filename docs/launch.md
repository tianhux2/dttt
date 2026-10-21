# Launch

How to set up a machine, start the runs of `runs/`, watch them and restart them. Every command, flag, file and
environment variable named here exists in this repository; if one does not, this document is wrong, so stop and
report it.

## 0. Quick start

```bash
uv sync --extra cuda                         # on GPU; --extra tpu on a TPU VM
# data/tokens -> the token streams (a local copy on every host), ckpt -> a directory all hosts share
mkdir -p log/trunk_8k && .venv/bin/python -u -m ttt.train --config runs/trunk_8k.json \
    > log/trunk_8k/train_main.log 2>&1
```

This is one host with 8 devices, the split `runs/` is set for; other device counts override it (section 3), several
hosts start one process each (section 4). The same command, run again after any interruption, resumes from the
newest checkpoint. The next runs of the chain use the same command with another `--config`
([config.md](config.md)).

## 1. Environment

- Python 3.12 (`.python-version`); `uv` installs it if the host has another one.
- `uv sync --extra cuda` on GPU (CUDA 13), `uv sync --extra tpu` on a TPU VM. Never both: they conflict, and
  `uv.lock` pins every version, jax 0.11.2 with libtpu 0.0.48.1 on TPU. On GPU the same environment also comes as a
  Docker image (section 5).
- The package is installed editable, so `python -m ttt.train` works from any directory. The tools in `tools/`
  import `src.ttt` and must run from the repository root.
- `JAX_PLATFORMS`: unset or `tpu` on TPU, `cuda` on GPU, `cpu` for the tests.
- `TTT_ATTENTION` (default `auto`): Pallas flash attention on TPU, cuDNN on GPU, XLA elsewhere.
- `TTT_SUFFIX_LAYER_REMAT` (default `1`): `0` keeps the suffix layers' activations instead of recomputing them,
  about 5 % faster for about 10 GiB more per device at 8K with 2 sequences per device.
- On GPU: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.94` (JAX takes 75 % of the memory otherwise) and
  `XLA_FLAGS="--xla_gpu_enable_triton_gemm=false --xla_gpu_all_reduce_combine_threshold_bytes=268435456 --xla_gpu_all_gather_combine_threshold_bytes=268435456 --xla_gpu_reduce_scatter_combine_threshold_bytes=268435456"`.
- `TTT_HEARTBEAT_TIMEOUT_S` (default 100): how long a process may stop answering before every process is aborted.
  Set 900 when any file the run touches is on a shared file system, which can freeze a process for minutes.
- `JAX_COMPILATION_CACHE_DIR` defaults to `~/.jax_cache`, so a restart skips most of the compilation.
- `WANDB_API_KEY`: only when `wandb_project` is set. Only process 0 logs to W&B.

## 2. Before a launch

1. **Code.** The same commit on every host: check it out on every host, or rsync the repository from host 0 without
   `.venv`, `.git`, `log` and the data. Without `.git` on a host, set `TTT_GIT_COMMIT` (and `TTT_GIT_BRANCH`,
   `TTT_GIT_DIRTY=0`) so the checkpoints record the code.
2. **Environment.** `uv sync` with the platform's extra on every host.
3. **Data.** `data/tokens` on every host, a local copy ([data.md](data.md), section 6).
4. **Checkpoints.** `ckpt` must be ONE directory that every host reads and writes.
   - Process 0 writes each checkpoint (`model.eqx`, `opt.eqx`, the configs, `state.json`, then `DONE`) and deletes
     old ones beyond `keep_checkpoints`. A 1B checkpoint is 15 GB.
   - On a restart every process lists the directory itself and reads the whole checkpoint, so every host must see
     the same newest `DONE`: keep a network file system's metadata caches short.
   - A network mount (for example gcsfuse) can die silently under heavy reads: the directory looks empty, then
     "Checkpoint file not found" on that host while the others wait at the barrier, or SIGBUS on a memory-mapped
     read. Before every launch check each host with `mountpoint -q` and a real read; never scan the directory while
     a job runs.

## 3. Devices, batch and memory

- The step batch is 128 sequences at every length (256 in `runs/trunk_8k_b256.json`, a test): `batch_size ×
  grad_accum` is the step batch, and `batch_size` is a multiple of the device count. A device holds whole
  sequences (there is no tensor or sequence parallelism), so a run uses at most as many devices as its step batch.
- `runs/` splits it for 8 devices: `16 × 8` at 8K (2 sequences per device), `8 × 16` at 32K and 128K (1 per device),
  `fsdp_size 8`. On `D` devices with `s` sequences per device: `--batch_size <s·D> --grad_accum <128/(s·D)>`, and
  `--fsdp_size` the devices per host (FSDP inside each host, data parallel across them) or `0` (FSDP over all).
- The data order depends only on the step batch, so the split may change at any restart, also on another machine.
- `--probe_memory true` compiles the step and prints XLA's memory estimate per device without training. The step
  donates its inputs, so the peak is `argument` plus `temp`, not the printed total.
- Measured on GH200 (the trunk at 8K; memory with FSDP over 8 devices):

| sequences per device | tokens/s per device | memory per device |
|:--|--:|--:|
| 1 | 3.8k | |
| 2 | 4.5k | 42 GiB |
| 2, `TTT_SUFFIX_LAYER_REMAT=0` | 4.8k | 51 GiB |
| 4 | 4.8k | 69 GiB |
| 4, `TTT_SUFFIX_LAYER_REMAT=0` | | about 88 GiB |

  The method's arm is about 4 % slower. 3 per device does not divide a step batch of 128.
- The backward of the chunk scan all-reduces weight-sized gradients in every iteration, about 46 GiB per
  micro-batch: across hosts the network can be the limit. A step much slower than the single-host rate per device
  means it is.
- `runs/trunk_8k_10k.json`: the trunk's first 10000 steps as one micro-batch of 128 (no accumulation), for 64
  devices at 2 per device or 128 at 1. It writes the trunk's `ckpt_dir`, and `trunk_8k.json` resumes its last step,
  9999, on any device count. With one micro-batch per step the log has no `noise`; `tools/grad_noise.py` measures
  it on a checkpoint. It checkpoints every 2000 steps: launch it first with `--steps 5000`, which checkpoints
  `step_00004999`, then again without. `runs/method_5k_10k.json` has the same split; start it once
  `ckpt/trunk_8k/step_00004999` has its `DONE` (on other devices, or after the trunk's 10000 steps).

## 4. Several hosts

- One process per host, all started together, each with its own number: `TTT_COORDINATOR=<host 0>:<port>`,
  `TTT_NUM_PROCESSES=<hosts>`, `TTT_PROCESS_ID=<0 .. hosts-1>`. Every process then calls
  `jax.distributed.initialize` with them. A process uses all devices of its host (or those of
  `CUDA_VISIBLE_DEVICES`); `TTT_LOCAL_DEVICE_IDS` names them otherwise.
- Without the `TTT_*` variables, JAX finds the hosts itself (`NUM_PROCESSES`, `COORDINATOR_ADDRESS`, or a
  multi-host TPU slice). That depends on what the platform provides: use it only after it worked once.
- From host 0 over ssh, with the hostnames in `hosts.txt`, host 0 first, and passwordless ssh from host 0 to all:

```bash
C=$(head -1 hosts.txt):8476; n=$(wc -l < hosts.txt); i=0
for h in $(cat hosts.txt); do
  ssh $h "cd <repo> && mkdir -p log/trunk_8k && TTT_COORDINATOR=$C TTT_NUM_PROCESSES=$n TTT_PROCESS_ID=$i \
    setsid -f .venv/bin/python -u -m ttt.train --config runs/trunk_8k.json <split flags, section 3> \
    > log/trunk_8k/train_\$(hostname).log 2>&1 < /dev/null"
  i=$((i + 1))
done
```

  - `setsid -f ... < /dev/null`: started with `nohup ... &`, the ssh session can hang.
  - Write empty values as `--flag=`: a bare `""` is lost on the way through ssh.
- Under SLURM, all hosts from one command:

```bash
srun --nodes=<hosts> --ntasks-per-node=1 bash -c 'mkdir -p log/trunk_8k
  export TTT_COORDINATOR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1):8476
  export TTT_NUM_PROCESSES=$SLURM_NNODES TTT_PROCESS_ID=$SLURM_PROCID
  .venv/bin/python -u -m ttt.train --config runs/trunk_8k.json <split flags> > log/trunk_8k/train_$(hostname).log 2>&1'
```

- A host that does not start blocks all others in the initialisation. If one fails, stop all and start again.
- **GPU network.** On the first launch set `NCCL_DEBUG=INFO`: the log names the network, `Using network IB` or
  `Using network Socket` (TCP, far slower). Unset `NCCL_*` variables that a site's modules set for another network.
- **Logs.** Every process prints `[ttt] process i/N on <host>: main - training log follows` or `worker - quiet`.
  Process 0 writes the training log and the evaluation files (`log_dir`). On TPU the runtime assigns it, not
  `TTT_PROCESS_ID`, so it can be on any host: find it with `grep -l "main - training log follows"
  log/<run>/train_*.log` and link that file to `log/<run>/train_main.log` on its host.

## 5. Docker (GPU)

The image (`Dockerfile`) holds Python and the locked dependencies of `uv sync --extra cuda`. The repository is
mounted at run time: a code change needs no new image, a change of `uv.lock` does.

**The image.**

- Build it with Docker 23 or newer (BuildKit) and access to docker.io, ghcr.io and PyPI:
  `docker build -t ttt-h100:latest .` (the default `IMAGE_NAME` of `tools/docker_run.sh`). No GPU is needed: the
  build installs prebuilt wheels, and XLA compiles for the GPU at run time. An image runs only on the CPU
  architecture it was built on (x86_64 or aarch64); for another one add `--platform linux/amd64` or
  `linux/arm64`.
- Copy it to every host: `docker save ttt-h100:latest | gzip > ttt-h100.tar.gz` once, then
  `docker load -i ttt-h100.tar.gz` on each. The driver must support CUDA 13 (`nvidia-smi`: 580 or newer).

**`tools/docker_run.sh`.**

- `tools/docker_run.sh <command>` runs a command in it (no command: a shell) as the calling user, with all their
  groups: the files it writes are the user's, the checkpoints record the commit, and section 7 applies unchanged. It
  mounts the repository at `/workspace`, over it `ckpt`, `log` and `data/tokens` by their real paths (links to other
  file systems work; `data/tokens` read-only, so prepare data outside the container), and `~/.jax_cache`.
  `CODE_DIR` (the repository), `CKPT_DIR`, `LOG_DIR`, `DATA_DIR`, `CACHE_DIR` and `IMAGE_NAME` override the defaults.
- It passes `TTT_*`, `XLA_*`, `JAX_*`, `NCCL_*`, `WANDB_*` and `CUDA_VISIBLE_DEVICES` into the container, by name,
  except `JAX_COMPILATION_CACHE_DIR` (the image points it at the mounted cache); also `USER`, and `HOME=/tmp`. No
  other variable.
- The container sees every GPU of the host (`--gpus all`): Docker does not see a scheduler's allocation, so on a
  host shared with other jobs restrict it with `CUDA_VISIBLE_DEVICES`.
- A launch, with the variables of sections 1 and 4 set before it (on one host none of section 4):

```bash
mkdir -p log/trunk_8k && tools/docker_run.sh python -u -m ttt.train --config runs/trunk_8k.json \
    > log/trunk_8k/train_$(hostname).log 2>&1
```

**Before the first run**, on a GPU host:

- `docker run --rm --gpus all ttt-h100:latest nvidia-smi`: Docker, the NVIDIA Container Toolkit and the driver.
- `cat /sys/class/infiniband/*/ports/*/link_layer`: `InfiniBand` or `Ethernet` (RoCE) work as they are; no device
  means TCP sockets, and AWS EFA (`efa*`) or HPE Slingshot (`/dev/cxi*`) need an aws-ofi-nccl plugin that the image
  does not have.
- `env | grep -E '^(NCCL|XLA|JAX)_'`: these variables are forwarded into the container (section 4, GPU network).

## 6. First day on a new machine

On GPU in Docker, section 5 comes first and steps 2–4 run in `tools/docker_run.sh`.

1. The devices: `python -c "import jax; print(jax.devices())"`. On TPU also, on one host confined to its own chips
   ([evaluation.md](evaluation.md), section 2): `JAX_PLATFORMS=tpu python tests/test_flash_attention.py`. The Pallas
   kernels are TPU-only, so the CPU tests cannot cover them.
2. `--config runs/tiny_smoke.json`: 20 steps of a tiny model, checkpoints at steps 9 and 19. The same command again
   resumes, finds nothing left to do and exits.
3. `--config runs/trunk_8k.json --probe_memory true` with the split of section 3. The same for `runs/plain_32k.json`
   and `runs/plain_128k.json`, adding `--continue_from= --doc_windows false` (a random model and contiguous slices
   are enough for the estimate).
4. Start the trunk and note the measured `s/step`.

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
| `batch_size × grad_accum` is the run's step batch; `batch_size` is a multiple of the device count | otherwise the run refuses to start |
| The streams on local disks at `data_root` on every host, `num_tokens` as in [data.md](data.md) | memory-mapped streams over a network mount can crash with SIGBUS |
| Every source has `windows_<seq_len>_4194304.npy` | `doc_windows` refuses to start without it; 128K must be built first |
| Only the `*_dc` web streams; `eval_sources` is PG-19 test | the other web streams contain PG-19 test books |
| `ckpt` is the same shared directory on every host, and every host sees the same newest `DONE` | every process resolves `resume: auto` itself |
| Every network mount alive (`mountpoint -q` and a real read on each host); no directory scan during a run | a dead mount stalls the barrier or crashes reads |
| `TTT_COORDINATOR`, `TTT_NUM_PROCESSES` and a distinct `TTT_PROCESS_ID` on every host, all started together | one missing host blocks the rest |
| `continue_from` points to the previous run's last `step_*` with a `DONE` | the chain of runs |
| `steps`, `decay_start`, `decay_end` are global steps | the schedule continues across runs |
| The main log is `log/<run>/train_main.log` on process 0's host | `tools/round_compare.py` reads the mix from it |
