# Getting onto the Memphis MI355X cluster from this workstation

Written 2026-10-04 for the agent working in `dataset_b/`, whose `evidence/access.txt` records
`ssh: socket: Operation not permitted` and a Docker-socket denial. Both are properties of the sandbox the
command ran in, not of the cluster. Everything below was verified from this VM on 2026-10-04.

## 1. Why you were blocked, and the way around it

- **The Claude Code Bash sandbox denies outbound network sockets.** `ssh` fails with `socket: Operation not
  permitted` inside it. Run network commands with the sandbox off: pass `dangerouslyDisableSandbox: true` on the
  Bash call (the user will be asked to approve it once), or ask the user to run the command themselves with the
  `!` prefix in the prompt, e.g. `! ssh cluster hostname`.
- **There is no Docker daemon on this VM.** `docker image inspect …` can only run on a cluster node. Do not try to
  inspect images locally; inspect them inside a Slurm job (or, read-only, over SSH to a node, see §5).
- Nothing else was wrong: the SSH key, alias and sudo mapping all work (checked below).

## 2. How to connect

The alias is already in `~/.ssh/config`:

```
Host amd-mi355x-1 cluster
  HostName 172.30.160.204
  User matars
  IdentityFile ~/.ssh/id_rsa
  ForwardAgent yes
```

- `ssh cluster` lands on **`amd-mi355x-1`** as user **`matars`**. That node is also an XAI compute node, so do not
  run GPU work on it interactively; use Slurm.
- Everything on the cluster is done as the shared account **`dn`**: `sudo -u dn <command>` (passwordless, verified
  with `sudo -n -u dn whoami`). Files you create must be owned by `dn`, so prefix every remote command with it.
- From node 1, `dn` can SSH to the other nodes without a password: `sudo -u dn ssh amd-mi355x-7 …`.
- Smoke test (run with the sandbox off):

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 cluster 'hostname; whoami; sudo -n -u dn whoami; sinfo -h -o "%P %a %D %T"'
# expected: amd-mi355x-1 / matars / dn / partition table (XAI = 9 nodes amd-mi355x-[1-9], TEST = 4)
```

## 3. The shared tree, and the one rule that bites

- Only `/opt/shared` is NFS and visible from every node. The project checkout is
  `/opt/shared/frontier-qwen3-profiling/Frontier` (an rsync copy of this worktree, owned by `dn`); datasets live in
  `/opt/shared/frontier-qwen3-profiling/datasets/`.
- **`/opt/shared` is exported with `root_squash`.** A container runs as root, which the server maps to `nobody`, so
  every directory a container writes into must exist and be `chmod a+rwX` *before* `docker run`, and files the
  container writes come back owned by `nobody` (world-readable). The first pilot of the linear_op work died on a
  `mkdir` for exactly this reason (`02_run_record.md`). `run_b.sbatch` already does `mkdir` + `chmod a+rwx` for its
  run dir; keep that pattern for any other output path.
- Copy code to the cluster with rsync as `dn`, from this VM:

```bash
rsync -avc --rsync-path="sudo -u dn rsync" \
  profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/dataset_b/ \
  cluster:/opt/shared/frontier-qwen3-profiling/Frontier/profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/dataset_b/
```

  (`-c` compares by checksum; the cluster copy was always verified byte-identical before a run, md5 noted in the
  RUN.md. Do the same.)

## 4. Running things: Slurm, Docker, images

- Submit from the VM; the job runs Docker on the allocated node:

```bash
ssh cluster "sudo -u dn bash -c 'cd /opt/shared/frontier-qwen3-profiling/Frontier && \
  sbatch -w amd-mi355x-7 -t 00:30:00 -J <shortname>-b-pilot --export=ALL,B_TASK_DIR=...,B_PLAN=... \
  profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/dataset_b/run_b.sbatch'"
ssh cluster "squeue -u dn -o '%i %P %T %N %j %M'"          # watch
ssh cluster "sudo -u dn tail -f /opt/shared/frontier-qwen3-profiling/Frontier/data/profiling/sweep_work/logs/<job>.out"
```

- Partition `XAI`, 8× MI355X (gfx950) per node, `--gres=gpu:8`. Pin the node with `-w`: **Docker images are per
  node and differ between nodes.** Job names: short user prefix (the shared login is `dn`, so use your own tag).
- The reference sbatch for the whole linear_op/attention work is
  `profiling_knowledge/scripts/slurm/qwen3_mi355x_profiling.sbatch` (docker flags, aiter cache fallback, root-squash
  handling, `LINEAR_DOCKER_ENV` passthrough). The single-process probe pattern is
  `profiling_knowledge/scripts/slurm/qwen3_posprobe*.sbatch` (`srun` + `docker run` on one GPU).
- Docker flags that are known to work on these nodes:
  `--device=/dev/kfd --device=/dev/dri --group-add video --group-add 110 --ipc=host --cap-add=SYS_PTRACE
  --security-opt seccomp=unconfined --shm-size 8G` (plus `--network=host` if you serve).
- `rocm-smi --setperfdeterminism <MHz>` works for `dn` without sudo on compute nodes (used for the 1900 MHz lock);
  always reset to `auto` in an exit trap (an incident is recorded in `09_` §3b).

### Images present on 2026-10-04 (`docker image ls` per node, as `dn`)

| node | images |
|---|---|
| amd-mi355x-1 | `10.1.1.0:5050/dnis/sglang-rocm:20261001-7-ionic39.0.26.07.10.001-1_ubu22.04` |
| amd-mi355x-2 | `dnis/sglang-rocm:20261001-6`, `20261001-7` (local and registry tags), `dnis/vllm-rocm-mooncake:20260922` |
| amd-mi355x-7 | `lmsysorg/sglang:v0.5.11-rocm700-mi35x`, `v0.5.18`, `v0.5.19-rocm10`, `sglang-rocm:v0.5.17/v0.5.18` |
| amd-mi355x-8 | `dnis/vllm-rocm-mooncake:20260923-2/-3`, `20260924-51` |
| amd-mi355x-9 | `lmsysorg/sglang-rocm:v0.5.16…v0.5.20`, `lmsysorg/sglang:v0.5.17-rocm720-mi35x` |

The image your access notes name, `dnis/sglang-rocm:20260923-2-ionic39.0.26.07.10.001-1_ubu22.04`, is **not on
nodes 1, 2, 7, 8 or 9**. The internal registry is `10.1.1.0:5050`; either pull that tag onto your target node
(`sudo -u dn docker pull 10.1.1.0:5050/dnis/sglang-rocm:20260923-2-…`, ask the owner if the tag is gone) or pin
your inspection to a tag that exists (`20261001-7` on nodes 1 and 2). Record the immutable `sha256:` ID from
`docker image inspect` on the node you will run on, as `run_b.sbatch` requires.

### Queue right now

`squeue` shows XAI nodes 2, 7, 9 running other people's jobs (`yhadad_*`, `regeveyal_*`), two pending; six XAI nodes
idle. Check before pinning a node.

## 5. Read-only image inspection without a GPU job

`docker image inspect` and `docker run --entrypoint … <image> python -c …` on CPU-only paths do not need an
allocation, but the cluster convention is that anything touching the node goes through Slurm. For a quick
inspection either use a 5-minute `srun`:

```bash
ssh cluster "sudo -u dn srun -p XAI -w amd-mi355x-1 --gres=gpu:0 -t 00:05:00 \
  docker image inspect --format '{{.Id}} {{json .RepoDigests}}' 10.1.1.0:5050/dnis/sglang-rocm:20261001-7-ionic39.0.26.07.10.001-1_ubu22.04"
```

or run `inspect_image_b.py` the same way with its evidence directory under the shared tree.

## 6. Where the skills and the written knowledge are

- **Claude Code skills** (invoke with the Skill tool by name): `gpu-cluster:cluster-admin` loads the cluster
  reference (both clusters, Slurm conventions, node recovery), `gpu-cluster:grafana` for node metrics,
  `gpu-cluster:docker-lru-cleanup` if a node's root disk is full of images. The plugin is installed at
  `~/.claude/plugins/cache/gpu-team/gpu-cluster/1.4.0/` (`skills/*/SKILL.md`, plus `commands/` and `scripts/`).
  The Jerusalem skills (`setup-jerusalem`, `k8s-jerusalem`, `cluster-onboarding`) are for a different cluster;
  do not use them for this work.
- **Cluster facts learned the hard way**: `profiling_knowledge/qwen3_30b_a3b_mi355x_profiling/02_run_record.md`
  §"Cluster facts" (root_squash, per-node images, aiter cache, node pinning) and the README's §"How to reproduce
  or extend" in the same directory.
- **ROCm / image gotchas**: `profiling_knowledge/MI355X_ROCM_COOKBOOK.md` (note gotcha 4's 2026-09-17 correction:
  do not set `FRONTIER_PROFILING_FORCE_TORCH_ROPE_FALLBACK` on the sglang images).
- **Servers and checkouts outside Slurm**: `profiling_knowledge/INFRASTRUCTURE_MAP.md` (older `~/frontier-work`
  checkouts on nodes 1/3/8; the shared `/opt/shared` tree is the one to use now).
- **How a measurement run is documented**: every dataset directory carries a README/RUN.md with job id, node,
  image, commit, md5 (`run_position/README_dense_fixed_*.md` are the templates); the measurement contract is
  `10_measurement_contract.md`; `13_run_position_anomalies_root_cause.md` §3 shows the probe/sbatch pattern for
  single-GPU experiments with rocprofv3 and `AMD_LOG_LEVEL` traces.
- **Local analysis venv**: `/home/dn/.virtualenvs/qwen3-profiling` (pandas, numpy, python-pptx, PyMuPDF); plotting
  uses `/usr/bin/python3` (matplotlib). A second venv exists at `/home/dn/Frontier-qwen3-profiling/.venv`.
  Neither has plotly; `pip` can reach PyPI if you need it.

## 7. Checklist before your first GPU job

1. `ssh cluster 'sudo -n -u dn whoami'` prints `dn` (sandbox off).
2. rsync `dataset_b/` to the shared checkout; verify md5 of the files the job will run.
3. Pick a node whose `docker image ls` has your image; record its `sha256:` ID there.
4. Pre-create and `chmod a+rwX` every output directory under `/opt/shared/...` the container will write to.
5. `sbatch -w <node> -t <time> -J <tag>-… --export=ALL,…` from the VM; watch with `squeue -u dn`; read the `.out`.
6. Write the README/RUN.md next to the output before touching the data.
