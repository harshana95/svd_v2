# Polaris cheat sheet

ALCF Polaris uses **PBS**, not Slurm. Docs: [Running Jobs on Polaris](https://docs.alcf.anl.gov/polaris/running-jobs/).

## This project

| | |
|---|---|
| Login | `ssh polaris.alcf.anl.gov` |
| Account | `lighthouse-purdue` |
| Repo | `/lus/eagle/projects/lighthouse-purdue/harshana/svd_v2` |
| Filesystems | `home` + `eagle` |
| Node | 4× A100 40GB, 32 CPU cores, billed as a whole node |

```bash
cd /lus/eagle/projects/lighthouse-purdue/harshana/svd_v2
```

## Environment

Do this on a **compute node** (interactive or batch), not the login node.

```bash
source scripts/alcf/load_conda.sh

# optional project venv on top of ALCF conda
source .venv/alcf/bin/activate
```

Use **`source`**, not `./load_conda.sh`. Executing the file loads conda in a subshell, so `conda: command not found` when it returns. Do not run `module load conda` by itself either — that modulefile is broken on current CPE.

Compute nodes need the proxy for HuggingFace / pip / Comet:

```bash
export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=$HTTP_PROXY https_proxy=$HTTPS_PROXY
```

`scripts/alcf/sjob` sets proxy, `HF_HOME`, and conda for you.

## Submit a training job

From the repo root:

```bash
qsub scripts/alcf/sjob
```

Override config / flags without editing the script:

```bash
qsub -v OPT=checkpoints/NAFNet.yml,COMMENT=polaris scripts/alcf/sjob
qsub -v OPT=checkpoints/NAFNet.yml,TEST=1 scripts/alcf/sjob
qsub -v OPT=checkpoints/NAFNet.yml,INFER=1 scripts/alcf/sjob
qsub -v OPT=checkpoints/NAFNet.yml,NGPUS=4 scripts/alcf/sjob
```

Longer 1-node runs (`prod` requires **10+** nodes):

```bash
qsub -q preemptable -l walltime=12:00:00 -v OPT=checkpoints/NAFNet.yml scripts/alcf/sjob
qsub -q capacity    -l walltime=24:00:00 -v OPT=checkpoints/NAFNet.yml scripts/alcf/sjob
```

## Interactive session

```bash
qsub -I -A lighthouse-purdue -q debug -l select=1:system=polaris \
     -l place=scatter -l walltime=01:00:00 -l filesystems=home:eagle
```

Once the prompt appears you are on a compute node:

```bash
cd /lus/eagle/projects/lighthouse-purdue/harshana/svd_v2
source scripts/alcf/load_conda.sh
nvidia-smi
python trainer.py -opt checkpoints/NAFNet.yml -test
```

## Job status

```bash
qstat -u $USER                 # your jobs
qstat -x -u $USER              # include finished (2 weeks of history)
qstat -f <jobid>               # full details
qstat -T -u $USER              # estimated start time
qstat -n -u $USER              # which nodes
qstat -Q                       # all queues
watch -n 10 'qstat -u $USER'   # refresh
```

`S` column:

| | |
|---|---|
| `Q` | queued |
| `R` | running |
| `H` | held |
| `E` | exiting |
| `F` | finished (needs `-x`) |

```bash
qdel <jobid>                   # cancel
qhold <jobid>                  # hold
qrls <jobid>                   # release hold
```

Logs from `scripts/alcf/sjob` (`#PBS -j oe`, merged stdout/stderr):

```bash
ls -lt svd.o* scripts/alcf/svd.o* logs/
```

PBS writes `svd.o<jobid>` in the directory you ran `qsub` from.

## Queues (Polaris)

| Queue | Nodes | Walltime | Notes |
|---|---|---|---|
| `debug` | 1–2 | ≤ 1 h | default in `sjob`; good for tests |
| `debug-scaling` | 1–10 | ≤ 1 h | 1 job per user |
| `preemptable` | 1–10 | ≤ 72 h | can be killed without warning |
| `capacity` | 1–4 | ≤ 168 h | max 2 jobs queued/running per user |
| `prod` | **10–496** | ≤ 24 h | routing queue; do **not** use for 1-node work |

```bash
qstat -Qf debug          # limits for a queue
```

## Hardware / launch

One Polaris node = 4 GPUs. Default in `sjob` is `NGPUS=4`.

```bash
# inside a job
echo $PBS_JOBID
cat $PBS_NODEFILE
nvidia-smi
```

Training (same as Gilbreth, just 4 GPUs/node):

```bash
python trainer.py -opt checkpoints/<project>/<experiment>.yml
accelerate launch --num_processes 4 --mixed_precision bf16 trainer.py -opt checkpoints/<project>/<experiment>.yml
```

## Common gotchas

- Do not run training on the login node.
- `prod` will silently fail or never run if you request fewer than 10 nodes.
- `preemptable` jobs can vanish if someone submits to `demand`.
- Compute nodes have no direct internet without the proxy above.
- `conda/2025-09-25` pins old gcc/hdf5. Always `source scripts/alcf/load_conda.sh` — do not `./` it and do not `module load conda` by hand.
- `#PBS` lines in the script are the defaults; `qsub -q` / `-l` / `-v` on the command line override them.
