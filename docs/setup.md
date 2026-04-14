# Detect-to-Protect — Team DCC Setup Guide

> Project path: `/hpc/group/coursess26/ids705/team-project/detect-to-protect`  
> All work lives in the shared folder. Do not clone to your home directory.  
> **Every session: `source /hpc/group/coursess26/ids705/team-project/detect-to-protect/activate.sh`**

---

## 1. First-Time Setup (do once per netid)

### 1.1 SSH into DCC

```bash
ssh <netid>@dcc-login.oit.duke.edu
```

### 1.2 Get a GPU node

```bash
srun --partition=courses-gpu \
     --gres=gpu:1 \
     --mem=32G \
     --cpus-per-task=4 \
     --time=02:00:00 \
     --pty bash
```

Note the GPU node hostname (e.g. `dcc-courses-gpu-01`) — you'll need it for the SSH tunnel.

### 1.3 Activate the shared environment

```bash
source /hpc/group/coursess26/ids705/team-project/detect-to-protect/activate.sh
```

That's it. The script handles `module load`, `PATH`, and `LD_LIBRARY_PATH` automatically.  
You should see: `DTP env active. CUDA: True`

### 1.4 Register the Jupyter kernel (once per netid only)

```bash
python -m ipykernel install --user \
    --name dtp \
    --display-name "DTP (GPU)"
```

### 1.5 Fix the kernel to point at the shared env (once per netid only)

After registering, patch the kernel.json to ensure it uses the correct Python:

```bash
sed -i 's|/opt/apps/rhel9/Anaconda3-2024.02/bin/python|/hpc/group/coursess26/ids705/team-project/detect-to-protect/envs/dtp/bin/python|g' \
    ~/.local/share/jupyter/kernels/dtp/kernel.json
```

Verify it's correct:
```bash
cat ~/.local/share/jupyter/kernels/dtp/kernel.json
# "argv" should start with .../envs/dtp/bin/python
```

### 1.6 Validate your setup

```bash
python -c "
import torch, sys
print('Python: ', sys.executable)
print('Torch:  ', torch.__file__)
print('CUDA:   ', torch.cuda.is_available())
print('Device: ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
"
```

**Expected output:**
```
Python:  .../detect-to-protect/envs/dtp/bin/python
Torch:   .../detect-to-protect/envs/dtp/lib/.../torch/__init__.py
CUDA:    True
Device:  Tesla P100-PCIE-16GB
```

---

## 2. Every Session — Interactive GPU Work

```bash
# 1. SSH in
ssh <netid>@dcc-login.oit.duke.edu

# 2. Get a GPU node
srun --partition=courses-gpu --gres=gpu:1 --mem=32G --cpus-per-task=4 --time=02:00:00 --pty bash

# 3. Activate everything (one line)
source /hpc/group/coursess26/ids705/team-project/detect-to-protect/activate.sh
```

---

## 3. Running Jupyter Notebook

### 3.1 Launch on the GPU node

```bash
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0
```

### 3.2 On your local machine (new terminal tab)

```bash
# Replace dcc-courses-gpu-01 with your actual node hostname
ssh -L 8888:dcc-courses-gpu-01:8888 <netid>@dcc-login.oit.duke.edu
```

### 3.3 Open in browser

```
http://localhost:8888
```

Navigate to `notebooks/train.ipynb`. Select kernel: **DTP (GPU)**.

### 3.4 Verify correct kernel inside notebook

Run this in the first cell to confirm:
```python
import sys
print(sys.executable)
# Must show: .../envs/dtp/bin/python
```

If it shows `/opt/apps/rhel9/...` — go to Section 7.

---

## 4. Running Training as a Batch Job (recommended for full runs)

Use `sbatch` for any run longer than ~30 min — it survives SSH disconnects.  
Create `scripts/submit_train.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=dtp-train
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

source /hpc/group/coursess26/ids705/team-project/detect-to-protect/activate.sh

python src/train.py
```

Submit and monitor:
```bash
mkdir -p logs
sbatch scripts/submit_train.sh
squeue -u <netid>                      # check job status
tail -f logs/train_<job_id>.out        # live log output
```

### 4.1 Running baseline prediction jobs (batch)

Do not run `python src/predict_baseline.py` directly on a login node. Use `sbatch`.

```bash
# baseline-scratch
CLIP_LEN=32 ANCHOR_OFFSET_SEC=0.0 RUN_NAME=baseline-scratch sbatch scripts/submit_predict_baseline.sh

# baseline-clip64-ofs0.0
CLIP_LEN=64 ANCHOR_OFFSET_SEC=0.0 RUN_NAME=baseline-clip64-ofs0.0 sbatch scripts/submit_predict_baseline.sh

# baseline-clip100-ofs0.0
CLIP_LEN=100 ANCHOR_OFFSET_SEC=0.0 RUN_NAME=baseline-clip100-ofs0.0 sbatch scripts/submit_predict_baseline.sh

# baseline-clip32-ofs0.5
CLIP_LEN=32 ANCHOR_OFFSET_SEC=0.5 RUN_NAME=baseline-clip32-ofs0.5 sbatch scripts/submit_predict_baseline.sh
```

Monitor prediction logs:
```bash
squeue -u <netid>
tail -f logs/predict_baseline_<jobid>.out
```

---

## 5. Project Structure

```
detect-to-protect/
├── activate.sh          # ← source this every session
├── docs/
│   ├── setup.md
│   └── project-decisions.md
├── data/
│   ├── train.csv
│   ├── test.csv
│   ├── sample_submission.csv
│   ├── frames/          # raw video frames (train/ test/)
│   ├── segmentation/    # YOLO seg masks   (train/ test/)
│   └── depth/           # DepthAnything v2  (train/ test/)
├── envs/
│   └── dtp/             # shared conda env — never modify without team consensus
├── notebooks/
│   ├── preprocess.ipynb
│   └── train.ipynb
├── scripts/
│   ├── submit_train.sh
│   ├── submit_train_v2.sh
│   ├── submit_train_baseline.sh
│   └── submit_predict_baseline.sh
├── src/
│   ├── __init__.py
│   ├── data/            # video_loader, segmentation, depth, dataset
│   ├── model/           # video_transformer, classifier
│   ├── train.py
│   └── predict.py
├── logs/                # sbatch output logs
├── requirements.txt
└── outputs/             # checkpoints + submissions
```

---

## 6. Installing New Packages

Always install into the shared env — never into `~/.local`.

```bash
# activate.sh must be sourced first, then use the env pip explicitly
/hpc/group/coursess26/ids705/team-project/detect-to-protect/envs/dtp/bin/pip install <package>

# Commit the change
pip freeze > /hpc/group/coursess26/ids705/team-project/detect-to-protect/requirements.txt
```

**Never** run bare `pip install <package>` — it lands in `~/.local` and teammates won't have it.

---

## 7. Weights & Biases (W&B) Setup

Use W&B to track training curves, run configs, and best checkpoints.

### 7.1 Install and login (once per netid)

```bash
source /hpc/group/coursess26/ids705/team-project/detect-to-protect/activate.sh
/hpc/group/coursess26/ids705/team-project/detect-to-protect/envs/dtp/bin/pip install wandb
/hpc/group/coursess26/ids705/team-project/detect-to-protect/envs/dtp/bin/wandb login
```

Paste your API key from: `https://wandb.ai/settings/account`

Notes:
- W&B may create credentials in `~/.netrc` (this is normal).
- If `wandb` is not found, use the full path shown above.

### 7.2 Optional: set API key in environment

```bash
export WANDB_API_KEY=<your_api_key>
export WANDB_MODE=online
unset WANDB_DISABLED
```

To persist across sessions:
```bash
echo 'export WANDB_API_KEY=<your_api_key>' >> ~/.bashrc
```

### 7.3 Submit jobs with W&B env exported

```bash
sbatch --export=ALL,WANDB_API_KEY=$WANDB_API_KEY scripts/submit_train_baseline.sh
```

Open project runs in browser (replace with your account/project):
```text
https://wandb.ai/<entity>/<project>
```

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA: False` + driver version warning | Wrong torch build; reinstall: `pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121` |
| Python shows `/opt/apps/rhel9/...` in notebook | Run Section 1.5 kernel fix, then change kernel in top-right |
| `ModuleNotFoundError: No module named 'torch'` in notebook | Wrong kernel selected; change to **DTP (GPU)** and verify with `print(sys.executable)` |
| `ImportError: libcudnn.so.9` | `LD_LIBRARY_PATH` not set; re-source `activate.sh` |
| Torch loads from `~/.local/...` | `pip uninstall torch torchvision -y` then reinstall via shared env pip |
| Job killed immediately | Not on a GPU node; always `srun` or `sbatch` before running training |
| SSH tunnel drops mid-training | Switch to `sbatch` — job continues even after disconnect |
| Prediction process is `Killed` at start | You ran on login node/CPU; submit `scripts/submit_predict_baseline.sh` with `sbatch` |
| Accidentally edited `~/detect-to-protect` | `rm -rf ~/detect-to-protect`; work only in the shared path |
| W&B page shows no runs | Check correct entity/project URL and clear dashboard filters |
| `wandb: command not found` | Use full binary path: `/hpc/group/.../envs/dtp/bin/wandb` |

---

## 9. Quick Reference — Copy-Paste Session Startup

```bash
# Terminal 1 — on DCC
ssh <netid>@dcc-login.oit.duke.edu
srun --partition=courses-gpu --gres=gpu:1 --mem=32G --cpus-per-task=4 --time=02:00:00 --pty bash
source /hpc/group/coursess26/ids705/team-project/detect-to-protect/activate.sh
jupyter notebook --no-browser --port=8888 --ip=0.0.0.0
```

```bash
# Terminal 2 — your Mac (replace gpu-01 with your actual node hostname)
ssh -L 8888:dcc-courses-gpu-01:8888 <netid>@dcc-login.oit.duke.edu
```

Then open `http://localhost:8888` → `notebooks/train.ipynb` → kernel: **DTP (GPU)**.
