module load Anaconda3/2024.02
export PATH="/hpc/group/coursess26/ids705/team-project/detect-to-protect/envs/dtp/bin:$PATH"
export LD_LIBRARY_PATH=$(find /hpc/group/coursess26/ids705/team-project/detect-to-protect/envs/dtp/lib/python3.11/site-packages/nvidia -name "*.so*" -exec dirname {} \; 2>/dev/null | sort -u | tr '\n' ':'):${LD_LIBRARY_PATH:-}
cd /hpc/group/coursess26/ids705/team-project/detect-to-protect
echo "DTP env active. CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
