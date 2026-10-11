# Body network + head cVAE pseudo-CT, as submitted to BIC-MAC 2026.
#
# Build from the repository root so the weights are in the build context:
#     docker build -f inference/Dockerfile -t bic-mac-pseudoct .
#
# The base image matches the torch/CUDA the pipeline was validated with
# (torch 2.5.1 + cu121); CUDA 12.1 covers the challenge's A40 (sm_86).
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

WORKDIR /app

# Fail fast and unbuffered, so a crash shows up in the logs.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/mpl \
    OMP_NUM_THREADS=8

COPY inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code
COPY inference/predict.py inference/unet.py inference/transforms.py \
     inference/dataset.py inference/head_model.py inference/head_crop.py \
     inference/blend.py ./

# Weights last: they are the largest layer and change least often. Only the
# two inference checkpoints go in; the pretraining checkpoints are not needed.
COPY weights/body_finetune_e0092.pth weights/head_finetune_e070.pth ./weights/

# The harness mounts /data/features (ro) and /data/output (rw) and runs the
# container with no arguments.
ENTRYPOINT ["python", "predict.py"]
