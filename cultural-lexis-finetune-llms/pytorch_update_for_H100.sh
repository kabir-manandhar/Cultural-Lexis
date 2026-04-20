# Force install PyTorch 2.8 specifically for CUDA 12.8
uv pip install torch==2.8.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128 \
    --force-reinstall