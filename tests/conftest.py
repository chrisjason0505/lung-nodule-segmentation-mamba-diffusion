import os

# the unit tests run on CPU in float64; always use the pure-PyTorch Mamba here,
# even on machines where the fused CUDA kernels are installed
os.environ["LUNGSEG_MAMBA_BACKEND"] = "torch"
