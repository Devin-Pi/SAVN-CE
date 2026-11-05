FROM rdpi/ss_v2:latest
ENV PATH=/opt/conda/envs/ss/bin:$PATH
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6:/opt/conda/envs/ss/lib/python3.9/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
RUN python -m pip install --no-cache-dir numba==0.60.0 llvmlite==0.43.0 gym==0.23.0 attrs==24.3.0 yacs==0.1.8 ifcfg torchsummary moviepy==1.0.3 librosa==0.10.2.post1 soundfile==0.13.1
COPY docker/patch_habitat.py /tmp/patch_habitat.py
RUN python /tmp/patch_habitat.py
WORKDIR /workspace/SAVN-CE
ENV PYTHONPATH=/workspace/SAVN-CE
CMD ["sleep", "infinity"]
