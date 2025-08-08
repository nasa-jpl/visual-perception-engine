#!/bin/bash
set -e

build_torch2trt() {
    echo "[INFO] Cloning and patching torch2trt..."

    git clone https://github.com/NVIDIA-AI-IOT/torch2trt /tmp/torch2trt
    pushd /tmp/torch2trt

    # Patch flattener.py
    sed -i '/^[[:space:]]*return[[:space:]]\+isinstance(x, torch.Tensor)/c\    return isinstance(x, torch.Tensor) and (x.dtype is torch.half or x.dtype is torch.float or x.dtype == torch.bool)' ./torch2trt/flattener.py

    # Patch CMakeLists.txt
    sed -i 's|^set(CUDA_ARCHITECTURES.*|#|g' ./CMakeLists.txt
    sed -i 's|Catch2_FOUND|False|g' ./CMakeLists.txt

    # Patch setup.py for CUDA_HOME
    python3 - <<EOF
import os, textwrap
p = "/tmp/torch2trt/setup.py"
block = textwrap.dedent("""\
    import os
    import setuptools
    CUDA_HOME = os.environ.get("CUDA_HOME") or "/usr/local/cuda"
    os.environ["CUDA_HOME"] = CUDA_HOME
    os.environ.setdefault("CUDA_PATH", CUDA_HOME)
    import torch.utils.cpp_extension as _cpp
    _cpp.CUDA_HOME = os.environ["CUDA_HOME"]
""")
with open(p, "r+", encoding="utf-8") as f:
    src = f.read()
    if block not in src:
        f.seek(0)
        f.write(block + "\\n" + src)
        f.truncate()
EOF

    # Build the wheel
    python3 -m pip wheel --no-cache-dir --no-deps -w . .

    popd
    mv /tmp/torch2trt/torch2trt-0.5.0-py3-none-any.whl .
    rm -rf /tmp/torch2trt
    echo "[INFO] torch2trt wheel built successfully."
}
