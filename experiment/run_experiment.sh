#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Override if your interpreter is not on PATH, e.g. PYTHON_EXEC=/path/to/venv/bin/python
PYTHON_EXEC="${PYTHON_EXEC:-python3}"

echo "================================================================="
echo "  Real-Time Roofline: Arithmetic Intensity Best Cap Sweep"
echo "================================================================="

# 1. Compile CUDA kernels if not already built or if .cu is newer
if [ ! -f "$SCRIPT_DIR/intensity_benchmark_kernels.so" ] || [ "$SCRIPT_DIR/intensity_benchmark_kernels.cu" -nt "$SCRIPT_DIR/intensity_benchmark_kernels.so" ]; then
    echo "[1/2] Compiling intensity_benchmark_kernels.so..."
    bash "$SCRIPT_DIR/compile_kernels.sh"
else
    echo "[1/2] intensity_benchmark_kernels.so is up to date."
fi

# 2. Run Python Experiment Harness
echo "[2/2] Launching Python Benchmark Sweep..."
$PYTHON_EXEC "$SCRIPT_DIR/benchmark_ai_powercaps.py" "$@"
