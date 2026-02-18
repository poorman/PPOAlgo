#!/bin/bash
# Optimized startup script for PPOAlgo optimizer
# Configures system for maximum performance with i9-13900K + RTX 3080

echo "🚀 Starting PPOAlgo Optimizer - Turbo Edition"
echo "================================================"

# System info
echo "📊 System Configuration:"
echo "   CPU: $(lscpu | grep 'Model name' | cut -d ':' -f2 | xargs)"
echo "   Cores: $(nproc) logical cores"
echo "   RAM: $(free -h | awk '/^Mem:/ {print $2}')"

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "   GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "   VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"

    # Set GPU to persistence mode for faster operations
    sudo nvidia-smi -pm 1 2>/dev/null || true

    # Set GPU to maximum performance
    sudo nvidia-smi -pl 350 2>/dev/null || true  # RTX 3080 max power

    # Enable GPU boost
    sudo nvidia-settings -a "[gpu:0]/GpuPowerMizerMode=1" 2>/dev/null || true
else
    echo "   GPU: Not detected"
fi

# CPU Performance tuning
echo ""
echo "⚙️ Optimizing CPU performance..."

# Set CPU governor to performance mode (if available)
if [ -f /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
    for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo performance | sudo tee $cpu > /dev/null 2>&1 || true
    done
    echo "   ✅ CPU governor set to performance mode"
fi

# Increase file descriptor limits
ulimit -n 65536 2>/dev/null || true
echo "   ✅ File descriptor limit increased"

# Set Python optimizations
export PYTHONOPTIMIZE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=$(nproc)
export MKL_NUM_THREADS=$(nproc)
export OPENBLAS_NUM_THREADS=$(nproc)
export NUMBA_NUM_THREADS=$(nproc)

# CUDA optimizations
export CUDA_CACHE_MAXSIZE=4294967296  # 4GB cache
export CUDA_FORCE_PTX_JIT=1
export CUDA_CACHE_PATH=/tmp/cuda_cache
mkdir -p /tmp/cuda_cache

echo "   ✅ Environment variables configured"

# Database optimizations
echo ""
echo "🗄️ Optimizing PostgreSQL connection pool..."
export POSTGRES_MAX_CONNECTIONS=200
export POSTGRES_POOL_SIZE=50

# Warm up GPU if available
if command -v python3 &> /dev/null && command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "🔥 Warming up GPU..."
    python3 -c "
try:
    import cupy as cp
    print('   Testing GPU memory allocation...')

    # Allocate 2GB to warm up memory
    test_array = cp.zeros((1024, 1024, 512), dtype=cp.float32)
    print(f'   ✅ GPU memory allocation successful: {test_array.nbytes / 1e9:.1f} GB')

    # Test computation
    result = cp.sum(test_array)
    print(f'   ✅ GPU computation test passed')

    # Clean up
    del test_array
    cp.cuda.MemoryPool().free_all_blocks()

    # Print GPU info
    mem_info = cp.cuda.runtime.memGetInfo()
    print(f'   GPU Memory: {mem_info[1] / 1e9:.1f} GB total, {mem_info[0] / 1e9:.1f} GB free')

except Exception as e:
    print(f'   ⚠️ GPU warmup failed: {e}')
    print('   Continuing with CPU-only mode...')
" 2>/dev/null || echo "   ⚠️ GPU warmup skipped"
fi

# Network optimizations
echo ""
echo "🌐 Optimizing network settings..."
# Increase TCP buffer sizes for better API throughput
sudo sysctl -w net.core.rmem_max=134217728 2>/dev/null || true
sudo sysctl -w net.core.wmem_max=134217728 2>/dev/null || true
sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 134217728" 2>/dev/null || true
sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 134217728" 2>/dev/null || true
echo "   ✅ Network buffers optimized"

# Start the application with optimized settings
echo ""
echo "🚀 Starting optimizer server..."
echo "================================================"

# Use the optimized server file if it exists
if [ -f "/app/server_optimized.py" ]; then
    echo "Using optimized server configuration..."
    exec uvicorn server_optimized:app \
        --host 0.0.0.0 \
        --port 8082 \
        --workers 1 \
        --loop uvloop \
        --access-log \
        --log-level info
else
    echo "Using standard server configuration..."
    exec uvicorn server:app \
        --host 0.0.0.0 \
        --port 8082 \
        --workers 1 \
        --loop uvloop \
        --access-log \
        --log-level info
fi