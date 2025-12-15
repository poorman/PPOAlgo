# 🚀 How to Run with RTX 3090 GPU Acceleration

To fully utilize your NVIDIA RTX 3090 GPU for massive grid-search backtesting:

## 1. Prerequisites
- NVIDIA Drivers installed on host
- NVIDIA Container Toolkit installed (allows Docker to access GPU)
  - Windows: Installed with Docker Desktop
  - Linux: `sudo apt-get install -y nvidia-container-toolkit`

## 2. Build the GPU Image
Run this command in the `optimizer` directory:

```bash
cd optimizer
docker build -f Dockerfile.gpu -t ppoalgo-optimizer-gpu .
```

## 3. Run the GPU Container
Stop the existing optimizer and run the GPU version:

```bash
# Stop CPU optimizer
docker stop ppoalgo_optimizer
docker rm ppoalgo_optimizer

# Run GPU optimizer
docker run -d \
  --name ppoalgo_optimizer \
  --gpus all \
  --network ppoalgo_default \
  -p 8082:8000 \
  -e PPOALGO_API=http://ppoalgo_api_1:8000 \
  -e POSTGRES_HOST=e0fdfc8ce6e4_ppoalgo_db_1 \
  -v $(pwd)/results:/app/results \
  ppoalgo-optimizer-gpu
```

## 4. Verify GPU Access
Check the logs to see if CuPy loaded:

```bash
docker logs ppoalgo_optimizer
```
You should see: `INFO:root:CuPy loaded - GPU acceleration enabled`

## 5. Usage
1. Go to http://localhost:8082
2. Check the **"Available GPU: RTX 3090"** box
3. Click "Start Optimization"
4. The system will now parallelize thousands of backtests on your GPU!
