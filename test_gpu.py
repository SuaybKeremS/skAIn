"""Quick GPU verification script"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use NVIDIA RTX (GPU 0)

import tensorflow as tf

print("=" * 50)
print("TensorFlow GPU Diagnostic")
print("=" * 50)

print(f"\nTensorFlow version: {tf.__version__}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

# List all physical devices
print("\n--- Physical Devices ---")
for device in tf.config.list_physical_devices():
    print(f"  {device.device_type}: {device.name}")

# Check GPU specifically
gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPUs detected: {len(gpus)}")

if gpus:
    for gpu in gpus:
        print(f"  GPU: {gpu.name}")
    
    # Try a simple GPU operation
    print("\n--- Testing GPU Computation ---")
    try:
        with tf.device('/GPU:0'):
            a = tf.random.normal([1000, 1000])
            b = tf.random.normal([1000, 1000])
            c = tf.matmul(a, b)
            print(f"Matrix multiplication on GPU: SUCCESS")
            print(f"Result shape: {c.shape}")
    except Exception as e:
        print(f"GPU computation FAILED: {e}")
else:
    print("\n⚠ NO GPU DETECTED!")
    print("\nPossible fixes:")
    print("  1. Install tensorflow[and-cuda]: pip install tensorflow[and-cuda]")
    print("  2. Check CUDA installation: nvcc --version")
    print("  3. Check cuDNN installation")
    print("  4. Verify nvidia-smi shows the GPU")

print("\n" + "=" * 50)
