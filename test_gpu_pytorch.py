"""Quick PyTorch GPU verification script"""
import torch

print("=" * 50)
print("PyTorch GPU Diagnostic")
print("=" * 50)

print(f"\nPyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Compute capability: {props.major}.{props.minor}")
    
    # Test GPU computation
    print("\n--- Testing GPU Computation ---")
    try:
        device = torch.device('cuda:0')
        a = torch.randn(1000, 1000, device=device)
        b = torch.randn(1000, 1000, device=device)
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        print("✓ Matrix multiplication on GPU: SUCCESS")
        print(f"  Result shape: {c.shape}")
        
        # Memory test
        print(f"\n  Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
        print(f"  Memory cached: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")
    except Exception as e:
        print(f"✗ GPU computation FAILED: {e}")
else:
    print("\n⚠ NO GPU DETECTED!")
    print("\nTo install PyTorch with CUDA support:")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

print("\n" + "=" * 50)
