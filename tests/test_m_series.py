"""
Test Efficient Segmentation on Apple M-Series GPU
=================================================

Compare CPU vs MPS (Metal Performance Shaders) performance on macOS
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import time
import numpy as np
from src.models.efficient_segmentation import FastSegNet


def test_device_performance(model, device_name, input_shape=(1, 3, 512, 1024), iterations=100):
    """
    Test model performance on a specific device
    """
    print(f"\n{'='*60}")
    print(f"Testing on: {device_name.upper()}")
    print(f"{'='*60}")
    
    # Setup device
    if device_name == 'mps':
        if not torch.backends.mps.is_available():
            print("❌ MPS not available on this system")
            return None
        device = torch.device('mps')
    elif device_name == 'cuda':
        if not torch.cuda.is_available():
            print("❌ CUDA not available on this system")
            return None
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # Move model to device
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)
    
    print(f"Input shape: {input_shape}")
    print(f"Device: {device}")
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Synchronize if needed (MPS doesn't have synchronize in PyTorch 1.13)
    if device_name == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"Running {iterations} iterations...")
    times = []
    
    with torch.no_grad():
        for i in range(iterations):
            start = time.time()
            output = model(dummy_input)
            
            # Synchronize to get accurate timing (CUDA only in PyTorch 1.13)
            if device_name == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end = time.time()
            times.append(end - start)
            
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i+1}/{iterations}")
    
    # Calculate statistics
    times = np.array(times)
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    fps = 1.0 / avg_time
    
    results = {
        'device': device_name,
        'avg_time_ms': avg_time * 1000,
        'std_time_ms': std_time * 1000,
        'min_time_ms': min_time * 1000,
        'max_time_ms': max_time * 1000,
        'fps': fps
    }
    
    print(f"\n{device_name.upper()} Results:")
    print(f"  Average time: {results['avg_time_ms']:.2f} ± {results['std_time_ms']:.2f} ms")
    print(f"  Min time: {results['min_time_ms']:.2f} ms")
    print(f"  Max time: {results['max_time_ms']:.2f} ms")
    print(f"  FPS: {results['fps']:.2f}")
    
    return results


def compare_all_devices():
    """
    Compare performance across all available devices
    """
    print("="*60)
    print("Apple M-Series GPU Performance Test")
    print("Real-Time Semantic Segmentation")
    print("="*60)
    
    # Create model
    print("\nInitializing model...")
    model = FastSegNet(num_classes=19)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Test different resolutions
    resolutions = [
        (1, 3, 256, 512),   # Small - 256x512
        (1, 3, 512, 1024),  # Medium - 512x1024
        (1, 3, 720, 1280),  # HD - 720x1280
    ]
    
    # Available devices on macOS
    devices = ['cpu']
    if torch.backends.mps.is_available():
        devices.append('mps')
    
    print(f"\nAvailable devices: {', '.join(devices)}")
    
    all_results = {}
    
    for input_shape in resolutions:
        resolution_name = f"{input_shape[2]}x{input_shape[3]}"
        print(f"\n{'='*60}")
        print(f"Testing Resolution: {resolution_name}")
        print(f"{'='*60}")
        
        all_results[resolution_name] = {}
        
        for device in devices:
            # Create fresh model for each device
            test_model = FastSegNet(num_classes=19)
            result = test_device_performance(test_model, device, input_shape, iterations=50)
            if result:
                all_results[resolution_name][device] = result
    
    # Print comparison summary
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("="*60)
    
    for resolution_name, device_results in all_results.items():
        print(f"\n{resolution_name}:")
        print("-" * 60)
        
        for device, result in device_results.items():
            real_time = "✓ Real-time" if result['fps'] >= 30 else "✗ Not real-time"
            print(f"  {device.upper():8} | {result['fps']:6.2f} FPS | "
                  f"{result['avg_time_ms']:6.2f} ms | {real_time}")
        
        # Calculate speedup if both CPU and MPS available
        if 'cpu' in device_results and 'mps' in device_results:
            cpu_time = device_results['cpu']['avg_time_ms']
            mps_time = device_results['mps']['avg_time_ms']
            speedup = cpu_time / mps_time
            print(f"\n  MPS Speedup: {speedup:.2f}x faster than CPU")
            print(f"  Latency reduction: {cpu_time - mps_time:.2f} ms")
    
    # Recommendations
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if 'mps' in devices:
        best_resolution = None
        best_fps = 0
        
        for resolution_name, device_results in all_results.items():
            if 'mps' in device_results:
                fps = device_results['mps']['fps']
                if fps >= 30 and fps > best_fps:
                    best_fps = fps
                    best_resolution = resolution_name
        
        if best_resolution:
            print(f"\n✓ Best real-time configuration on M-Series GPU:")
            print(f"  Resolution: {best_resolution}")
            print(f"  FPS: {best_fps:.2f}")
            print(f"  Device: MPS (Apple Silicon GPU)")
        else:
            print("\n⚠ No configuration achieved 30+ FPS for real-time")
            print("  Consider:")
            print("  - Using smaller input resolution")
            print("  - Further model optimization")
            print("  - Quantization (INT8)")
    else:
        print("\n⚠ MPS (Apple Silicon GPU) not available")
        print("  Running on CPU only")
    
    print("\n" + "="*60)
    
    return all_results


def test_mps_optimizations():
    """
    Test MPS-specific optimizations
    """
    if not torch.backends.mps.is_available():
        print("MPS not available")
        return
    
    print("\n" + "="*60)
    print("MPS Optimization Test")
    print("="*60)
    
    model = FastSegNet(num_classes=19).to('mps')
    model.eval()
    
    input_tensor = torch.randn(1, 3, 512, 1024).to('mps')
    
    # Test 1: Standard inference
    print("\n[Test 1] Standard Inference")
    times = []
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
        # Warmup complete
        
        for _ in range(50):
            start = time.time()
            _ = model(input_tensor)
            # MPS timing (no synchronize in PyTorch 1.13)
            times.append(time.time() - start)
    
    print(f"Average FPS: {1.0/np.mean(times):.2f}")
    
    # Test 2: Half precision (FP16)
    print("\n[Test 2] Half Precision (FP16)")
    model_fp16 = model.half()
    input_fp16 = input_tensor.half()
    
    times = []
    with torch.no_grad():
        for _ in range(10):
            _ = model_fp16(input_fp16)
        # Warmup complete
        
        for _ in range(50):
            start = time.time()
            _ = model_fp16(input_fp16)
            # MPS timing (no synchronize in PyTorch 1.13)
            times.append(time.time() - start)
    
    print(f"Average FPS: {1.0/np.mean(times):.2f}")
    print("Note: FP16 may not provide speedup on all M-series chips")


if __name__ == "__main__":
    # Check MPS availability
    print("System Information:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  MPS available: {torch.backends.mps.is_available()}")
    print(f"  MPS built: {torch.backends.mps.is_built()}")
    
    if torch.backends.mps.is_available():
        print(f"  ✓ Apple Silicon GPU detected!")
    else:
        print(f"  ⚠ MPS not available - will test CPU only")
    
    # Run comprehensive comparison
    results = compare_all_devices()
    
    # Test MPS optimizations if available
    if torch.backends.mps.is_available():
        test_mps_optimizations()
    
    print("\n✓ Testing complete!")
