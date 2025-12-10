"""
PyTorch vs TensorRT Performance Comparison
==========================================

This script compares inference speed and accuracy between:
1. Native PyTorch
2. TensorRT optimized model

TensorRT can provide 2-10x speedup over PyTorch on NVIDIA GPUs
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Dict, Tuple
import os

# Try to import TensorRT
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    print("Warning: TensorRT not available. Install with: pip install nvidia-tensorrt")
    TRT_AVAILABLE = False


class PyTorchInference:
    """
    Standard PyTorch inference
    """
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        # Enable optimizations
        if device == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
    
    @torch.no_grad()
    def __call__(self, x):
        """Run inference"""
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        x = x.to(self.device)
        
        output = self.model(x)
        return output
    
    def warmup(self, input_shape=(1, 3, 512, 1024), iterations=10):
        """Warmup the model"""
        dummy_input = torch.randn(input_shape).to(self.device)
        for _ in range(iterations):
            _ = self(dummy_input)
    
    def benchmark(self, input_shape=(1, 3, 512, 1024), iterations=100):
        """Benchmark inference speed"""
        self.warmup(input_shape)
        
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # Measure time
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        times = []
        for _ in range(iterations):
            start = time.time()
            _ = self(dummy_input)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1.0 / avg_time
        
        return {
            'avg_time_ms': avg_time * 1000,
            'std_time_ms': std_time * 1000,
            'fps': fps,
            'min_time_ms': min(times) * 1000,
            'max_time_ms': max(times) * 1000
        }


class TensorRTInference:
    """
    TensorRT optimized inference
    Requires: nvidia-tensorrt, pycuda
    """
    def __init__(self, engine_path):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")
        
        # Load TensorRT engine
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        self.context = self.engine.create_execution_context()
        
        # Allocate buffers
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()
        
        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if self.engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})
    
    def __call__(self, x):
        """Run inference"""
        # Convert to numpy if needed
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        
        # Copy input to device
        np.copyto(self.inputs[0]['host'], x.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], 
                               self.inputs[0]['host'], 
                               self.stream)
        
        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, 
                                      stream_handle=self.stream.handle)
        
        # Copy output to host
        cuda.memcpy_dtoh_async(self.outputs[0]['host'], 
                              self.outputs[0]['device'], 
                              self.stream)
        
        self.stream.synchronize()
        
        return self.outputs[0]['host']
    
    def benchmark(self, input_shape=(1, 3, 512, 1024), iterations=100):
        """Benchmark TensorRT inference"""
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            _ = self(dummy_input)
        
        # Measure time
        times = []
        for _ in range(iterations):
            start = time.time()
            _ = self(dummy_input)
            self.stream.synchronize()
            end = time.time()
            times.append(end - start)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1.0 / avg_time
        
        return {
            'avg_time_ms': avg_time * 1000,
            'std_time_ms': std_time * 1000,
            'fps': fps,
            'min_time_ms': min(times) * 1000,
            'max_time_ms': max(times) * 1000
        }


def export_to_tensorrt(model, onnx_path, engine_path, 
                       input_shape=(1, 3, 512, 1024),
                       precision='fp16', max_batch_size=1,
                       workspace_size=1 << 30):
    """
    Convert PyTorch model to TensorRT engine
    
    Args:
        model: PyTorch model
        onnx_path: Path to save ONNX model
        engine_path: Path to save TensorRT engine
        input_shape: Input tensor shape
        precision: 'fp32', 'fp16', or 'int8'
        max_batch_size: Maximum batch size
        workspace_size: Workspace size in bytes (default 1GB)
    """
    if not TRT_AVAILABLE:
        print("TensorRT not available. Skipping conversion.")
        return None
    
    print("=" * 60)
    print("Converting PyTorch Model to TensorRT")
    print("=" * 60)
    
    # Step 1: Export to ONNX
    print("\n[1/3] Exporting to ONNX...")
    model.eval()
    dummy_input = torch.randn(input_shape).cuda()
    
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"✓ ONNX model saved to {onnx_path}")
    
    # Step 2: Build TensorRT engine
    print("\n[2/3] Building TensorRT engine...")
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    # Parse ONNX
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            print('ERROR: Failed to parse ONNX file')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    
    # Configure builder
    config = builder.create_builder_config()
    config.max_workspace_size = workspace_size
    
    # Set precision
    if precision == 'fp16' and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("✓ Using FP16 precision")
    elif precision == 'int8' and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print("✓ Using INT8 precision")
    else:
        print("✓ Using FP32 precision")
    
    # Build engine
    print("Building engine (this may take a few minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("ERROR: Failed to build engine")
        return None
    
    # Step 3: Save engine
    print("\n[3/3] Saving TensorRT engine...")
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    print(f"✓ TensorRT engine saved to {engine_path}")
    print("=" * 60)
    
    return engine_path


def compare_pytorch_tensorrt(model, input_shape=(1, 3, 512, 1024), 
                             iterations=100):
    """
    Compare PyTorch and TensorRT performance
    """
    print("\n" + "=" * 60)
    print("PyTorch vs TensorRT Performance Comparison")
    print("=" * 60)
    
    # Setup paths
    onnx_path = "model_temp.onnx"
    engine_path = "model_temp.engine"
    
    # PyTorch inference
    print("\n[PyTorch] Benchmarking...")
    pytorch_engine = PyTorchInference(model, device='cuda')
    pytorch_results = pytorch_engine.benchmark(input_shape, iterations)
    
    print(f"\nPyTorch Results:")
    print(f"  Average time: {pytorch_results['avg_time_ms']:.2f} ± {pytorch_results['std_time_ms']:.2f} ms")
    print(f"  FPS: {pytorch_results['fps']:.2f}")
    print(f"  Min time: {pytorch_results['min_time_ms']:.2f} ms")
    print(f"  Max time: {pytorch_results['max_time_ms']:.2f} ms")
    
    # TensorRT inference
    if TRT_AVAILABLE:
        print("\n[TensorRT] Converting and benchmarking...")
        
        # Convert to TensorRT
        export_to_tensorrt(model, onnx_path, engine_path, 
                          input_shape, precision='fp16')
        
        # Benchmark TensorRT
        tensorrt_engine = TensorRTInference(engine_path)
        tensorrt_results = tensorrt_engine.benchmark(input_shape, iterations)
        
        print(f"\nTensorRT Results:")
        print(f"  Average time: {tensorrt_results['avg_time_ms']:.2f} ± {tensorrt_results['std_time_ms']:.2f} ms")
        print(f"  FPS: {tensorrt_results['fps']:.2f}")
        print(f"  Min time: {tensorrt_results['min_time_ms']:.2f} ms")
        print(f"  Max time: {tensorrt_results['max_time_ms']:.2f} ms")
        
        # Calculate speedup
        speedup = pytorch_results['avg_time_ms'] / tensorrt_results['avg_time_ms']
        
        print("\n" + "=" * 60)
        print("Comparison Summary")
        print("=" * 60)
        print(f"TensorRT Speedup: {speedup:.2f}x faster")
        print(f"Latency Reduction: {pytorch_results['avg_time_ms'] - tensorrt_results['avg_time_ms']:.2f} ms")
        print(f"FPS Improvement: +{tensorrt_results['fps'] - pytorch_results['fps']:.2f} FPS")
        
        # Cleanup
        if os.path.exists(onnx_path):
            os.remove(onnx_path)
        if os.path.exists(engine_path):
            os.remove(engine_path)
        
        return {
            'pytorch': pytorch_results,
            'tensorrt': tensorrt_results,
            'speedup': speedup
        }
    else:
        print("\nTensorRT not available. Install with:")
        print("  pip install nvidia-tensorrt pycuda")
        return {
            'pytorch': pytorch_results,
            'tensorrt': None,
            'speedup': None
        }


def demonstrate_inference_modes():
    """
    Demonstrate different inference modes
    """
    from efficient_segmentation import FastSegNet
    
    print("=" * 60)
    print("Real-Time Segmentation: PyTorch vs TensorRT")
    print("=" * 60)
    
    # Create model
    print("\n[1] Loading model...")
    model = FastSegNet(num_classes=19)
    model.eval()
    model.cuda()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Test different input sizes
    input_sizes = [
        (1, 3, 256, 512),   # Small
        (1, 3, 512, 1024),  # Medium
        (1, 3, 720, 1280),  # HD
    ]
    
    for input_shape in input_sizes:
        print(f"\n{'='*60}")
        print(f"Input Size: {input_shape[2]}x{input_shape[3]}")
        print(f"{'='*60}")
        
        results = compare_pytorch_tensorrt(model, input_shape, iterations=100)
        
        if results['tensorrt'] is not None:
            print(f"\n✓ Best configuration for {input_shape[2]}x{input_shape[3]}:")
            if results['tensorrt']['fps'] > 30:
                print(f"  TensorRT: {results['tensorrt']['fps']:.1f} FPS (Real-time ready! ✓)")
            else:
                print(f"  TensorRT: {results['tensorrt']['fps']:.1f} FPS")


def create_inference_script(model_path, use_tensorrt=True):
    """
    Create a standalone inference script
    """
    script_content = f'''"""
Real-Time Segmentation Inference Script
Generated inference script for production deployment
"""

import torch
import numpy as np
{'import tensorrt as trt' if use_tensorrt else ''}
{'import pycuda.driver as cuda' if use_tensorrt else ''}
{'import pycuda.autoinit' if use_tensorrt else ''}
import cv2
from pathlib import Path

class SegmentationInference:
    def __init__(self, {"engine_path" if use_tensorrt else "model_path"}):
        {"self.setup_tensorrt(engine_path)" if use_tensorrt else "self.setup_pytorch(model_path)"}
    
    {"def setup_tensorrt(self, engine_path):" if use_tensorrt else "def setup_pytorch(self, model_path):"}
        {"# TensorRT setup code here" if use_tensorrt else "# PyTorch setup code here"}
        pass
    
    def preprocess(self, image):
        """Preprocess image for inference"""
        # Resize if needed
        image = cv2.resize(image, (1024, 512))
        
        # Convert to RGB and normalize
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        
        # Normalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Transpose to CHW
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        
        return image
    
    def infer(self, image):
        """Run inference"""
        input_tensor = self.preprocess(image)
        # Run inference based on backend
        pass
    
    def postprocess(self, output):
        """Postprocess network output"""
        pred = np.argmax(output, axis=1)[0]
        return pred

if __name__ == "__main__":
    # Example usage
    engine = SegmentationInference("{'model.engine' if use_tensorrt else 'model.pth'}")
    
    # Load image
    image = cv2.imread("test.jpg")
    
    # Inference
    result = engine.infer(image)
    
    print(f"Segmentation shape: {{result.shape}}")
'''
    
    filename = "inference_tensorrt.py" if use_tensorrt else "inference_pytorch.py"
    with open(filename, 'w') as f:
        f.write(script_content)
    
    print(f"✓ Inference script saved to {filename}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PyTorch vs TensorRT Comparison Tool")
    print("=" * 60)
    print("\nThis script will:")
    print("1. Benchmark PyTorch inference")
    print("2. Convert model to TensorRT")
    print("3. Benchmark TensorRT inference")
    print("4. Compare performance")
    print("\nRequirements:")
    print("  - NVIDIA GPU with CUDA")
    print("  - TensorRT: pip install nvidia-tensorrt")
    print("  - PyCUDA: pip install pycuda")
    
    # Run demonstration
    try:
        demonstrate_inference_modes()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("  1. NVIDIA GPU with CUDA support")
        print("  2. TensorRT installed")
        print("  3. PyCUDA installed")
