"""
Real-Time Inference and Optimization
====================================

Techniques for achieving real-time performance:
1. Model quantization (INT8)
2. TensorRT optimization
3. ONNX export
4. Pruning and distillation
5. Multi-threading
"""

import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np
from typing import Tuple
import time


class ModelOptimizer:
    """
    Optimize segmentation models for real-time inference
    """
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
    
    def benchmark(self, input_size=(1, 3, 512, 1024), iterations=100):
        """Benchmark model inference speed"""
        self.model.eval()
        dummy_input = torch.randn(input_size).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)
        
        # Benchmark
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        start = time.time()
        with torch.no_grad():
            for _ in range(iterations):
                _ = self.model(dummy_input)
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        end = time.time()
        
        avg_time = (end - start) / iterations
        fps = 1.0 / avg_time
        
        return {
            'avg_time_ms': avg_time * 1000,
            'fps': fps,
            'latency_ms': avg_time * 1000
        }
    
    def export_to_onnx(self, save_path, input_size=(1, 3, 512, 1024)):
        """Export model to ONNX format"""
        self.model.eval()
        dummy_input = torch.randn(input_size).to(self.device)
        
        torch.onnx.export(
            self.model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height', 3: 'width'}
            }
        )
        
        print(f"Model exported to {save_path}")
        
        # Verify ONNX model
        onnx_model = onnx.load(save_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified successfully!")
        
        return save_path
    
    def quantize_dynamic(self):
        """Apply dynamic quantization (INT8)"""
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {nn.Conv2d, nn.Linear},
            dtype=torch.qint8
        )
        
        print("Model quantized to INT8")
        return quantized_model
    
    def apply_pruning(self, amount=0.3):
        """Apply structured pruning to reduce model size"""
        import torch.nn.utils.prune as prune
        
        parameters_to_prune = []
        for module in self.model.modules():
            if isinstance(module, nn.Conv2d):
                parameters_to_prune.append((module, 'weight'))
        
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount
        )
        
        # Make pruning permanent
        for module, _ in parameters_to_prune:
            prune.remove(module, 'weight')
        
        print(f"Applied {amount*100}% pruning")
        return self.model


class ONNXInference:
    """
    Fast inference using ONNX Runtime
    """
    def __init__(self, onnx_path, device='cuda'):
        providers = ['CUDAExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def __call__(self, image):
        """Run inference"""
        # Prepare input
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # Run inference
        output = self.session.run(
            [self.output_name],
            {self.input_name: image}
        )[0]
        
        return output
    
    def benchmark(self, input_size=(1, 3, 512, 1024), iterations=100):
        """Benchmark ONNX inference"""
        dummy_input = np.random.randn(*input_size).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            _ = self(dummy_input)
        
        # Benchmark
        start = time.time()
        for _ in range(iterations):
            _ = self(dummy_input)
        end = time.time()
        
        avg_time = (end - start) / iterations
        fps = 1.0 / avg_time
        
        return {
            'avg_time_ms': avg_time * 1000,
            'fps': fps
        }


class TensorRTOptimizer:
    """
    Optimize model using TensorRT for maximum speed
    Requires: nvidia-tensorrt
    """
    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
    
    def build_engine(self, engine_path, precision='fp16', max_batch_size=1):
        """
        Build TensorRT engine from ONNX
        
        Args:
            precision: 'fp32', 'fp16', or 'int8'
        """
        try:
            import tensorrt as trt
        except ImportError:
            print("TensorRT not installed. Install with: pip install nvidia-tensorrt")
            return None
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # Parse ONNX
        with open(self.onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('Failed to parse ONNX')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None
        
        # Build engine
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 30  # 1GB
        
        if precision == 'fp16':
            config.set_flag(trt.BuilderFlag.FP16)
        elif precision == 'int8':
            config.set_flag(trt.BuilderFlag.INT8)
        
        engine = builder.build_engine(network, config)
        
        # Save engine
        with open(engine_path, 'wb') as f:
            f.write(engine.serialize())
        
        print(f"TensorRT engine saved to {engine_path}")
        return engine_path


class RealtimeSegmentation:
    """
    Complete real-time segmentation pipeline
    """
    def __init__(self, model_path, backend='pytorch', device='cuda'):
        self.backend = backend
        self.device = device
        
        if backend == 'pytorch':
            self.model = torch.load(model_path)
            self.model.eval()
            self.model.to(device)
        elif backend == 'onnx':
            self.model = ONNXInference(model_path, device)
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def preprocess(self, image):
        """Preprocess image for inference"""
        # Convert to tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image)
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std
        
        return image.unsqueeze(0)
    
    def postprocess(self, output, original_size):
        """Postprocess network output"""
        # Get class predictions
        pred = output.argmax(dim=1).squeeze(0)
        
        # Resize to original size
        if pred.shape != original_size:
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(0).unsqueeze(0).float(),
                size=original_size,
                mode='nearest'
            ).squeeze().long()
        
        return pred
    
    def infer(self, image):
        """Run inference on single image"""
        original_size = image.shape[-2:]
        
        # Preprocess
        input_tensor = self.preprocess(image)
        
        if self.backend == 'pytorch':
            input_tensor = input_tensor.to(self.device)
            with torch.no_grad():
                output = self.model(input_tensor)
        else:
            output = self.model(input_tensor.cpu().numpy())
            output = torch.from_numpy(output)
        
        # Postprocess
        prediction = self.postprocess(output, original_size)
        
        return prediction
    
    def infer_video(self, video_path, output_path=None):
        """Process video in real-time"""
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_times = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB and tensor
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
            
            # Inference
            start = time.time()
            prediction = self.infer(frame_tensor)
            end = time.time()
            
            frame_times.append(end - start)
            
            # Visualize (optional)
            if output_path:
                # Create colored segmentation map
                colored_pred = self.colorize_prediction(prediction)
                overlay = cv2.addWeighted(frame, 0.6, colored_pred, 0.4, 0)
                
                # Add FPS info
                fps_text = f"FPS: {1.0/(end-start):.1f}"
                cv2.putText(overlay, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                out.write(overlay)
        
        cap.release()
        if output_path:
            out.release()
        
        avg_fps = 1.0 / np.mean(frame_times)
        print(f"Average FPS: {avg_fps:.2f}")
        
        return avg_fps
    
    def colorize_prediction(self, prediction):
        """Convert prediction to colored image"""
        # Simple color palette
        palette = np.array([
            [128, 64, 128], [244, 35, 232], [70, 70, 70],
            [102, 102, 156], [190, 153, 153], [153, 153, 153],
            [250, 170, 30], [220, 220, 0], [107, 142, 35],
            [152, 251, 152], [70, 130, 180], [220, 20, 60],
            [255, 0, 0], [0, 0, 142], [0, 0, 70],
            [0, 60, 100], [0, 80, 100], [0, 0, 230],
            [119, 11, 32]
        ])
        
        colored = palette[prediction.cpu().numpy()]
        return colored.astype(np.uint8)


if __name__ == "__main__":
    print("=" * 60)
    print("Real-Time Inference Optimization Tools")
    print("=" * 60)
    print("\nAvailable Optimizations:")
    print("1. ONNX Export - Cross-platform deployment")
    print("2. Dynamic Quantization - INT8 for faster inference")
    print("3. TensorRT - Maximum GPU performance")
    print("4. Model Pruning - Reduce model size")
    print("\nBackends:")
    print("- PyTorch (Development)")
    print("- ONNX Runtime (Production)")
    print("- TensorRT (Edge devices)")
    print("=" * 60)
