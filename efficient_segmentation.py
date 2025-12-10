"""
Real-Time Semantic Segmentation Project
========================================

This project implements efficient segmentation architectures for real-time inference
while maintaining high accuracy.

Key Design Principles:
1. Efficient encoder-decoder architectures
2. Lightweight backbones (MobileNet, EfficientNet)
3. Fast inference optimization techniques
4. Multi-scale feature fusion
5. Depthwise separable convolutions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise Separable Convolution: Reduces computation by ~9x compared to standard conv
    Used in MobileNet and other efficient architectures
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # Depthwise: separate convolution per input channel
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        
        # Pointwise: 1x1 convolution to combine channels
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class InvertedResidual(nn.Module):
    """
    Inverted Residual Block (MobileNetV2 building block)
    Expand -> Depthwise -> Project with skip connection
    """
    def __init__(self, in_channels, out_channels, stride, expand_ratio=6):
        super().__init__()
        self.stride = stride
        hidden_dim = in_channels * expand_ratio
        self.use_residual = self.stride == 1 and in_channels == out_channels
        
        layers = []
        if expand_ratio != 1:
            # Expansion (pointwise)
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, 
                     groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Projection (pointwise)
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class AttentionRefinementModule(nn.Module):
    """
    Attention Refinement Module (ARM) from BiSeNet
    Refines features using global average pooling and attention
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Attention branch
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        attention = self.attention(x)
        x = x * attention
        return x


class FeatureFusionModule(nn.Module):
    """
    Feature Fusion Module (FFM)
    Efficiently fuses features from different scales
    """
    def __init__(self, in_channels, out_channels, reduction=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Channel attention
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x1, x2):
        # Concatenate features
        x = torch.cat([x1, x2], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        
        # Apply attention
        attention = self.attention(x)
        x = x + x * attention
        x = self.relu(x)
        return x


class LightweightEncoder(nn.Module):
    """
    Lightweight Encoder using Inverted Residuals
    Efficient feature extraction with reduced parameters
    """
    def __init__(self, in_channels=3):
        super().__init__()
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        # Inverted residual blocks
        self.layer1 = InvertedResidual(32, 16, stride=1, expand_ratio=1)
        self.layer2 = nn.Sequential(
            InvertedResidual(16, 24, stride=2, expand_ratio=6),
            InvertedResidual(24, 24, stride=1, expand_ratio=6)
        )
        self.layer3 = nn.Sequential(
            InvertedResidual(24, 32, stride=2, expand_ratio=6),
            InvertedResidual(32, 32, stride=1, expand_ratio=6),
            InvertedResidual(32, 32, stride=1, expand_ratio=6)
        )
        self.layer4 = nn.Sequential(
            InvertedResidual(32, 64, stride=2, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6),
            InvertedResidual(64, 64, stride=1, expand_ratio=6)
        )
        self.layer5 = nn.Sequential(
            InvertedResidual(64, 96, stride=1, expand_ratio=6),
            InvertedResidual(96, 96, stride=1, expand_ratio=6),
            InvertedResidual(96, 96, stride=1, expand_ratio=6)
        )
    
    def forward(self, x):
        x = self.conv1(x)      # 1/2
        x1 = self.layer1(x)    # 1/2
        x2 = self.layer2(x1)   # 1/4
        x3 = self.layer3(x2)   # 1/8
        x4 = self.layer4(x3)   # 1/16
        x5 = self.layer5(x4)   # 1/16
        
        return x2, x3, x5  # Multi-scale features


class EfficientSegmentationHead(nn.Module):
    """
    Efficient Segmentation Head
    Lightweight decoder for real-time inference
    """
    def __init__(self, low_channels, high_channels, num_classes):
        super().__init__()
        
        # Process high-level features
        self.high_arm = AttentionRefinementModule(high_channels, 128)
        
        # Process low-level features
        self.low_conv = nn.Sequential(
            nn.Conv2d(low_channels, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Fusion module
        self.fusion = FeatureFusionModule(256, 128, reduction=4)
        
        # Final classifier
        self.classifier = nn.Sequential(
            DepthwiseSeparableConv(128, 128),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, num_classes, 1)
        )
    
    def forward(self, low_feat, high_feat):
        # Process features
        high_feat = self.high_arm(high_feat)
        low_feat = self.low_conv(low_feat)
        
        # Upsample high-level features
        high_feat = F.interpolate(high_feat, size=low_feat.shape[2:],
                                 mode='bilinear', align_corners=False)
        
        # Fuse features
        x = self.fusion(low_feat, high_feat)
        
        # Generate output
        x = self.classifier(x)
        return x


class FastSegNet(nn.Module):
    """
    Fast Segmentation Network for Real-Time Inference
    
    Architecture inspired by BiSeNet, MobileNet, and efficient design principles
    
    Key Features:
    - Lightweight encoder with inverted residuals
    - Multi-scale feature extraction
    - Attention refinement modules
    - Efficient feature fusion
    - Target: >30 FPS on edge devices
    
    Args:
        num_classes: Number of segmentation classes
        in_channels: Input image channels (default: 3 for RGB)
    """
    def __init__(self, num_classes=19, in_channels=3):
        super().__init__()
        
        self.encoder = LightweightEncoder(in_channels)
        self.head = EfficientSegmentationHead(24, 96, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        input_size = x.shape[2:]
        
        # Extract multi-scale features
        low_feat, mid_feat, high_feat = self.encoder(x)
        
        # Generate segmentation output
        output = self.head(low_feat, high_feat)
        
        # Upsample to input size
        output = F.interpolate(output, size=input_size,
                              mode='bilinear', align_corners=False)
        
        return output
    
    def get_params_and_flops(self, input_size=(1, 3, 512, 1024)):
        """Calculate model parameters and FLOPs"""
        from thop import profile, clever_format
        
        dummy_input = torch.randn(input_size)
        flops, params = profile(self, inputs=(dummy_input,))
        flops, params = clever_format([flops, params], "%.3f")
        
        return params, flops


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Fast Segmentation Network - Real-Time Architecture")
    print("=" * 60)
    
    # Initialize model
    model = FastSegNet(num_classes=19)
    model.eval()
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 512, 1024)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Calculate model size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    
    # Calculate inference time
    import time
    iterations = 100
    
    with torch.no_grad():
        # Warm up
        for _ in range(10):
            _ = model(dummy_input)
        
        # Measure time
        start = time.time()
        for _ in range(iterations):
            _ = model(dummy_input)
        end = time.time()
    
    avg_time = (end - start) / iterations
    fps = 1.0 / avg_time
    
    print(f"\nInference Performance:")
    print(f"Average time: {avg_time*1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    
    print("\n" + "=" * 60)
    print("Architecture Design Complete!")
    print("=" * 60)
