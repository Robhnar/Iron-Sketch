"""
Factory class for creating different CNN model architectures.
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torchvision.models.segmentation import fcn_resnet50
from .deeplabv3plus_resnet50 import create_deeplabv3plus_resnet50
from .huggingface_deeplabv3 import create_huggingface_deeplabv3
from .deeplabv3_mobilenet_google import create_google_deeplabv3
from .canny_edge_detector import create_canny_model


class ModelFactory:
    """Factory for creating segmentation models."""

    SUPPORTED_ARCHITECTURES = ['unet', 'deeplabv3plus', 'fcn8s', 'deeplabv3plus_resnet50', 'deeplabv3_hf', 'deeplabv3_google', 'canny']

    @staticmethod
    def create_model(architecture: str, pretrained: bool = True, **kwargs) -> nn.Module:
        """
        Create a segmentation model by architecture name.

        Args:
            architecture: One of supported architectures
            pretrained: Whether to use pretrained encoder weights
            **kwargs: Additional parameters for specific architectures (e.g., Canny parameters)

        Returns:
            PyTorch model ready for training or inference
        """
        architecture = architecture.lower()

        if architecture not in ModelFactory.SUPPORTED_ARCHITECTURES:
            raise ValueError(
                f"Unsupported architecture: {architecture}. "
                f"Choose from {ModelFactory.SUPPORTED_ARCHITECTURES}"
            )

        if architecture == 'unet':
            return ModelFactory._create_unet(pretrained)
        elif architecture == 'deeplabv3plus':
            return ModelFactory._create_deeplabv3plus(pretrained)
        elif architecture == 'fcn8s':
            return ModelFactory._create_fcn(pretrained)
        elif architecture == 'deeplabv3plus_resnet50':
            return create_deeplabv3plus_resnet50(pretrained=pretrained)
        elif architecture == 'deeplabv3_hf':
            return create_huggingface_deeplabv3(mode='edge')
        elif architecture == 'deeplabv3_google':
            return create_google_deeplabv3(mode='edge')
        elif architecture == 'canny':
            return create_canny_model(**kwargs)

    @staticmethod
    def _create_unet(pretrained: bool) -> nn.Module:
        """
        Create U-Net with ResNet34 encoder.

        Architecture: U-Net
        Encoder: ResNet34
        Parameters: ~7.7M
        """
        encoder_weights = 'imagenet' if pretrained else None

        model = smp.Unet(
            encoder_name='resnet34',
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None
        )

        return model

    @staticmethod
    def _create_deeplabv3plus(pretrained: bool) -> nn.Module:
        """
        Create DeepLabV3+ with MobileNetV2 encoder.

        Architecture: DeepLabV3+
        Encoder: MobileNetV2
        Parameters: ~2.5M
        """
        encoder_weights = 'imagenet' if pretrained else None

        model = smp.DeepLabV3Plus(
            encoder_name='mobilenet_v2',
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=1,
            activation=None
        )

        return model

    @staticmethod
    def _create_fcn(pretrained: bool) -> nn.Module:
        """
        Create FCN with ResNet50 encoder (similar to FCN-8s).

        Architecture: Fully Convolutional Network
        Encoder: ResNet50
        Parameters: ~14M
        """
        model = fcn_resnet50(pretrained=pretrained, num_classes=1)

        return FCNWrapper(model)

    @staticmethod
    def get_parameter_count(model: nn.Module) -> int:
        """Count total trainable parameters in model."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def get_architecture_info(architecture: str) -> dict:
        """Get information about a specific architecture."""
        info = {
            'unet': {
                'name': 'U-Net',
                'encoder': 'ResNet34',
                'params': '~7.7M',
                'description': 'Encoder-decoder with skip connections',
                'best_for': 'Balanced performance and accuracy'
            },
            'deeplabv3plus': {
                'name': 'DeepLabV3+',
                'encoder': 'MobileNetV2',
                'params': '~2.5M',
                'description': 'ASPP with lightweight encoder',
                'best_for': 'Fast inference and mobile deployment'
            },
            'fcn8s': {
                'name': 'FCN-ResNet50',
                'encoder': 'ResNet50',
                'params': '~14M',
                'description': 'Fully convolutional with deep encoder',
                'best_for': 'High accuracy on complex patterns'
            },
            'deeplabv3plus_resnet50': {
                'name': 'DeepLabV3+ ResNet50',
                'encoder': 'ResNet50',
                'params': '~40M',
                'description': 'ASPP with deep ResNet50 encoder and decoder',
                'best_for': 'Complex patterns with multi-scale context'
            },
            'deeplabv3_hf': {
                'name': 'SegFormer (Hugging Face)',
                'encoder': 'MiT-B0',
                'params': '~3.7M',
                'description': 'Pre-trained transformer-based segmentation',
                'best_for': 'Quick inference without training'
            },
            'deeplabv3_google': {
                'name': 'DeepLabV3 MobileNetV2 (Google)',
                'encoder': 'MobileNetV2',
                'params': '~2.1M',
                'description': 'Google pre-trained on PASCAL VOC with 21 classes',
                'best_for': 'Object detection and edge extraction'
            },
            'canny': {
                'name': 'Canny Edge Detection',
                'encoder': 'Classical CV',
                'params': '0 (algorithm)',
                'description': 'Tunable edge detection with Gaussian blur and hysteresis',
                'best_for': 'Fast edge detection, clear boundaries, no training required'
            }
        }

        return info.get(architecture.lower(), {})


class FCNWrapper(nn.Module):
    """Wrapper to make FCN output compatible with other models."""

    def __init__(self, fcn_model):
        super().__init__()
        self.model = fcn_model

    def forward(self, x):
        output = self.model(x)
        return output['out']
