"""
Utilities for model management, saving, loading, and device detection.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile


def get_device() -> torch.device:
    """
    Detect and return the best available device (CUDA or CPU).

    Returns:
        torch.device: CUDA device if available, otherwise CPU
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("✓ Using CPU (GPU not available)")

    return device


def save_model(
    model: nn.Module,
    filepath: str,
    architecture: str,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save model checkpoint with metadata.

    Args:
        model: PyTorch model to save
        filepath: Path to save checkpoint
        architecture: Model architecture name
        metadata: Additional metadata to save
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'architecture': architecture,
        'metadata': metadata or {}
    }

    torch.save(checkpoint, filepath)
    print(f"✓ Model saved to {filepath}")


def load_model(
    filepath: str,
    model: nn.Module,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Load model checkpoint from file.

    Args:
        filepath: Path to checkpoint file
        model: Model instance to load weights into
        device: Device to load model to

    Returns:
        Model with loaded weights
    """
    if device is None:
        device = get_device()

    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"✓ Model loaded from {filepath}")
    return model


def load_model_from_bytes(
    model_bytes: bytes,
    model: nn.Module,
    device: Optional[torch.device] = None
) -> nn.Module:
    """
    Load model checkpoint from bytes (e.g., downloaded from storage).

    Args:
        model_bytes: Model checkpoint as bytes
        model: Model instance to load weights into
        device: Device to load model to

    Returns:
        Model with loaded weights
    """
    if device is None:
        device = get_device()

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
        tmp_file.write(model_bytes)
        tmp_path = tmp_file.name

    try:
        checkpoint = torch.load(tmp_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print("✓ Model loaded from bytes")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return model


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get information about a model.

    Args:
        model: PyTorch model

    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info = {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'parameter_count_formatted': format_parameter_count(total_params),
        'memory_size_mb': total_params * 4 / (1024 ** 2)
    }

    return info


def format_parameter_count(count: int) -> str:
    """Format parameter count in human-readable format."""
    if count >= 1_000_000:
        return f"{count / 1_000_000:.1f}M"
    elif count >= 1_000:
        return f"{count / 1_000:.1f}K"
    else:
        return str(count)


def model_to_bytes(model: nn.Module, architecture: str, metadata: Optional[Dict] = None) -> bytes:
    """
    Convert model to bytes for upload to storage.

    Args:
        model: PyTorch model
        architecture: Model architecture name
        metadata: Additional metadata

    Returns:
        Model checkpoint as bytes
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'architecture': architecture,
        'metadata': metadata or {}
    }

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pth') as tmp_file:
        torch.save(checkpoint, tmp_file.name)
        tmp_path = tmp_file.name

    try:
        with open(tmp_path, 'rb') as f:
            model_bytes = f.read()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return model_bytes
