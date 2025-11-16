#!/usr/bin/env python3
"""
Edge-to-Motion ML Model

Architecture:
  Input: binary edge image (B, 1, H, W)
  ↓ CNN Encoder (ResNet18 backbone)
  ↓ (B, 512, H//32, W//32)
  ↓ Spatial Attention + Flattening
  ↓ LSTM Decoder (variable-length sequence)
  ↓ Output: sequence of moveLinear commands [(Δx, Δy, Δz, speed, is_lift), ...]

Training:
  Loss = L1(predicted_params, target_params) + BCE(is_lift)
  Optimizer: Adam
  
Usage:
  # Training
  python ml_edge_to_motion.py train --data training_data.jsonl --epochs 50 --batch-size 4 --lr 1e-3 --output model.pt
  
  # Inference (on new edge image)
  python ml_edge_to_motion.py infer --model model.pt --image edge_image.png --output commands.json
"""
import json
import argparse
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import models
import cv2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")


# ==================== Dataset ====================

class EdgeMotionDataset(Dataset):
    """Load image + command sequence pairs from JSONL."""
    
    def __init__(self, jsonl_path: Path, root_dir: Path | None = None, img_size: int = 256):
        self.jsonl_path = jsonl_path
        self.root_dir = root_dir or jsonl_path.parent
        self.img_size = img_size
        
        self.pairs = []
        with open(jsonl_path) as f:
            for line in f:
                try:
                    pair = json.loads(line.strip())
                    self.pairs.append(pair)
                except Exception as e:
                    logger.warning(f"Skipped line: {e}")
        
        logger.info(f"Loaded {len(self.pairs)} training pairs from {jsonl_path}")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx: int):
        pair = self.pairs[idx]
        
        # Load image
        img_path = self.root_dir / pair['image_path']
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            logger.warning(f"Failed to load {img_path}, using blank")
            img = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        else:
            # Resize
            h, w = img.shape
            scale = self.img_size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h))
            # Pad to img_size
            pad_h = self.img_size - new_h
            pad_w = self.img_size - new_w
            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
        
        # Normalize to [0, 1]
        img = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        
        # Encode commands into fixed-size tensor
        commands = pair['commands']
        cmd_tensor = self._encode_commands(commands, max_len=256)
        cmd_mask = self._make_command_mask(len(commands), max_len=256)
        
        return {
            'image': img,
            'commands': cmd_tensor,
            'mask': cmd_mask,
            'meta': pair.get('metadata', {})
        }
    
    def _encode_commands(self, commands: List[Dict], max_len: int = 256) -> torch.Tensor:
        """Encode command list to (max_len, 5) tensor: [Δx, Δy, Δz, speed, is_lift]."""
        # Get baseline offsets from first command
        x0 = commands[0]['x'] if commands else 0
        y0 = commands[0]['y'] if commands else 0
        z0 = commands[0]['z'] if commands else 0
        
        encoded = []
        prev_x, prev_y, prev_z = x0, y0, z0
        
        for cmd in commands[:max_len]:
            dx = cmd['x'] - prev_x
            dy = cmd['y'] - prev_y
            dz = cmd['z'] - prev_z
            speed = cmd['speed']
            is_lift = float(cmd['is_lift'])
            
            encoded.append([dx, dy, dz, speed, is_lift])
            prev_x, prev_y, prev_z = cmd['x'], cmd['y'], cmd['z']
        
        # Pad to max_len
        encoded = np.array(encoded, dtype=np.float32)
        if len(encoded) < max_len:
            pad = np.zeros((max_len - len(encoded), 5), dtype=np.float32)
            encoded = np.vstack([encoded, pad])
        
        return torch.from_numpy(encoded)
    
    def _make_command_mask(self, actual_len: int, max_len: int = 256) -> torch.Tensor:
        """Mask for actual vs padded commands."""
        mask = torch.ones(max_len, dtype=torch.float32)
        mask[actual_len:] = 0
        return mask


# ==================== Model ====================

class EdgeMotionModel(nn.Module):
    """
    CNN encoder + LSTM decoder for edge image → motion sequence.
    """
    
    def __init__(self, lstm_hidden: int = 256, seq_len: int = 256):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.seq_len = seq_len
        
        # ResNet18 encoder (pretrained)
        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.AdaptiveAvgPool2d((4, 4))  # (B, 512, 4, 4)
        )
        
        # Flatten + FC to LSTM init
        self.fc_init = nn.Linear(512 * 4 * 4, lstm_hidden)
        
        # LSTM decoder
        self.lstm = nn.LSTM(input_size=5, hidden_size=lstm_hidden, num_layers=2, batch_first=True)
        
        # Output head: predict (Δx, Δy, Δz, speed, is_lift)
        self.output_head = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # [dx, dy, dz, speed, is_lift]
        )
    
    def forward(self, img: torch.Tensor, teacher_forcing: bool = False, teacher_input: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            img: (B, 1, H, W) edge image
            teacher_forcing: use ground truth for LSTM input (training)
            teacher_input: (B, seq_len, 5) ground truth commands
        
        Returns:
            output: (B, seq_len, 5) predicted commands
        """
        B = img.size(0)
        
        # Encode image
        enc = self.encoder(img)  # (B, 512, 4, 4)
        enc_flat = enc.view(B, -1)  # (B, 8192)
        h0 = torch.tanh(self.fc_init(enc_flat))  # (B, lstm_hidden)
        c0 = torch.zeros_like(h0)
        
        # Decode with LSTM
        output = []
        if teacher_forcing and teacher_input is not None:
            # Teacher forcing: use ground truth input
            lstm_out, _ = self.lstm(teacher_input, (h0.unsqueeze(0).expand(2, -1, -1), c0.unsqueeze(0).expand(2, -1, -1)))
            output = self.output_head(lstm_out)
        else:
            # Autoregressive: predict one step at a time
            cur_input = torch.zeros(B, 5, device=img.device)
            hx = (h0.unsqueeze(0).expand(2, -1, -1), c0.unsqueeze(0).expand(2, -1, -1))
            
            for t in range(self.seq_len):
                lstm_out, hx = self.lstm(cur_input.unsqueeze(1), hx)  # (B, 1, lstm_hidden)
                pred = self.output_head(lstm_out.squeeze(1))  # (B, 5)
                output.append(pred)
                cur_input = pred.detach()  # Use prediction for next step
        
        return torch.stack(output, dim=1) if isinstance(output, list) else output


# ==================== Training ====================

def train(
    data_path: Path,
    root_dir: Path | None = None,
    epochs: int = 50,
    batch_size: int = 4,
    lr: float = 1e-3,
    output_model: Path | None = None
):
    """Train the model."""
    output_model = output_model or Path("ml_edge_to_motion.pt")
    
    # Dataset & DataLoader
    dataset = EdgeMotionDataset(data_path, root_dir=root_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Model
    model = EdgeMotionModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn_l1 = nn.L1Loss()
    loss_fn_bce = nn.BCEWithLogitsLoss()
    
    logger.info(f"Starting training: {epochs} epochs, batch_size={batch_size}, lr={lr}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            img = batch['image'].to(device)
            cmd_true = batch['commands'].to(device)
            mask = batch['mask'].to(device)
            
            # Forward pass (teacher forcing)
            optimizer.zero_grad()
            pred = model(img, teacher_forcing=True, teacher_input=cmd_true)
            
            # Loss: L1 on [dx, dy, dz, speed], BCE on [is_lift]
            mask_expand = mask.unsqueeze(-1)  # (B, seq_len, 1)
            
            loss_param = loss_fn_l1(pred[..., :4] * mask_expand, cmd_true[..., :4] * mask_expand)
            loss_lift = loss_fn_bce(pred[..., 4] * mask.squeeze(-1), cmd_true[..., 4] * mask.squeeze(-1))
            loss = loss_param + 0.5 * loss_lift
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        logger.info(f"Epoch {epoch+1} - Loss: {avg_loss:.6f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss
            }, output_model)
            logger.info(f"Checkpoint saved to {output_model}")
    
    torch.save(model.state_dict(), output_model)
    logger.info(f"Training complete. Model saved to {output_model}")


def infer(model_path: Path, image_path: Path, output_path: Path | None = None):
    """Inference: edge image → commands."""
    output_path = output_path or Path(image_path.stem + "_commands.json")
    
    # Load model
    model = EdgeMotionModel().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Load image
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.error(f"Failed to load {image_path}")
        return
    
    # Resize and normalize
    img_size = 256
    h, w = img.shape
    scale = img_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    img = cv2.resize(img, (new_w, new_h))
    pad_h, pad_w = img_size - new_h, img_size - new_w
    img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0
    
    # Inference
    with torch.no_grad():
        pred = model(img_tensor, teacher_forcing=False)
    
    pred_np = pred[0].cpu().numpy()  # (seq_len, 5)
    
    # Decode back to absolute positions
    commands = []
    x, y, z = -825.0, -115.0, -363.7  # Default offsets
    
    for i, (dx, dy, dz, speed, is_lift_raw) in enumerate(pred_np):
        if np.allclose([dx, dy, dz, speed], 0):  # Padding
            break
        
        x += dx
        y += dy
        z += dz
        is_lift = bool(is_lift_raw > 0.5)
        
        commands.append({
            "cmd": "moveLinear",
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "rx": 180.0,
            "ry": 0.0,
            "rz": 90.0,
            "speed": max(10, float(speed)),
            "is_lift": is_lift
        })
    
    with open(output_path, 'w') as f:
        json.dump(commands, f, indent=2)
    
    logger.info(f"Generated {len(commands)} commands → {output_path}")


def main():
    ap = argparse.ArgumentParser()
    subparsers = ap.add_subparsers(dest='cmd')
    
    # Train
    train_p = subparsers.add_parser('train', help='Train the model')
    train_p.add_argument('--data', type=Path, required=True, help='JSONL training data')
    train_p.add_argument('--root', type=Path, default=None, help='Root directory for image paths')
    train_p.add_argument('--epochs', type=int, default=50)
    train_p.add_argument('--batch-size', type=int, default=4)
    train_p.add_argument('--lr', type=float, default=1e-3)
    train_p.add_argument('--output', type=Path, default=Path('ml_edge_to_motion.pt'))
    
    # Infer
    infer_p = subparsers.add_parser('infer', help='Inference on new image')
    infer_p.add_argument('--model', type=Path, required=True, help='Trained model checkpoint')
    infer_p.add_argument('--image', type=Path, required=True, help='Edge image')
    infer_p.add_argument('--output', type=Path, default=None)
    
    args = ap.parse_args()
    
    if args.cmd == 'train':
        train(args.data, args.root, args.epochs, args.batch_size, args.lr, args.output)
    elif args.cmd == 'infer':
        infer(args.model, args.image, args.output)
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
