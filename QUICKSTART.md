# Quick Start Guide - AI Welding Path Generator - IronSketch

## 🚀 Getting Started in 5 Minutes

### Step 1: Install Dependencies (2 minutes)

```bash
pip install -r requirements.txt
```

This installs PyTorch, Streamlit, OpenCV, and all required ML libraries. If you have CUDA-compatible GPU, PyTorch will automatically use it for faster training/inference.

### Step 2: Generate Sample Dataset (1 minute)

```bash
python3 create_sample_dataset.py
```

This creates `sample_dataset/` with 10 synthetic welding patterns:
- `input/` - Grayscale images with lines, circles, and patterns
- `target/` - Binary masks showing welding paths

### Step 3: Launch Application (30 seconds)

```bash
streamlit run app.py
```

The browser will automatically open to `http://localhost:8501`

### Step 4: Create Your First Dataset (1 minute)

1. Click **"Dataset Builder"** tab
2. Click **"Create Dataset"** sub-tab
3. Upload all files from `sample_dataset/input/` as **Input Images**
4. Upload all files from `sample_dataset/target/` as **Target Masks**
5. Click **"Create Dataset"** button
6. Wait for upload to complete (progress bar shows status)

### Step 5: Train Your First Model (5-10 minutes)

1. Click **"Train Models"** tab
2. Select architecture: **U-Net** (recommended for first try)
3. Enter model name: `my_first_model`
4. Select your dataset from dropdown
5. Set epochs: **10** (quick test) or **20** (better results)
6. Keep other defaults (batch=4, lr=0.001, optimizer=Adam)
7. Click **"Start Training"**
8. Watch real-time progress with loss curves and metrics

### Step 6: Generate Welding Paths (30 seconds)

1. Click **"Generate Paths"** tab
2. Upload a test image (use one from `sample_dataset/input/`)
3. Select your trained model
4. Configure parameters:
   - Speed: 20 mm/s
   - Scale: 0.5 mm/pixel
   - Z-height: 50 mm
5. Click **"Generate Paths"**
6. View results: original → AI mask → vector overlay
7. Download ABB script or G-code

## ✅ Verification

If everything works, you should see:
- ✓ Dataset created with 10 images (8 train, 2 val)
- ✓ Training completes in 5-10 minutes
- ✓ Model achieves >0.7 IoU score
- ✓ Generated paths match input patterns
- ✓ Robot script downloads successfully

## 🎯 Next Steps

### Use Your Own Images

1. **Take photos** of parts/patterns you want to weld
2. **Create masks** using any image editor:
   - Use white (255) for weld paths
   - Use black (0) for background
   - Save as PNG or JPG
   - Match dimensions: 256×384 pixels (or will auto-resize)
3. **Upload** via Dataset Builder
4. **Train** new model on your custom data

### Optimize Performance

**If training is slow:**
- Reduce batch size (try 2 instead of 4)
- Use DeepLabV3+ (fastest architecture)
- Check if CUDA GPU is detected (look for "Using GPU" message)

**If results are poor:**
- Train longer (30-50 epochs)
- Add more training images (aim for 50+ pairs)
- Try U-Net architecture (best accuracy)
- Check mask quality (clean edges, correct binary values)

**If masks are noisy:**
- Increase closing kernel (5 or 7)
- Increase min contour area (200-300 px)
- Enable skeletonization for thin lines

## 📊 Understanding Metrics

**During Training:**
- **Train Loss**: Lower is better (target: <0.1)
- **Val Loss**: Should decrease with train loss
- **IoU**: Path overlap accuracy (target: >0.7)
- **Dice**: Similar to IoU (target: >0.8)

**If val loss increases while train loss decreases:**
- Model is overfitting
- Reduce epochs
- Add more training data
- Increase augmentation

## 🔧 Robot Integration

### ABB Robot
1. Download `.script` file
2. Upload to robot controller
3. Load in RobotStudio
4. Verify coordinate system matches your setup
5. Run in test mode first

### CNC Plasma Cutter
1. Download `.nc` (G-code) file
2. Load into CNC controller
3. Check feed rate (default: speed × 60 mm/min)
4. Verify Z-clearance height
5. Test on scrap material first

## 🐛 Common Issues

**"No module named 'torch'"**
```bash
pip install torch torchvision
```

**"Failed to initialize Supabase"**
- Check `.env` file exists
- Verify VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY are set

**"No models available"**
- Train a model first in "Train Models" tab
- Wait for training to complete
- Check Supabase storage bucket permissions

**Training crashes with "CUDA out of memory"**
- Reduce batch size to 2 or 1
- Close other GPU applications
- Use CPU mode (automatic fallback)

**Downloaded script is empty**
- Check if paths were generated (should show metrics)
- Verify mask has white pixels (255 value)
- Try increasing min contour area slider

## 💡 Tips for Best Results

1. **Clean masks**: Use pure white (255) and black (0), no gray values
2. **Consistent lighting**: Keep input images well-lit and focused
3. **Sufficient data**: Minimum 20 pairs, ideal 50+
4. **Balanced dataset**: Include variety of patterns/angles
5. **Train longer**: 20-30 epochs for good results, 50+ for production
6. **Test incrementally**: Start with simple patterns, add complexity gradually
7. **Validate coordinates**: Test robot scripts in simulation first

## 📚 Example Workflows

### Workflow 1: Simple Line Welding
1. Create dataset with straight line images
2. Train U-Net for 20 epochs
3. Generate paths with speed=15 mm/s (thick lines)
4. Download ABB script
5. Deploy to robot

### Workflow 2: Complex Pattern Cutting
1. Create dataset with curved/intricate patterns
2. Train FCN-ResNet50 for 40 epochs (highest accuracy)
3. Enable skeletonization for thin lines
4. Set higher simplification epsilon (2.0)
5. Generate G-code for plasma cutter
6. Test on scrap metal

### Workflow 3: Batch Production
1. Train model on production parts
2. Go to "Batch Processing" tab
3. Upload 50+ photos
4. Process batch
5. Download all scripts
6. Load into robot queue

## 🎓 Learning Resources

**Understanding architectures:**
- U-Net: Best for balanced speed/accuracy
- DeepLabV3+: Best for real-time inference
- FCN-ResNet50: Best for complex patterns

**Key parameters:**
- Learning rate: Controls training step size (0.001 is safe default)
- Batch size: Images per training step (smaller = less memory)
- Epochs: Complete passes through dataset (more = better, but diminishing returns)
- Early stopping: Prevents overtraining (5-10 is good)

**Post-processing:**
- Closing kernel: Fills small gaps in paths
- Min area: Removes noise/small contours
- Simplification: Reduces path points (higher = simpler)
- Skeletonization: Makes paths single-pixel wide

## 🆘 Need Help?

1. Check README.md for full documentation
2. Review Troubleshooting section
3. Verify sample dataset works first
4. Check training metrics (IoU should be >0.5)
5. Test with simple images before complex patterns

## ⚡ Performance Benchmarks

**With GPU (NVIDIA RTX 3080):**
- Training: ~10s per epoch (50 images)
- Inference: ~50ms per image
- Recommended for production

**With CPU (Modern Intel i7):**
- Training: ~2min per epoch (50 images)
- Inference: ~200ms per image
- Suitable for development/testing

**Dataset sizes:**
- Small (10-20 pairs): Quick experiments, limited accuracy
- Medium (50-100 pairs): Production-ready for simple patterns
- Large (200+ pairs): High accuracy for complex patterns

---

**Ready to go? Run the Quick Start steps above and you'll have a working AI welding system in 10 minutes!**
