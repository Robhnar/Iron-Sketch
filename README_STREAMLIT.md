# Canny Edge Detection + Robot Welding Script Generator

A complete Python Streamlit web application that processes images through a full Canny edge detection pipeline and converts the resulting edges into robot welding scripts.

## Features

### Core Functionality
- **Image Upload**: Support for JPG/PNG with live preview
- **Preprocessing**: Optional resize (scale to specified dimension) + grayscale conversion
- **Custom Canny Implementation**:
  - Gaussian blur with configurable kernel size & sigma
  - Sobel gradients (Gx, Gy)
  - Magnitude & direction calculation
  - Manual Non-Maximum Suppression (NMS)
  - Double thresholding with normalized [0,1] mode
  - Hysteresis with 8-connectivity
- **Visualization**: Side-by-side display of all intermediate steps including HSV color direction map
- **Edge Vectorization**: Convert binary edges to vector paths using OpenCV contours
- **Robot Script Generation**: Output `.script` file with MoveL commands

### UI Controls (Sidebar)
- Resize dimension (0 = no resize, default 512)
- Gaussian kernel size (3, 5, 7, 9, 11)
- Gaussian sigma (0.5–5.0)
- Low threshold [0.0–1.0] (normalized)
- High threshold [0.0–1.0] (normalized)
- Robot workspace scaling (mm per pixel)
- Origin offset (X, Y in mm)
- Z height for welding
- Simplification epsilon (Douglas-Peucker)
- Toggle: Use custom edges vs OpenCV edges

### Output
- All intermediate images (PNG format)
- Robot welding script (`14_vector_edges.script`)
- Parameters file (`_params.txt`)
- ZIP download with all files

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the Streamlit app
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## How It Works

1. **Upload an image** (JPG/PNG)
2. **Configure parameters** in the sidebar:
   - Adjust preprocessing (resize, blur)
   - Set Canny thresholds (normalized 0-1 range)
   - Configure robot workspace mapping
3. **Click "Run Pipeline"**
4. **View results**:
   - All intermediate processing steps
   - Custom vs OpenCV Canny comparison
   - Vectorized paths info
5. **Download**:
   - Individual files (script, params)
   - Complete ZIP package

## Robot Script Format

Generated `.script` files use this format:

```
MoveL [[x_mm,y_mm,z_mm],[0,0,1,0]], v100, z10, tool0\WObj:=wobj0;
```

Where coordinates are mapped from image pixels to robot workspace using:
- `x_mm = x_pixel * scale_mm_per_px + origin_x_offset`
- `y_mm = y_pixel * scale_mm_per_px + origin_y_offset`
- `z_mm = fixed_z_height`

## Key Implementation Details

- **Manual NMS**: Implements 8-directional non-maximum suppression
- **Hysteresis**: Uses BFS with 8-connectivity for edge linking
- **Direction Visualization**: HSV color map (H=angle, V=magnitude)
- **Vectorization**: OpenCV contours + Douglas-Peucker simplification
- **No Database Required**: All processing done in-memory

## Parameters Guide

### Resize Dimension
- Smaller values = faster processing, less detail
- Larger values = more detail, slower processing
- 512 is a good default

### Gaussian Blur
- **Kernel size**: Larger = more blur, fewer noise edges
- **Sigma**: Higher = smoother, removes fine details

### Canny Thresholds
- **Low (0.1)**: Minimum edge strength to consider
- **High (0.3)**: Strong edge threshold
- Rule: high should be 2-3x low value

### Robot Workspace
- **mm/pixel**: Physical size of one pixel (e.g., 1.0 = 1mm per pixel)
- **Origin offsets**: Align image center with robot coordinate system
- **Z height**: Fixed welding height above workpiece

## Troubleshooting

**Issue**: No edges detected
- Increase low/high thresholds
- Reduce Gaussian blur (smaller kernel/sigma)

**Issue**: Too many noisy edges
- Decrease low threshold
- Increase Gaussian blur

**Issue**: Script has too many points
- Increase simplification epsilon (e.g., 2.0 or 3.0)
- Use smaller resize dimension

## Technical Stack

- **Streamlit**: Web application framework
- **OpenCV**: Image processing & contour detection
- **NumPy**: Numerical operations
- **Python 3.8+**: Required

## License

This is a standalone application for educational and industrial use.
