/**
 * Client-side Canny edge detection utilities
 * Uses Canvas API for image processing
 */

export interface CannyParams {
  lowThreshold: number;
  highThreshold: number;
  kernelSize: number;
  sigma: number;
}

/**
 * Apply Canny edge detection to an image using Canvas API
 * This is a simplified version for client-side preview
 */
export async function applyCannyEdgeDetection(
  imageUrl: string,
  params: CannyParams
): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';

    img.onload = () => {
      try {
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');

        if (!ctx) {
          reject(new Error('Could not get canvas context'));
          return;
        }

        ctx.drawImage(img, 0, 0);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);

        const edges = simpleEdgeDetection(imageData, params);

        ctx.putImageData(edges, 0, 0);
        resolve(canvas.toDataURL('image/png'));
      } catch (error) {
        reject(error);
      }
    };

    img.onerror = () => reject(new Error('Failed to load image'));
    img.src = imageUrl;
  });
}

/**
 * Simplified edge detection for client-side preview
 * Uses Sobel operator for gradient calculation
 */
function simpleEdgeDetection(
  imageData: ImageData,
  params: CannyParams
): ImageData {
  const { width, height, data } = imageData;
  const gray = new Uint8ClampedArray(width * height);

  // Convert to grayscale
  for (let i = 0; i < data.length; i += 4) {
    const idx = i / 4;
    gray[idx] = Math.round(
      0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]
    );
  }

  // Apply Gaussian blur (simplified)
  const blurred = gaussianBlur(gray, width, height, params.sigma);

  // Calculate gradients using Sobel operator
  const { magnitude, direction } = sobelGradient(blurred, width, height);

  // Non-maximum suppression
  const suppressed = nonMaximumSuppression(magnitude, direction, width, height);

  // Double threshold and hysteresis
  const edges = doubleThresholdHysteresis(
    suppressed,
    width,
    height,
    params.lowThreshold * 255,
    params.highThreshold * 255
  );

  // Convert back to ImageData
  const result = new ImageData(width, height);
  for (let i = 0; i < edges.length; i++) {
    const pixelIdx = i * 4;
    result.data[pixelIdx] = edges[i];
    result.data[pixelIdx + 1] = edges[i];
    result.data[pixelIdx + 2] = edges[i];
    result.data[pixelIdx + 3] = 255;
  }

  return result;
}

/**
 * Simplified Gaussian blur
 */
function gaussianBlur(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  sigma: number
): Float32Array {
  const result = new Float32Array(width * height);
  const kernelSize = Math.max(3, Math.ceil(sigma * 3) * 2 + 1);
  const kernel = createGaussianKernel(kernelSize, sigma);
  const half = Math.floor(kernelSize / 2);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      let weightSum = 0;

      for (let ky = -half; ky <= half; ky++) {
        for (let kx = -half; kx <= half; kx++) {
          const ny = y + ky;
          const nx = x + kx;

          if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
            const weight = kernel[ky + half][kx + half];
            sum += data[ny * width + nx] * weight;
            weightSum += weight;
          }
        }
      }

      result[y * width + x] = sum / weightSum;
    }
  }

  return result;
}

/**
 * Create Gaussian kernel
 */
function createGaussianKernel(size: number, sigma: number): number[][] {
  const kernel: number[][] = [];
  const half = Math.floor(size / 2);
  const sigma2 = 2 * sigma * sigma;

  for (let y = -half; y <= half; y++) {
    const row: number[] = [];
    for (let x = -half; x <= half; x++) {
      const value = Math.exp(-(x * x + y * y) / sigma2);
      row.push(value);
    }
    kernel.push(row);
  }

  return kernel;
}

/**
 * Sobel gradient calculation
 */
function sobelGradient(
  data: Float32Array,
  width: number,
  height: number
): { magnitude: Float32Array; direction: Float32Array } {
  const magnitude = new Float32Array(width * height);
  const direction = new Float32Array(width * height);

  const sobelX = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
  const sobelY = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      let gx = 0;
      let gy = 0;

      for (let ky = -1; ky <= 1; ky++) {
        for (let kx = -1; kx <= 1; kx++) {
          const pixel = data[(y + ky) * width + (x + kx)];
          gx += pixel * sobelX[ky + 1][kx + 1];
          gy += pixel * sobelY[ky + 1][kx + 1];
        }
      }

      const idx = y * width + x;
      magnitude[idx] = Math.sqrt(gx * gx + gy * gy);
      direction[idx] = Math.atan2(gy, gx);
    }
  }

  return { magnitude, direction };
}

/**
 * Non-maximum suppression
 */
function nonMaximumSuppression(
  magnitude: Float32Array,
  direction: Float32Array,
  width: number,
  height: number
): Float32Array {
  const result = new Float32Array(width * height);

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const idx = y * width + x;
      const angle = direction[idx];
      const mag = magnitude[idx];

      // Quantize angle to 4 directions
      let angle45 = ((angle * 180 / Math.PI) + 180) % 180;
      let q = 255, r = 255;

      if ((angle45 >= 0 && angle45 < 22.5) || (angle45 >= 157.5 && angle45 <= 180)) {
        q = magnitude[y * width + (x + 1)];
        r = magnitude[y * width + (x - 1)];
      } else if (angle45 >= 22.5 && angle45 < 67.5) {
        q = magnitude[(y - 1) * width + (x + 1)];
        r = magnitude[(y + 1) * width + (x - 1)];
      } else if (angle45 >= 67.5 && angle45 < 112.5) {
        q = magnitude[(y - 1) * width + x];
        r = magnitude[(y + 1) * width + x];
      } else if (angle45 >= 112.5 && angle45 < 157.5) {
        q = magnitude[(y - 1) * width + (x - 1)];
        r = magnitude[(y + 1) * width + (x + 1)];
      }

      if (mag >= q && mag >= r) {
        result[idx] = mag;
      }
    }
  }

  return result;
}

/**
 * Double threshold and hysteresis
 */
function doubleThresholdHysteresis(
  data: Float32Array,
  width: number,
  height: number,
  lowThreshold: number,
  highThreshold: number
): Uint8ClampedArray {
  const result = new Uint8ClampedArray(width * height);
  const STRONG = 255;
  const WEAK = 75;

  // Apply thresholds
  for (let i = 0; i < data.length; i++) {
    if (data[i] >= highThreshold) {
      result[i] = STRONG;
    } else if (data[i] >= lowThreshold) {
      result[i] = WEAK;
    }
  }

  // Hysteresis: connect weak edges to strong edges
  const queue: number[] = [];
  for (let i = 0; i < result.length; i++) {
    if (result[i] === STRONG) {
      queue.push(i);
    }
  }

  while (queue.length > 0) {
    const idx = queue.shift()!;
    const y = Math.floor(idx / width);
    const x = idx % width;

    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (dx === 0 && dy === 0) continue;

        const ny = y + dy;
        const nx = x + dx;

        if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
          const nidx = ny * width + nx;
          if (result[nidx] === WEAK) {
            result[nidx] = STRONG;
            queue.push(nidx);
          }
        }
      }
    }
  }

  // Remove weak edges
  for (let i = 0; i < result.length; i++) {
    if (result[i] !== STRONG) {
      result[i] = 0;
    }
  }

  return result;
}

/**
 * Generate example Canny parameters presets
 */
export const CANNY_PRESETS = {
  strict: {
    name: 'Strict (High Threshold)',
    lowThreshold: 0.2,
    highThreshold: 0.5,
    kernelSize: 5,
    sigma: 1.4
  },
  balanced: {
    name: 'Balanced (Default)',
    lowThreshold: 0.1,
    highThreshold: 0.3,
    kernelSize: 5,
    sigma: 1.4
  },
  sensitive: {
    name: 'Sensitive (Low Threshold)',
    lowThreshold: 0.05,
    highThreshold: 0.15,
    kernelSize: 5,
    sigma: 1.4
  },
  smooth: {
    name: 'Smooth (High Blur)',
    lowThreshold: 0.1,
    highThreshold: 0.3,
    kernelSize: 9,
    sigma: 2.5
  }
};
