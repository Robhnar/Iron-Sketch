export const TARGET_WIDTH = 256;
export const TARGET_HEIGHT = 384;

export interface ProcessedImage {
  dataUrl: string;
  width: number;
  height: number;
  blob: Blob;
}

export async function resizeImageToTarget(file: File): Promise<ProcessedImage> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const reader = new FileReader();

    reader.onload = (e) => {
      img.src = e.target?.result as string;
    };

    img.onload = () => {
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d');

      if (!ctx) {
        reject(new Error('Could not get canvas context'));
        return;
      }

      canvas.width = TARGET_WIDTH;
      canvas.height = TARGET_HEIGHT;

      const imgAspect = img.width / img.height;
      const targetAspect = TARGET_WIDTH / TARGET_HEIGHT;

      let sx = 0, sy = 0, sw = img.width, sh = img.height;

      if (imgAspect > targetAspect) {
        sw = img.height * targetAspect;
        sx = (img.width - sw) / 2;
      } else {
        sh = img.width / targetAspect;
        sy = (img.height - sh) / 2;
      }

      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, TARGET_WIDTH, TARGET_HEIGHT);
      ctx.drawImage(img, sx, sy, sw, sh, 0, 0, TARGET_WIDTH, TARGET_HEIGHT);

      canvas.toBlob((blob) => {
        if (!blob) {
          reject(new Error('Failed to create blob'));
          return;
        }

        resolve({
          dataUrl: canvas.toDataURL('image/png'),
          width: TARGET_WIDTH,
          height: TARGET_HEIGHT,
          blob: blob
        });
      }, 'image/png');
    };

    img.onerror = () => reject(new Error('Failed to load image'));
    reader.onerror = () => reject(new Error('Failed to read file'));
    reader.readAsDataURL(file);
  });
}

export async function createBinaryMask(imageData: ImageData): Promise<ImageData> {
  const data = imageData.data;

  for (let i = 0; i < data.length; i += 4) {
    const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
    const binary = avg > 127 ? 255 : 0;
    data[i] = binary;
    data[i + 1] = binary;
    data[i + 2] = binary;
    data[i + 3] = 255;
  }

  return imageData;
}

export function applyMorphologicalClosing(imageData: ImageData, kernelSize: number = 3): ImageData {
  const { width, height, data } = imageData;
  const output = new ImageData(width, height);
  const half = Math.floor(kernelSize / 2);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let maxVal = 0;

      for (let ky = -half; ky <= half; ky++) {
        for (let kx = -half; kx <= half; kx++) {
          const ny = y + ky;
          const nx = x + kx;

          if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
            const idx = (ny * width + nx) * 4;
            maxVal = Math.max(maxVal, data[idx]);
          }
        }
      }

      const idx = (y * width + x) * 4;
      output.data[idx] = maxVal;
      output.data[idx + 1] = maxVal;
      output.data[idx + 2] = maxVal;
      output.data[idx + 3] = 255;
    }
  }

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let minVal = 255;

      for (let ky = -half; ky <= half; ky++) {
        for (let kx = -half; kx <= half; kx++) {
          const ny = y + ky;
          const nx = x + kx;

          if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
            const idx = (ny * width + nx) * 4;
            minVal = Math.min(minVal, output.data[idx]);
          }
        }
      }

      const idx = (y * width + x) * 4;
      output.data[idx] = minVal;
      output.data[idx + 1] = minVal;
      output.data[idx + 2] = minVal;
      output.data[idx + 3] = 255;
    }
  }

  return output;
}

export async function simulateAIInference(inputImage: ProcessedImage): Promise<ImageData> {
  return new Promise((resolve) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    if (!ctx) throw new Error('Could not get canvas context');

    canvas.width = TARGET_WIDTH;
    canvas.height = TARGET_HEIGHT;

    const img = new Image();
    img.onload = () => {
      ctx.drawImage(img, 0, 0);
      let imageData = ctx.getImageData(0, 0, TARGET_WIDTH, TARGET_HEIGHT);

      const data = imageData.data;
      for (let i = 0; i < data.length; i += 4) {
        const brightness = (data[i] + data[i + 1] + data[i + 2]) / 3;
        const threshold = 100 + Math.sin(i / 1000) * 50;
        const isEdge = brightness > threshold;

        const value = isEdge ? 255 : 0;
        data[i] = value;
        data[i + 1] = value;
        data[i + 2] = value;
        data[i + 3] = 255;
      }

      imageData = applyMorphologicalClosing(imageData);
      resolve(imageData);
    };

    img.src = inputImage.dataUrl;
  });
}
