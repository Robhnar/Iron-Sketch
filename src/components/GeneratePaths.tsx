import React, { useState, useEffect } from 'react';
import { Upload, Download, Settings, Zap } from 'lucide-react';
import { listModels, Model } from '../lib/fileStorage';
import { resizeImageToTarget, simulateAIInference, TARGET_WIDTH, TARGET_HEIGHT } from '../utils/imageProcessing';
import { vectorizeMask, createOverlayImage } from '../utils/vectorization';
import { generateABBScript, generateGCode, generateParametersFile, CoordinateConfig } from '../utils/robotScript';

export function GeneratePaths() {
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('');
  const [speed, setSpeed] = useState(20);
  const [config, setConfig] = useState<CoordinateConfig>({
    mmPerPixel: 0.5,
    originOffsetX: 0,
    originOffsetY: 0,
    zHeight: 5
  });

  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [processedImage, setProcessedImage] = useState<any | null>(null);
  const [maskImage, setMaskImage] = useState<string | null>(null);
  const [overlayImage, setOverlayImage] = useState<string | null>(null);
  const [vectorPaths, setVectorPaths] = useState<any>(null);
  const [processing, setProcessing] = useState(false);

  useEffect(() => {
    loadModels();
  }, []);

  async function loadModels() {
    try {
      const data = await listModels();
      setModels(data);
      if (data.length > 0 && !selectedModelId) {
        setSelectedModelId(data[0].id);
      }
    } catch (error) {
      console.error('Error loading models:', error);
    }
  }

  async function handleImageUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    try {
      setProcessing(true);
      const processed = await resizeImageToTarget(file);
      setProcessedImage(processed);
      setUploadedImage(processed.dataUrl);
      setMaskImage(null);
      setOverlayImage(null);
      setVectorPaths(null);
    } catch (error) {
      console.error('Error processing image:', error);
      alert('Failed to process image');
    } finally {
      setProcessing(false);
    }
  }

  async function handleGeneratePaths() {
    if (!processedImage) return;

    try {
      setProcessing(true);
      const startTime = Date.now();

      const maskData = await simulateAIInference(processedImage);

      const canvas = document.createElement('canvas');
      canvas.width = TARGET_WIDTH;
      canvas.height = TARGET_HEIGHT;
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.putImageData(maskData, 0, 0);
        setMaskImage(canvas.toDataURL('image/png'));
      }

      const paths = vectorizeMask(maskData, 50);
      setVectorPaths(paths);

      const overlay = await createOverlayImage(
        uploadedImage!,
        paths.paths,
        TARGET_WIDTH,
        TARGET_HEIGHT
      );
      setOverlayImage(overlay);

      const processingTime = Date.now() - startTime;

      console.log('Processing completed:', {
        model_id: selectedModelId,
        num_paths: paths.totalPaths,
        num_points: paths.totalPoints,
        processing_time_ms: processingTime
      });

    } catch (error) {
      console.error('Error generating paths:', error);
      alert('Failed to generate paths');
    } finally {
      setProcessing(false);
    }
  }

  async function handleDownload() {
    if (!vectorPaths || !uploadedImage || !maskImage || !overlayImage) return;

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    const folderName = `welding_output_${timestamp}`;

    const abbScript = generateABBScript({
      paths: vectorPaths.paths,
      speed,
      config,
      imageHeight: TARGET_HEIGHT
    });

    const gcode = generateGCode({
      paths: vectorPaths.paths,
      speed,
      config,
      imageHeight: TARGET_HEIGHT
    });

    const parameters = generateParametersFile({
      paths: vectorPaths.paths,
      speed,
      config,
      imageHeight: TARGET_HEIGHT
    });

    const pathsJson = JSON.stringify(vectorPaths.paths, null, 2);

    const files = [
      { name: '00_input.png', content: uploadedImage, type: 'image/png' },
      { name: '01_ai_mask.png', content: maskImage, type: 'image/png' },
      { name: '02_vector_overlay.png', content: overlayImage, type: 'image/png' },
      { name: '03_paths.json', content: pathsJson, type: 'application/json' },
      { name: '04_robot_script.js', content: abbScript, type: 'text/javascript' },
      { name: '05_gcode.nc', content: gcode, type: 'text/plain' },
      { name: '_parameters.txt', content: parameters, type: 'text/plain' }
    ];

    for (const file of files) {
      const blob = file.content.startsWith('data:')
        ? await (await fetch(file.content)).blob()
        : new Blob([file.content], { type: file.type });

      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${folderName}_${file.name}`;
      a.click();
      URL.revokeObjectURL(url);
    }
  }

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Generate Welding Paths</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Model
            </label>
            <select
              value={selectedModelId}
              onChange={(e) => setSelectedModelId(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              {models.length === 0 ? (
                <option value="">No models available (Demo mode)</option>
              ) : (
                models.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name} ({model.architecture_type})
                  </option>
                ))
              )}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Welding Speed (mm/s)
            </label>
            <input
              type="range"
              min="10"
              max="30"
              value={speed}
              onChange={(e) => setSpeed(Number(e.target.value))}
              className="w-full"
            />
            <div className="text-sm text-gray-600 mt-1">{speed} mm/s</div>
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              mm/pixel
            </label>
            <input
              type="number"
              step="0.1"
              value={config.mmPerPixel}
              onChange={(e) => setConfig({ ...config, mmPerPixel: Number(e.target.value) })}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Offset X (mm)
            </label>
            <input
              type="number"
              value={config.originOffsetX}
              onChange={(e) => setConfig({ ...config, originOffsetX: Number(e.target.value) })}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Offset Y (mm)
            </label>
            <input
              type="number"
              value={config.originOffsetY}
              onChange={(e) => setConfig({ ...config, originOffsetY: Number(e.target.value) })}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Z-Height (mm)
            </label>
            <input
              type="number"
              value={config.zHeight}
              onChange={(e) => setConfig({ ...config, zHeight: Number(e.target.value) })}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            />
          </div>
        </div>

        <div className="flex gap-4">
          <label className="flex-1 cursor-pointer">
            <div className="flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
              <Upload size={20} />
              Upload Image
            </div>
            <input
              type="file"
              accept="image/jpeg,image/png"
              onChange={handleImageUpload}
              className="hidden"
            />
          </label>

          <button
            onClick={handleGeneratePaths}
            disabled={!uploadedImage || processing}
            className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            <Zap size={20} />
            {processing ? 'Processing...' : 'Generate Paths'}
          </button>

          <button
            onClick={handleDownload}
            disabled={!vectorPaths}
            className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            <Download size={20} />
            Download All
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <PreviewCard title="Original Image" image={uploadedImage} />
        <PreviewCard title="AI Mask" image={maskImage} />
        <PreviewCard title="Vector Overlay" image={overlayImage} />
      </div>

      {vectorPaths && (
        <div className="mt-6 bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Path Statistics</h3>
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <div className="text-3xl font-bold text-blue-600">{vectorPaths.totalPaths}</div>
              <div className="text-sm text-gray-600">Total Paths</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-green-600">{vectorPaths.totalPoints}</div>
              <div className="text-sm text-gray-600">Total Points</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-purple-600">
                {(vectorPaths.paths.reduce((sum: number, p: any) => sum + p.length, 0) * config.mmPerPixel).toFixed(1)}
              </div>
              <div className="text-sm text-gray-600">Total Length (mm)</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function PreviewCard({ title, image }: { title: string; image: string | null }) {
  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
      <h3 className="text-sm font-semibold text-gray-700 mb-3">{title}</h3>
      <div className="aspect-[2/3] bg-gray-100 rounded-lg overflow-hidden flex items-center justify-center">
        {image ? (
          <img src={image} alt={title} className="w-full h-full object-contain" />
        ) : (
          <div className="text-gray-400 text-sm">No image</div>
        )}
      </div>
    </div>
  );
}
