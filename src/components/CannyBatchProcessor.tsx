import React, { useState } from 'react';
import { Upload, Play, Download, Image as ImageIcon, Sliders } from 'lucide-react';
import { applyCannyEdgeDetection, CannyParams, CANNY_PRESETS } from '../utils/cannyEdgeDetection';

interface ProcessedImage {
  id: string;
  original: string;
  edges: string | null;
  fileName: string;
  processing: boolean;
}

export function CannyBatchProcessor() {
  const [images, setImages] = useState<ProcessedImage[]>([]);
  const [params, setParams] = useState<CannyParams>({
    lowThreshold: 0.1,
    highThreshold: 0.3,
    kernelSize: 5,
    sigma: 1.4
  });
  const [processing, setProcessing] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState<string>('balanced');

  async function handleImageUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;

    const newImages: ProcessedImage[] = files.map(file => ({
      id: crypto.randomUUID(),
      original: URL.createObjectURL(file),
      edges: null,
      fileName: file.name,
      processing: false
    }));

    setImages([...images, ...newImages]);
  }

  async function processAllImages() {
    setProcessing(true);

    for (let i = 0; i < images.length; i++) {
      if (images[i].edges) continue;

      setImages(prev => prev.map((img, idx) =>
        idx === i ? { ...img, processing: true } : img
      ));

      try {
        const edges = await applyCannyEdgeDetection(images[i].original, params);
        setImages(prev => prev.map((img, idx) =>
          idx === i ? { ...img, edges, processing: false } : img
        ));
      } catch (error) {
        console.error('Error processing image:', error);
        setImages(prev => prev.map((img, idx) =>
          idx === i ? { ...img, processing: false } : img
        ));
      }

      await new Promise(resolve => setTimeout(resolve, 100));
    }

    setProcessing(false);
  }

  async function processWithPreset(presetName: string) {
    const preset = CANNY_PRESETS[presetName as keyof typeof CANNY_PRESETS];
    if (!preset) return;

    setParams({
      lowThreshold: preset.lowThreshold,
      highThreshold: preset.highThreshold,
      kernelSize: preset.kernelSize,
      sigma: preset.sigma
    });
    setSelectedPreset(presetName);
  }

  function clearAll() {
    images.forEach(img => {
      URL.revokeObjectURL(img.original);
      if (img.edges) URL.revokeObjectURL(img.edges);
    });
    setImages([]);
  }

  async function downloadAll() {
    for (const img of images) {
      if (!img.edges) continue;

      const response = await fetch(img.edges);
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `canny_${img.fileName}`;
      a.click();
      URL.revokeObjectURL(url);
    }
  }

  async function downloadImagePairs() {
    const processedImages = images.filter(img => img.edges);
    if (processedImages.length === 0) return;

    for (const img of processedImages) {
      const baseName = img.fileName.replace(/\.[^/.]+$/, '');

      const originalResponse = await fetch(img.original);
      const originalBlob = await originalResponse.blob();
      const originalUrl = URL.createObjectURL(originalBlob);
      const aOriginal = document.createElement('a');
      aOriginal.href = originalUrl;
      aOriginal.download = `${baseName}_input.png`;
      aOriginal.click();
      URL.revokeObjectURL(originalUrl);

      await new Promise(resolve => setTimeout(resolve, 100));

      const edgesResponse = await fetch(img.edges!);
      const edgesBlob = await edgesResponse.blob();
      const edgesUrl = URL.createObjectURL(edgesBlob);
      const aEdges = document.createElement('a');
      aEdges.href = edgesUrl;
      aEdges.download = `${baseName}_target.png`;
      aEdges.click();
      URL.revokeObjectURL(edgesUrl);

      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Canny Batch Processor</h2>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <div>
            <h3 className="text-sm font-semibold text-gray-900 mb-3">Upload Images</h3>
            <label className="flex items-center justify-center gap-2 px-6 py-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-blue-400 cursor-pointer transition-colors">
              <Upload size={24} />
              <span>Select Multiple Images</span>
              <input
                type="file"
                accept="image/jpeg,image/png"
                multiple
                onChange={handleImageUpload}
                className="hidden"
              />
            </label>
            <p className="text-xs text-gray-500 mt-2">
              {images.length} image{images.length !== 1 ? 's' : ''} loaded
            </p>
          </div>

          <div>
            <h3 className="text-sm font-semibold text-gray-900 mb-3">Quick Presets</h3>
            <div className="grid grid-cols-2 gap-2">
              {Object.entries(CANNY_PRESETS).map(([key, preset]) => (
                <button
                  key={key}
                  onClick={() => processWithPreset(key)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                    selectedPreset === key
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}
                >
                  {preset.name}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="bg-gray-50 rounded-lg p-4 mb-6">
          <div className="flex items-center gap-2 mb-4">
            <Sliders size={18} />
            <h3 className="text-sm font-semibold text-gray-900">Parameters</h3>
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Low Threshold: {params.lowThreshold.toFixed(2)}
              </label>
              <input
                type="range"
                min="0.01"
                max="0.5"
                step="0.01"
                value={params.lowThreshold}
                onChange={(e) => setParams({ ...params, lowThreshold: Number(e.target.value) })}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                High Threshold: {params.highThreshold.toFixed(2)}
              </label>
              <input
                type="range"
                min="0.1"
                max="0.8"
                step="0.01"
                value={params.highThreshold}
                onChange={(e) => setParams({ ...params, highThreshold: Number(e.target.value) })}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Kernel Size: {params.kernelSize}
              </label>
              <input
                type="range"
                min="3"
                max="15"
                step="2"
                value={params.kernelSize}
                onChange={(e) => setParams({ ...params, kernelSize: Number(e.target.value) })}
                className="w-full"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-gray-700 mb-1">
                Sigma: {params.sigma.toFixed(1)}
              </label>
              <input
                type="range"
                min="0.5"
                max="5.0"
                step="0.1"
                value={params.sigma}
                onChange={(e) => setParams({ ...params, sigma: Number(e.target.value) })}
                className="w-full"
              />
            </div>
          </div>
        </div>

        <div className="flex flex-wrap gap-3">
          <button
            onClick={processAllImages}
            disabled={processing || images.length === 0}
            className="flex items-center gap-2 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            <Play size={20} />
            {processing ? 'Processing...' : 'Process All'}
          </button>

          <button
            onClick={downloadAll}
            disabled={images.filter(img => img.edges).length === 0}
            className="flex items-center gap-2 px-5 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            <Download size={20} />
            Edges Only
          </button>

          <button
            onClick={downloadImagePairs}
            disabled={images.filter(img => img.edges).length === 0}
            className="flex items-center gap-2 px-5 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            <ImageIcon size={20} />
            Pairs (Input+Target)
          </button>

          <button
            onClick={clearAll}
            disabled={images.length === 0}
            className="flex items-center gap-2 px-6 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            Clear All
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {images.map((img) => (
          <div key={img.id} className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
            <div className="text-xs font-medium text-gray-700 mb-2 truncate">{img.fileName}</div>
            <div className="grid grid-cols-2 gap-2">
              <div>
                <div className="text-xs text-gray-500 mb-1">Original</div>
                <div className="aspect-square bg-gray-100 rounded overflow-hidden">
                  <img src={img.original} alt="Original" className="w-full h-full object-cover" />
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-500 mb-1">Edges</div>
                <div className="aspect-square bg-gray-100 rounded overflow-hidden flex items-center justify-center">
                  {img.processing ? (
                    <div className="text-xs text-gray-400">Processing...</div>
                  ) : img.edges ? (
                    <img src={img.edges} alt="Edges" className="w-full h-full object-cover" />
                  ) : (
                    <ImageIcon className="text-gray-300" size={32} />
                  )}
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {images.length === 0 && (
        <div className="bg-gray-50 rounded-lg p-12 text-center">
          <ImageIcon className="mx-auto mb-4 text-gray-400" size={48} />
          <p className="text-gray-600">No images loaded. Upload images to start batch processing.</p>
        </div>
      )}
    </div>
  );
}
