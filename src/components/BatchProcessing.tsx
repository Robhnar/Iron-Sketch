import React, { useState, useEffect } from 'react';
import { Upload, Download, Play, CheckCircle, XCircle, Clock } from 'lucide-react';
import { Model } from '../lib/fileStorage';

interface ProcessingHistory {
  id: string;
  model_id: string;
  input_image_url: string;
  output_mask_url?: string;
  overlay_image_url?: string;
  paths_json: any[];
  num_paths: number;
  num_points: number;
  processing_time_ms: number;
}
import { resizeImageToTarget, simulateAIInference, TARGET_WIDTH, TARGET_HEIGHT } from '../utils/imageProcessing';
import { vectorizeMask, createOverlayImage } from '../utils/vectorization';
import { generateABBScript, generateGCode, generateParametersFile, CoordinateConfig } from '../utils/robotScript';

interface BatchItem {
  id: string;
  file: File;
  preview: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  result?: ProcessingHistory;
  error?: string;
}

export function BatchProcessing() {
  const [models, setModels] = useState<Model[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>('');
  const [batchItems, setBatchItems] = useState<BatchItem[]>([]);
  const [processing, setProcessing] = useState(false);
  const [config, setConfig] = useState<CoordinateConfig>({
    mmPerPixel: 0.5,
    originOffsetX: 0,
    originOffsetY: 0,
    zHeight: 5
  });
  const [speed, setSpeed] = useState(20);

  useEffect(() => {
    loadModels();
  }, []);

  async function loadModels() {
    try {
      const stored = localStorage.getItem('models');
      const models = stored ? JSON.parse(stored) : [];
      setModels(models);
      if (models.length > 0 && !selectedModelId) {
        setSelectedModelId(models[0].id);
      }
    } catch (error) {
      console.error('Error loading models:', error);
    }
  }

  async function handleFileUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files || []);

    const newItems: BatchItem[] = await Promise.all(
      files.map(async (file) => {
        const processed = await resizeImageToTarget(file);
        return {
          id: crypto.randomUUID(),
          file,
          preview: processed.dataUrl,
          status: 'pending' as const
        };
      })
    );

    setBatchItems([...batchItems, ...newItems]);
  }

  async function processBatch() {
    if (batchItems.length === 0) {
      alert('Please upload images first');
      return;
    }

    setProcessing(true);

    for (let i = 0; i < batchItems.length; i++) {
      const item = batchItems[i];
      if (item.status !== 'pending') continue;

      setBatchItems(prev => prev.map(b =>
        b.id === item.id ? { ...b, status: 'processing' } : b
      ));

      try {
        const startTime = Date.now();

        const processed = await resizeImageToTarget(item.file);

        const maskData = await simulateAIInference(processed);

        const canvas = document.createElement('canvas');
        canvas.width = TARGET_WIDTH;
        canvas.height = TARGET_HEIGHT;
        const ctx = canvas.getContext('2d');
        if (!ctx) throw new Error('Could not get canvas context');

        ctx.putImageData(maskData, 0, 0);
        const maskUrl = canvas.toDataURL('image/png');

        const paths = vectorizeMask(maskData, 50);

        const overlayUrl = await createOverlayImage(
          processed.dataUrl,
          paths.paths,
          TARGET_WIDTH,
          TARGET_HEIGHT
        );

        const processingTime = Date.now() - startTime;

        const historyRecord: ProcessingHistory = {
          id: crypto.randomUUID(),
          model_id: selectedModelId || '',
          input_image_url: processed.dataUrl,
          output_mask_url: maskUrl,
          overlay_image_url: overlayUrl,
          paths_json: paths.paths,
          num_paths: paths.totalPaths,
          num_points: paths.totalPoints,
          processing_time_ms: processingTime
        };

        const storedHistory = localStorage.getItem('processing_history');
        const history = storedHistory ? JSON.parse(storedHistory) : [];
        history.push(historyRecord);
        localStorage.setItem('processing_history', JSON.stringify(history));

        setBatchItems(prev => prev.map(b =>
          b.id === item.id ? { ...b, status: 'completed', result: historyRecord } : b
        ));

      } catch (error) {
        console.error('Error processing item:', error);
        setBatchItems(prev => prev.map(b =>
          b.id === item.id ? { ...b, status: 'error', error: String(error) } : b
        ));
      }

      await new Promise(resolve => setTimeout(resolve, 100));
    }

    setProcessing(false);
  }

  async function downloadAllResults() {
    const completedItems = batchItems.filter(item => item.status === 'completed' && item.result);

    if (completedItems.length === 0) {
      alert('No completed items to download');
      return;
    }

    for (const item of completedItems) {
      if (!item.result) continue;

      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
      const baseName = item.file.name.replace(/\.[^/.]+$/, '');
      const folderName = `${baseName}_${timestamp}`;

      const paths = item.result.paths_json as any[];

      const abbScript = generateABBScript({
        paths: paths.map(p => ({ points: p.points, length: p.length })),
        speed,
        config,
        imageHeight: TARGET_HEIGHT
      });

      const gcode = generateGCode({
        paths: paths.map(p => ({ points: p.points, length: p.length })),
        speed,
        config,
        imageHeight: TARGET_HEIGHT
      });

      const parameters = generateParametersFile({
        paths: paths.map(p => ({ points: p.points, length: p.length })),
        speed,
        config,
        imageHeight: TARGET_HEIGHT
      });

      const files = [
        { name: '00_input.png', content: item.result.input_image_url, type: 'image/png' },
        { name: '01_ai_mask.png', content: item.result.output_mask_url!, type: 'image/png' },
        { name: '02_vector_overlay.png', content: item.result.overlay_image_url!, type: 'image/png' },
        { name: '03_paths.json', content: JSON.stringify(paths, null, 2), type: 'application/json' },
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

      await new Promise(resolve => setTimeout(resolve, 200));
    }
  }

  function clearCompleted() {
    setBatchItems(batchItems.filter(item => item.status !== 'completed'));
  }

  const stats = {
    total: batchItems.length,
    pending: batchItems.filter(i => i.status === 'pending').length,
    processing: batchItems.filter(i => i.status === 'processing').length,
    completed: batchItems.filter(i => i.status === 'completed').length,
    error: batchItems.filter(i => i.status === 'error').length
  };

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Batch Processing</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Model
            </label>
            <select
              value={selectedModelId}
              onChange={(e) => setSelectedModelId(e.target.value)}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              disabled={processing}
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
              disabled={processing}
            />
            <div className="text-sm text-gray-600 mt-1">{speed} mm/s</div>
          </div>
        </div>

        <div className="flex gap-4 mb-6">
          <label className="flex-1 cursor-pointer">
            <div className="flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
              <Upload size={20} />
              Upload Multiple Images
            </div>
            <input
              type="file"
              accept="image/jpeg,image/png"
              multiple
              onChange={handleFileUpload}
              className="hidden"
              disabled={processing}
            />
          </label>

          <button
            onClick={processBatch}
            disabled={processing || batchItems.length === 0}
            className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            <Play size={20} />
            {processing ? 'Processing...' : 'Process All'}
          </button>

          <button
            onClick={downloadAllResults}
            disabled={stats.completed === 0}
            className="flex-1 flex items-center justify-center gap-2 px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            <Download size={20} />
            Download All
          </button>
        </div>

        <div className="grid grid-cols-5 gap-4 p-4 bg-gray-50 rounded-lg">
          <StatCard label="Total" value={stats.total} color="gray" />
          <StatCard label="Pending" value={stats.pending} color="blue" />
          <StatCard label="Processing" value={stats.processing} color="yellow" />
          <StatCard label="Completed" value={stats.completed} color="green" />
          <StatCard label="Errors" value={stats.error} color="red" />
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {batchItems.map((item) => (
          <div key={item.id} className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
            <div className="aspect-[2/3] bg-gray-100 relative">
              <img src={item.preview} alt={item.file.name} className="w-full h-full object-cover" />

              <div className="absolute top-2 right-2">
                {item.status === 'pending' && (
                  <div className="bg-blue-500 text-white p-2 rounded-full">
                    <Clock size={16} />
                  </div>
                )}
                {item.status === 'processing' && (
                  <div className="bg-yellow-500 text-white p-2 rounded-full animate-spin">
                    <Clock size={16} />
                  </div>
                )}
                {item.status === 'completed' && (
                  <div className="bg-green-500 text-white p-2 rounded-full">
                    <CheckCircle size={16} />
                  </div>
                )}
                {item.status === 'error' && (
                  <div className="bg-red-500 text-white p-2 rounded-full">
                    <XCircle size={16} />
                  </div>
                )}
              </div>
            </div>

            <div className="p-3">
              <p className="text-xs font-medium text-gray-700 truncate mb-1">
                {item.file.name}
              </p>
              {item.result && (
                <div className="grid grid-cols-2 gap-2 text-xs text-gray-600">
                  <div>
                    <span className="font-medium">{item.result.num_paths}</span> paths
                  </div>
                  <div>
                    <span className="font-medium">{item.result.num_points}</span> points
                  </div>
                </div>
              )}
              {item.error && (
                <p className="text-xs text-red-600 mt-1">Error processing</p>
              )}
            </div>
          </div>
        ))}
      </div>

      {batchItems.length === 0 && (
        <div className="bg-gray-50 rounded-lg p-12 text-center">
          <Upload className="mx-auto mb-4 text-gray-400" size={48} />
          <p className="text-gray-600">No images uploaded yet. Click "Upload Multiple Images" to start batch processing.</p>
        </div>
      )}
    </div>
  );
}

function StatCard({ label, value, color }: { label: string; value: number; color: string }) {
  const colorClasses = {
    gray: 'bg-gray-100 text-gray-900',
    blue: 'bg-blue-100 text-blue-900',
    yellow: 'bg-yellow-100 text-yellow-900',
    green: 'bg-green-100 text-green-900',
    red: 'bg-red-100 text-red-900'
  };

  return (
    <div className={`p-3 rounded-lg ${colorClasses[color as keyof typeof colorClasses]}`}>
      <div className="text-xs font-medium opacity-75">{label}</div>
      <div className="text-2xl font-bold">{value}</div>
    </div>
  );
}
