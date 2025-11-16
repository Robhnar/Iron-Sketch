import React, { useState, useEffect } from 'react';
import { Play, TrendingUp } from 'lucide-react';
import { Model, Dataset } from '../lib/fileStorage';

interface TrainingRun {
  id: string;
  model_id: string;
  dataset_id: string;
  epochs: number;
  status: string;
  started_at: string;
}

export function TrainModels() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [models, setModels] = useState<Model[]>([]);
  const [trainingRuns, setTrainingRuns] = useState<TrainingRun[]>([]);

  const [selectedDataset, setSelectedDataset] = useState<string>('');
  const [architecture, setArchitecture] = useState<'unet' | 'deeplabv3plus' | 'fcn8s' | 'deeplabv3plus_resnet50' | 'deeplabv3_hf' | 'deeplabv3_google' | 'canny'>('unet');
  const [modelName, setModelName] = useState('');
  const [epochs, setEpochs] = useState(50);
  const [learningRate, setLearningRate] = useState(0.001);
  const [batchSize, setBatchSize] = useState(4);
  const [optimizer, setOptimizer] = useState<'adam' | 'sgd' | 'rmsprop'>('adam');

  const [cannyParams, setCannyParams] = useState({
    lowThreshold: 0.1,
    highThreshold: 0.3,
    kernelSize: 5,
    sigma: 1.4,
    resizeDim: 512
  });
  const [previewImage, setPreviewImage] = useState<string | null>(null);

  const [isTraining, setIsTraining] = useState(false);
  const [trainingProgress, setTrainingProgress] = useState<any>(null);

  useEffect(() => {
    loadDatasets();
    loadModels();
    loadTrainingRuns();
  }, []);

  async function loadDatasets() {
    try {
      const stored = localStorage.getItem('datasets');
      if (stored) {
        const allDatasets = JSON.parse(stored);
        setDatasets(allDatasets.filter((d: Dataset) => d.status === 'ready'));
      }
    } catch (error) {
      console.error('Error loading datasets:', error);
    }
  }

  async function loadModels() {
    try {
      const stored = localStorage.getItem('models');
      if (stored) {
        setModels(JSON.parse(stored));
      }
    } catch (error) {
      console.error('Error loading models:', error);
    }
  }

  async function loadTrainingRuns() {
    try {
      const stored = localStorage.getItem('training_runs');
      if (stored) {
        setTrainingRuns(JSON.parse(stored).slice(0, 10));
      }
    } catch (error) {
      console.error('Error loading training runs:', error);
    }
  }

  async function handleStartTraining() {
    if (!selectedDataset || !modelName) {
      alert('Please select a dataset and enter a model name');
      return;
    }

    try {
      setIsTraining(true);

      const newModel: Model = {
        id: crypto.randomUUID(),
        name: modelName,
        architecture_type: architecture,
        parameters_json: {
          epochs,
          learning_rate: learningRate,
          batch_size: batchSize,
          optimizer
        },
        performance_metrics: {},
        file_size_mb: 0,
        is_pretrained: false,
        created_at: new Date().toISOString()
      };

      const trainingRun: TrainingRun = {
        id: crypto.randomUUID(),
        model_id: newModel.id,
        dataset_id: selectedDataset,
        epochs,
        status: 'running',
        started_at: new Date().toISOString()
      };

      const storedModels = localStorage.getItem('models');
      const models = storedModels ? JSON.parse(storedModels) : [];
      models.push(newModel);
      localStorage.setItem('models', JSON.stringify(models));

      const storedRuns = localStorage.getItem('training_runs');
      const runs = storedRuns ? JSON.parse(storedRuns) : [];
      runs.push(trainingRun);
      localStorage.setItem('training_runs', JSON.stringify(runs));

      simulateTraining(trainingRun.id, newModel.id);

    } catch (error) {
      console.error('Error starting training:', error);
      alert('Failed to start training');
      setIsTraining(false);
    }
  }

  async function simulateTraining(trainingRunId: string, modelId: string) {
    const lossHistory = [];
    const metricsHistory = [];

    for (let epoch = 0; epoch < epochs; epoch++) {
      await new Promise(resolve => setTimeout(resolve, 500));

      const trainLoss = 0.8 * Math.exp(-epoch / 15) + 0.05 + Math.random() * 0.05;
      const valLoss = 0.85 * Math.exp(-epoch / 15) + 0.08 + Math.random() * 0.05;
      const dice = 1 - valLoss;
      const iou = dice * 0.85;

      lossHistory.push({ epoch: epoch + 1, train: trainLoss, val: valLoss });
      metricsHistory.push({ epoch: epoch + 1, dice, iou, accuracy: dice * 0.95 });

      setTrainingProgress({
        epoch: epoch + 1,
        totalEpochs: epochs,
        trainLoss,
        valLoss,
        dice,
        iou
      });

      const runs = JSON.parse(localStorage.getItem('training_runs') || '[]');
      const runIndex = runs.findIndex((r: TrainingRun) => r.id === trainingRunId);
      if (runIndex >= 0) {
        runs[runIndex] = { ...runs[runIndex], loss_history: lossHistory, metrics_history: metricsHistory };
        localStorage.setItem('training_runs', JSON.stringify(runs));
      }
    }

    const bestEpoch = metricsHistory.reduce((best, curr, idx) =>
      curr.dice > metricsHistory[best].dice ? idx : best, 0
    ) + 1;

    const runs = JSON.parse(localStorage.getItem('training_runs') || '[]');
    const runIndex = runs.findIndex((r: TrainingRun) => r.id === trainingRunId);
    if (runIndex >= 0) {
      runs[runIndex] = { ...runs[runIndex], status: 'completed', best_epoch: bestEpoch };
      localStorage.setItem('training_runs', JSON.stringify(runs));
    }

    const models = JSON.parse(localStorage.getItem('models') || '[]');
    const modelIndex = models.findIndex((m: Model) => m.id === modelId);
    if (modelIndex >= 0) {
      models[modelIndex] = { ...models[modelIndex], performance_metrics: metricsHistory[bestEpoch - 1] };
      localStorage.setItem('models', JSON.stringify(models));
    }

    setIsTraining(false);
    setTrainingProgress(null);
    loadModels();
    loadTrainingRuns();
    alert('Training completed successfully!');
  }

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Train New Model</h2>

          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Model Name
              </label>
              <input
                type="text"
                value={modelName}
                onChange={(e) => setModelName(e.target.value)}
                placeholder="e.g., Welding UNet v1"
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                disabled={isTraining}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Architecture
              </label>
              <select
                value={architecture}
                onChange={(e) => setArchitecture(e.target.value as any)}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                disabled={isTraining}
              >
                <option value="unet">U-Net (7.7M params)</option>
                <option value="deeplabv3plus">DeepLabV3+ (2.5M params)</option>
                <option value="fcn8s">FCN-8s (14M params)</option>
                <option value="canny">Canny Edge Detection (Algorithm)</option>
              </select>
            </div>

            {architecture === 'canny' && (
              <div className="col-span-2 p-4 bg-blue-50 rounded-lg border border-blue-200">
                <h4 className="text-sm font-semibold text-blue-900 mb-3">Canny Parameters</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs font-medium text-gray-700 mb-1">
                      Low Threshold: {cannyParams.lowThreshold.toFixed(2)}
                    </label>
                    <input
                      type="range"
                      min="0.01"
                      max="0.5"
                      step="0.01"
                      value={cannyParams.lowThreshold}
                      onChange={(e) => setCannyParams({ ...cannyParams, lowThreshold: Number(e.target.value) })}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-700 mb-1">
                      High Threshold: {cannyParams.highThreshold.toFixed(2)}
                    </label>
                    <input
                      type="range"
                      min="0.1"
                      max="0.8"
                      step="0.01"
                      value={cannyParams.highThreshold}
                      onChange={(e) => setCannyParams({ ...cannyParams, highThreshold: Number(e.target.value) })}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-700 mb-1">
                      Kernel Size: {cannyParams.kernelSize}
                    </label>
                    <input
                      type="range"
                      min="3"
                      max="15"
                      step="2"
                      value={cannyParams.kernelSize}
                      onChange={(e) => setCannyParams({ ...cannyParams, kernelSize: Number(e.target.value) })}
                      className="w-full"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-700 mb-1">
                      Sigma: {cannyParams.sigma.toFixed(1)}
                    </label>
                    <input
                      type="range"
                      min="0.5"
                      max="5.0"
                      step="0.1"
                      value={cannyParams.sigma}
                      onChange={(e) => setCannyParams({ ...cannyParams, sigma: Number(e.target.value) })}
                      className="w-full"
                    />
                  </div>
                </div>
                <p className="text-xs text-blue-700 mt-2">
                  For Canny, training means finding optimal parameters. Test on sample images from your dataset.
                </p>
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Dataset
              </label>
              <select
                value={selectedDataset}
                onChange={(e) => setSelectedDataset(e.target.value)}
                className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
                disabled={isTraining}
              >
                <option value="">Select dataset...</option>
                {datasets.map((ds) => (
                  <option key={ds.id} value={ds.id}>
                    {ds.name} ({ds.num_images} images)
                  </option>
                ))}
              </select>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Epochs
                </label>
                <input
                  type="number"
                  value={epochs}
                  onChange={(e) => setEpochs(Number(e.target.value))}
                  min="1"
                  max="200"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                  disabled={isTraining}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Batch Size
                </label>
                <input
                  type="number"
                  value={batchSize}
                  onChange={(e) => setBatchSize(Number(e.target.value))}
                  min="1"
                  max="32"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                  disabled={isTraining}
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Learning Rate
                </label>
                <input
                  type="number"
                  value={learningRate}
                  onChange={(e) => setLearningRate(Number(e.target.value))}
                  step="0.0001"
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                  disabled={isTraining}
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Optimizer
                </label>
                <select
                  value={optimizer}
                  onChange={(e) => setOptimizer(e.target.value as any)}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg"
                  disabled={isTraining}
                >
                  <option value="adam">Adam</option>
                  <option value="sgd">SGD</option>
                  <option value="rmsprop">RMSprop</option>
                </select>
              </div>
            </div>

            <button
              onClick={handleStartTraining}
              disabled={isTraining || !selectedDataset || !modelName}
              className="w-full flex items-center justify-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
            >
              <Play size={20} />
              {isTraining ? 'Training...' : 'Start Training'}
            </button>
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">Training Progress</h2>

          {trainingProgress ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between mb-4">
                <span className="text-sm font-medium text-gray-700">
                  Epoch {trainingProgress.epoch} / {trainingProgress.totalEpochs}
                </span>
                <span className="text-sm text-gray-500">
                  {((trainingProgress.epoch / trainingProgress.totalEpochs) * 100).toFixed(0)}%
                </span>
              </div>

              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-blue-600 h-2 rounded-full transition-all"
                  style={{ width: `${(trainingProgress.epoch / trainingProgress.totalEpochs) * 100}%` }}
                />
              </div>

              <div className="grid grid-cols-2 gap-4 mt-6">
                <MetricCard label="Train Loss" value={trainingProgress.trainLoss.toFixed(4)} color="blue" />
                <MetricCard label="Val Loss" value={trainingProgress.valLoss.toFixed(4)} color="red" />
                <MetricCard label="Dice Score" value={trainingProgress.dice.toFixed(4)} color="green" />
                <MetricCard label="IoU" value={trainingProgress.iou.toFixed(4)} color="purple" />
              </div>

              <div className="mt-6 p-4 bg-blue-50 rounded-lg">
                <p className="text-sm text-blue-800">
                  Training in progress... This is a simulated training process for demonstration.
                </p>
              </div>
            </div>
          ) : (
            <div className="flex items-center justify-center h-64 text-gray-400">
              <div className="text-center">
                <TrendingUp size={48} className="mx-auto mb-4 opacity-50" />
                <p>No active training</p>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4">Trained Models</h2>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Name</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Architecture</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Dice Score</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">IoU</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Created</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {models.length === 0 ? (
                <tr>
                  <td colSpan={5} className="px-4 py-8 text-center text-gray-500">
                    No models trained yet
                  </td>
                </tr>
              ) : (
                models.map((model) => (
                  <tr key={model.id} className="hover:bg-gray-50">
                    <td className="px-4 py-3 text-sm font-medium text-gray-900">{model.name}</td>
                    <td className="px-4 py-3 text-sm text-gray-600">{model.architecture_type}</td>
                    <td className="px-4 py-3 text-sm text-gray-600">
                      {model.performance_metrics?.dice?.toFixed(4) || 'N/A'}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-600">
                      {model.performance_metrics?.iou?.toFixed(4) || 'N/A'}
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-600">
                      {new Date(model.created_at).toLocaleDateString()}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function MetricCard({ label, value, color }: { label: string; value: string; color: string }) {
  const colorClasses = {
    blue: 'bg-blue-50 text-blue-900',
    red: 'bg-red-50 text-red-900',
    green: 'bg-green-50 text-green-900',
    purple: 'bg-purple-50 text-purple-900'
  };

  return (
    <div className={`p-4 rounded-lg ${colorClasses[color as keyof typeof colorClasses]}`}>
      <div className="text-xs font-medium opacity-75 mb-1">{label}</div>
      <div className="text-2xl font-bold">{value}</div>
    </div>
  );
}
