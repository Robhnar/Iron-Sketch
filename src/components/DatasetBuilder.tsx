import React, { useState, useEffect } from 'react';
import { Upload, Plus, Trash2, Save, FolderOpen, FileArchive, Folder } from 'lucide-react';
import { Dataset } from '../lib/fileStorage';
import { resizeImageToTarget } from '../utils/imageProcessing';

interface ImagePair {
  id: string;
  inputFile: File | null;
  targetFile: File | null;
  inputPreview: string | null;
  targetPreview: string | null;
}

export function DatasetBuilder() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [currentDataset, setCurrentDataset] = useState<Dataset | null>(null);
  const [datasetName, setDatasetName] = useState('');
  const [datasetDescription, setDatasetDescription] = useState('');
  const [trainSplit, setTrainSplit] = useState(0.8);

  const [imagePairs, setImagePairs] = useState<ImagePair[]>([]);
  const [augmentation, setAugmentation] = useState({
    rotation: 15,
    brightness: 20,
    flipping: true
  });

  const [saving, setSaving] = useState(false);
  const [importMode, setImportMode] = useState<'manual' | 'folder' | 'archive'>('manual');
  const [useDatabase, setUseDatabase] = useState(true);

  useEffect(() => {
    loadDatasets();
  }, []);

  async function loadDatasets() {
    try {
      const stored = localStorage.getItem('datasets');
      if (stored) {
        setDatasets(JSON.parse(stored));
      }
    } catch (error) {
      console.error('Error loading datasets:', error);
    }
  }

  function addImagePair() {
    setImagePairs([
      ...imagePairs,
      {
        id: crypto.randomUUID(),
        inputFile: null,
        targetFile: null,
        inputPreview: null,
        targetPreview: null
      }
    ]);
  }

  async function handleBulkImageImport(e: React.ChangeEvent<HTMLInputElement>) {
    const files = Array.from(e.target.files || []);
    if (files.length === 0) return;

    const newPairs: ImagePair[] = [];

    for (const file of files) {
      try {
        const processed = await resizeImageToTarget(file);
        newPairs.push({
          id: crypto.randomUUID(),
          inputFile: file,
          targetFile: null,
          inputPreview: processed.dataUrl,
          targetPreview: null
        });
      } catch (error) {
        console.error('Error processing file:', file.name, error);
      }
    }

    setImagePairs([...imagePairs, ...newPairs]);
    alert(`Imported ${newPairs.length} images`);
  }

  async function handleArchiveImport(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    alert('Archive import feature: Extract images from ZIP/TAR archives. Coming soon!');
  }

  async function exportDatasetAsArchive() {
    if (imagePairs.length === 0) {
      alert('No images to export');
      return;
    }

    alert(`Export ${imagePairs.length} image pairs as ZIP archive with metadata.json`);
  }

  function removeImagePair(id: string) {
    setImagePairs(imagePairs.filter(pair => pair.id !== id));
  }

  async function handleInputImage(id: string, file: File) {
    try {
      const processed = await resizeImageToTarget(file);

      setImagePairs(imagePairs.map(pair =>
        pair.id === id
          ? { ...pair, inputFile: file, inputPreview: processed.dataUrl }
          : pair
      ));
    } catch (error) {
      console.error('Error processing input image:', error);
      alert('Failed to process image');
    }
  }

  async function handleTargetImage(id: string, file: File) {
    try {
      const processed = await resizeImageToTarget(file);

      setImagePairs(imagePairs.map(pair =>
        pair.id === id
          ? { ...pair, targetFile: file, targetPreview: processed.dataUrl }
          : pair
      ));
    } catch (error) {
      console.error('Error processing target image:', error);
      alert('Failed to process image');
    }
  }

  async function handleCreateDataset() {
    if (!datasetName || imagePairs.length === 0) {
      alert('Please enter a dataset name and add at least one image pair');
      return;
    }

    const incompletePairs = imagePairs.filter(pair => !pair.inputFile || !pair.targetFile);
    if (incompletePairs.length > 0) {
      alert('Please complete all image pairs before saving');
      return;
    }

    try {
      setSaving(true);

      const newDataset: Dataset = {
        id: crypto.randomUUID(),
        name: datasetName,
        description: datasetDescription,
        num_images: imagePairs.length,
        augmentation_config: augmentation,
        train_split: trainSplit,
        status: 'ready',
        created_at: new Date().toISOString()
      };

      const imageData = imagePairs.map(pair => ({
        id: crypto.randomUUID(),
        dataset_id: newDataset.id,
        input_image_url: pair.inputPreview!,
        target_mask_url: pair.targetPreview!,
        original_filename: pair.inputFile!.name,
        split_type: Math.random() < trainSplit ? 'train' : 'val'
      }));

      const stored = localStorage.getItem('datasets');
      const datasets = stored ? JSON.parse(stored) : [];
      datasets.push(newDataset);
      localStorage.setItem('datasets', JSON.stringify(datasets));

      const storedImages = localStorage.getItem('dataset_images');
      const allImages = storedImages ? JSON.parse(storedImages) : [];
      allImages.push(...imageData);
      localStorage.setItem('dataset_images', JSON.stringify(allImages));

      alert('Dataset created successfully!');
      setDatasetName('');
      setDatasetDescription('');
      setImagePairs([]);
      loadDatasets();

    } catch (error) {
      console.error('Error creating dataset:', error);
      alert('Failed to create dataset');
    } finally {
      setSaving(false);
    }
  }

  async function loadDataset(dataset: Dataset) {
    setCurrentDataset(dataset);

    const storedImages = localStorage.getItem('dataset_images');
    if (storedImages) {
      const allImages = JSON.parse(storedImages);
      const images = allImages.filter((img: any) => img.dataset_id === dataset.id);

      const pairs: ImagePair[] = images.map((img: any) => ({
        id: img.id,
        inputFile: null,
        targetFile: null,
        inputPreview: img.input_image_url,
        targetPreview: img.target_mask_url
      }));
      setImagePairs(pairs);
      setDatasetName(dataset.name);
      setDatasetDescription(dataset.description);
      setTrainSplit(dataset.train_split);
      setAugmentation(dataset.augmentation_config as any);
    }
  }

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Dataset Builder</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Dataset Name
            </label>
            <input
              type="text"
              value={datasetName}
              onChange={(e) => setDatasetName(e.target.value)}
              placeholder="e.g., Welding Seams Dataset v1"
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Train/Val Split
            </label>
            <input
              type="range"
              min="0.5"
              max="0.95"
              step="0.05"
              value={trainSplit}
              onChange={(e) => setTrainSplit(Number(e.target.value))}
              className="w-full"
            />
            <div className="text-sm text-gray-600 mt-1">
              Train: {(trainSplit * 100).toFixed(0)}% / Val: {((1 - trainSplit) * 100).toFixed(0)}%
            </div>
          </div>
        </div>

        <div className="mb-6">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Description
          </label>
          <textarea
            value={datasetDescription}
            onChange={(e) => setDatasetDescription(e.target.value)}
            placeholder="Brief description of this dataset..."
            rows={2}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
          />
        </div>

        <div className="grid grid-cols-3 gap-4 mb-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Rotation (±degrees)
            </label>
            <input
              type="number"
              value={augmentation.rotation}
              onChange={(e) => setAugmentation({ ...augmentation, rotation: Number(e.target.value) })}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Brightness (±%)
            </label>
            <input
              type="number"
              value={augmentation.brightness}
              onChange={(e) => setAugmentation({ ...augmentation, brightness: Number(e.target.value) })}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Horizontal Flip
            </label>
            <label className="flex items-center cursor-pointer mt-2">
              <input
                type="checkbox"
                checked={augmentation.flipping}
                onChange={(e) => setAugmentation({ ...augmentation, flipping: e.target.checked })}
                className="w-5 h-5 text-blue-600 border-gray-300 rounded focus:ring-blue-500"
              />
              <span className="ml-2 text-sm text-gray-700">Enable</span>
            </label>
          </div>
        </div>

        <div className="mb-6 p-4 bg-gray-50 rounded-lg">
          <h3 className="text-sm font-semibold text-gray-900 mb-3">Import Mode</h3>
          <div className="flex gap-4">
            <label className="flex items-center cursor-pointer">
              <input
                type="radio"
                checked={importMode === 'manual'}
                onChange={() => setImportMode('manual')}
                className="mr-2"
              />
              <span className="text-sm">Manual (one by one)</span>
            </label>
            <label className="flex items-center cursor-pointer">
              <input
                type="radio"
                checked={importMode === 'folder'}
                onChange={() => setImportMode('folder')}
                className="mr-2"
              />
              <span className="text-sm">Bulk Import (multiple files)</span>
            </label>
            <label className="flex items-center cursor-pointer">
              <input
                type="radio"
                checked={importMode === 'archive'}
                onChange={() => setImportMode('archive')}
                className="mr-2"
              />
              <span className="text-sm">From Archive (ZIP/TAR)</span>
            </label>
          </div>
        </div>

        <div className="flex flex-wrap gap-4">
          {importMode === 'manual' && (
            <button
              onClick={addImagePair}
              className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <Plus size={20} />
              Add Image Pair
            </button>
          )}

          {importMode === 'folder' && (
            <label className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors cursor-pointer">
              <Folder size={20} />
              Import Multiple Images
              <input
                type="file"
                accept="image/jpeg,image/png"
                multiple
                onChange={handleBulkImageImport}
                className="hidden"
              />
            </label>
          )}

          {importMode === 'archive' && (
            <label className="flex items-center gap-2 px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors cursor-pointer">
              <FileArchive size={20} />
              Import from Archive
              <input
                type="file"
                accept=".zip,.tar,.tar.gz"
                onChange={handleArchiveImport}
                className="hidden"
              />
            </label>
          )}

          <button
            onClick={handleCreateDataset}
            disabled={saving || imagePairs.length === 0}
            className="flex items-center gap-2 px-6 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            <Save size={20} />
            {saving ? 'Saving...' : 'Save Dataset'}
          </button>

          <button
            onClick={exportDatasetAsArchive}
            disabled={imagePairs.length === 0}
            className="flex items-center gap-2 px-6 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
          >
            <FileArchive size={20} />
            Export as Archive
          </button>
        </div>
      </div>

      <div className="space-y-4 mb-6">
        {imagePairs.map((pair) => (
          <div key={pair.id} className="bg-white rounded-lg shadow-sm border border-gray-200 p-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Input Image
                </label>
                <label className="cursor-pointer">
                  <div className="aspect-[2/3] bg-gray-100 rounded-lg overflow-hidden flex items-center justify-center border-2 border-dashed border-gray-300 hover:border-blue-400 transition-colors">
                    {pair.inputPreview ? (
                      <img src={pair.inputPreview} alt="Input" className="w-full h-full object-contain" />
                    ) : (
                      <div className="text-center p-4">
                        <Upload className="mx-auto mb-2 text-gray-400" size={32} />
                        <p className="text-sm text-gray-500">Click to upload input</p>
                      </div>
                    )}
                  </div>
                  <input
                    type="file"
                    accept="image/jpeg,image/png"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) handleInputImage(pair.id, file);
                    }}
                    className="hidden"
                  />
                </label>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Target Mask
                </label>
                <label className="cursor-pointer">
                  <div className="aspect-[2/3] bg-gray-100 rounded-lg overflow-hidden flex items-center justify-center border-2 border-dashed border-gray-300 hover:border-green-400 transition-colors">
                    {pair.targetPreview ? (
                      <img src={pair.targetPreview} alt="Target" className="w-full h-full object-contain" />
                    ) : (
                      <div className="text-center p-4">
                        <Upload className="mx-auto mb-2 text-gray-400" size={32} />
                        <p className="text-sm text-gray-500">Click to upload target</p>
                      </div>
                    )}
                  </div>
                  <input
                    type="file"
                    accept="image/jpeg,image/png"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) handleTargetImage(pair.id, file);
                    }}
                    className="hidden"
                  />
                </label>
              </div>
            </div>

            <button
              onClick={() => removeImagePair(pair.id)}
              className="mt-4 flex items-center gap-2 px-4 py-2 text-red-600 hover:bg-red-50 rounded-lg transition-colors"
            >
              <Trash2 size={16} />
              Remove Pair
            </button>
          </div>
        ))}
      </div>

      {imagePairs.length === 0 && (
        <div className="bg-gray-50 rounded-lg p-12 text-center">
          <FolderOpen className="mx-auto mb-4 text-gray-400" size={48} />
          <p className="text-gray-600">No image pairs added yet. Click "Add Image Pair" to start building your dataset.</p>
        </div>
      )}

      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mt-6">
        <h3 className="text-xl font-bold text-gray-900 mb-4">Existing Datasets</h3>

        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Name</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Images</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Split</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Created</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {datasets.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-4 py-8 text-center text-gray-500">
                    No datasets created yet
                  </td>
                </tr>
              ) : (
                datasets.map((dataset) => (
                  <tr key={dataset.id} className="hover:bg-gray-50">
                    <td className="px-4 py-3 text-sm font-medium text-gray-900">{dataset.name}</td>
                    <td className="px-4 py-3 text-sm text-gray-600">{dataset.num_images}</td>
                    <td className="px-4 py-3 text-sm text-gray-600">
                      {(dataset.train_split * 100).toFixed(0)}% / {((1 - dataset.train_split) * 100).toFixed(0)}%
                    </td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                        dataset.status === 'ready' ? 'bg-green-100 text-green-800' :
                        dataset.status === 'building' ? 'bg-yellow-100 text-yellow-800' :
                        'bg-gray-100 text-gray-800'
                      }`}>
                        {dataset.status}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-600">
                      {new Date(dataset.created_at).toLocaleDateString()}
                    </td>
                    <td className="px-4 py-3">
                      <button
                        onClick={() => loadDataset(dataset)}
                        className="text-blue-600 hover:text-blue-800 text-sm font-medium"
                      >
                        View
                      </button>
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
