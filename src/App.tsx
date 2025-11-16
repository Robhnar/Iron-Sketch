import React, { useState } from 'react';
import { Zap } from 'lucide-react';
import { TabNavigation, TabId } from './components/TabNavigation';
import { GeneratePaths } from './components/GeneratePaths';
import { TrainModels } from './components/TrainModels';
import { DatasetBuilder } from './components/DatasetBuilder';
import { BatchProcessing } from './components/BatchProcessing';
import { CannyBatchProcessor } from './components/CannyBatchProcessor';

function App() {
  const [activeTab, setActiveTab] = useState<TabId>('generate');

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-gradient-to-r from-blue-700 to-blue-900 text-white shadow-lg">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="flex items-center gap-4">
            <div className="bg-white/10 p-3 rounded-xl backdrop-blur-sm">
              <Zap size={32} className="text-yellow-300" />
            </div>
            <div>
              <h1 className="text-3xl font-bold tracking-tight">AI Welding Path Generator - IronSketch</h1>
              <p className="text-blue-100 mt-1">Transform images into robot-executable welding paths with AI and edge detection</p>
            </div>
          </div>
        </div>
      </header>

      <TabNavigation activeTab={activeTab} onTabChange={setActiveTab} />

      <main className="py-6">
        {activeTab === 'generate' && <GeneratePaths />}
        {activeTab === 'train' && <TrainModels />}
        {activeTab === 'dataset' && <DatasetBuilder />}
        {activeTab === 'batch' && <BatchProcessing />}
        {activeTab === 'canny' && <CannyBatchProcessor />}
      </main>

      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <h3 className="font-semibold text-gray-900 mb-3">Supported Architectures</h3>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>U-Net (7.7M parameters)</li>
                <li>DeepLabV3+ (2.5M parameters)</li>
                <li>FCN-8s (14M parameters)</li>
                <li>Canny Edge Detection (Algorithm)</li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 mb-3">Output Formats</h3>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>JavaScript Robot Script (.js)</li>
                <li>G-code for CNC Plasma (.nc)</li>
                <li>JSON Path Coordinates (.json)</li>
                <li>Visual Overlays (.png)</li>
              </ul>
            </div>
            <div>
              <h3 className="font-semibold text-gray-900 mb-3">Features</h3>
              <ul className="text-sm text-gray-600 space-y-1">
                <li>Real-time AI inference</li>
                <li>Custom model training</li>
                <li>Batch processing</li>
                <li>Dataset management</li>
              </ul>
            </div>
          </div>
          <div className="mt-8 pt-6 border-t border-gray-200 text-center text-sm text-gray-500">
            AI Welding Path Generator - IronSketch | Production-ready automation for robotic welding systems
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
