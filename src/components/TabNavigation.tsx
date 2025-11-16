import React from 'react';

export type TabId = 'generate' | 'train' | 'dataset' | 'batch' | 'canny';

interface Tab {
  id: TabId;
  label: string;
  icon: string;
}

interface TabNavigationProps {
  activeTab: TabId;
  onTabChange: (tab: TabId) => void;
}

const tabs: Tab[] = [
  { id: 'generate', label: 'Generate Paths', icon: '🎯' },
  { id: 'train', label: 'Train Models', icon: '🧠' },
  { id: 'dataset', label: 'Dataset Builder', icon: '📁' },
  { id: 'batch', label: 'Batch Processing', icon: '⚡' },
  { id: 'canny', label: 'Canny Batch', icon: '🔍' }
];

export function TabNavigation({ activeTab, onTabChange }: TabNavigationProps) {
  return (
    <div className="border-b border-gray-200 bg-white shadow-sm">
      <nav className="flex space-x-1 px-6" aria-label="Tabs">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            className={`
              px-6 py-4 text-sm font-medium border-b-2 transition-colors
              ${activeTab === tab.id
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-gray-600 hover:text-gray-900 hover:border-gray-300'
              }
            `}
          >
            <span className="mr-2">{tab.icon}</span>
            {tab.label}
          </button>
        ))}
      </nav>
    </div>
  );
}
