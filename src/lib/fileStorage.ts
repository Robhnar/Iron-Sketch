export interface Model {
  id: string;
  name: string;
  architecture_type: string;
  backbone?: string;
  parameters_json: Record<string, any>;
  performance_metrics: Record<string, any>;
  file_url?: string;
  file_size_mb: number;
  is_pretrained: boolean;
  created_at: string;
}

export interface Dataset {
  id: string;
  name: string;
  description: string;
  num_images: number;
  augmentation_config?: Record<string, any>;
  train_split: number;
  status: string;
  created_at: string;
}

const DEMO_MODELS: Model[] = [
  {
    id: '1',
    name: 'U-Net Demo',
    architecture_type: 'unet',
    backbone: 'ResNet34',
    parameters_json: { epochs: 20 },
    performance_metrics: { iou: 0.85 },
    file_url: '',
    file_size_mb: 7.7,
    is_pretrained: true,
    created_at: new Date().toISOString()
  },
  {
    id: '2',
    name: 'DeepLabV3+ ResNet50',
    architecture_type: 'deeplabv3plus',
    backbone: 'ResNet50',
    parameters_json: { epochs: 30 },
    performance_metrics: { iou: 0.88 },
    file_url: '',
    file_size_mb: 40,
    is_pretrained: true,
    created_at: new Date().toISOString()
  },
  {
    id: '3',
    name: 'SegFormer Pre-trained',
    architecture_type: 'unet',
    backbone: 'MiT-B0',
    parameters_json: {},
    performance_metrics: { pretrained: 'ADE20K' },
    file_url: '',
    file_size_mb: 3.7,
    is_pretrained: true,
    created_at: new Date().toISOString()
  },
  {
    id: '4',
    name: 'DeepLabV3 MobileNetV2 (Google)',
    architecture_type: 'deeplabv3plus',
    backbone: 'MobileNetV2',
    parameters_json: {},
    performance_metrics: { pretrained: 'PASCAL VOC 2012' },
    file_url: '',
    file_size_mb: 2.1,
    is_pretrained: true,
    created_at: new Date().toISOString()
  }
];

export async function listModels(): Promise<Model[]> {
  return Promise.resolve(DEMO_MODELS);
}

export async function getModel(id: string): Promise<Model | null> {
  const model = DEMO_MODELS.find(m => m.id === id);
  return Promise.resolve(model || null);
}

const DEMO_DATASETS: Dataset[] = [
  {
    id: '1',
    name: 'Sample Welding Dataset',
    description: 'Demo dataset with 50 welding images',
    num_images: 50,
    train_split: 0.8,
    status: 'ready',
    created_at: new Date().toISOString()
  }
];

export async function listDatasets(): Promise<Dataset[]> {
  return Promise.resolve(DEMO_DATASETS);
}

export async function getDataset(id: string): Promise<Dataset | null> {
  const dataset = DEMO_DATASETS.find(d => d.id === id);
  return Promise.resolve(dataset || null);
}
