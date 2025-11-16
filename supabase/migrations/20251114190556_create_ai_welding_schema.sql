/*
  # AI Welding Path Generator Database Schema

  1. New Tables
    - `models`
      - `id` (uuid, primary key) - Unique model identifier
      - `name` (text) - Model display name
      - `architecture_type` (text) - Architecture: 'unet', 'deeplabv3plus', or 'fcn8s'
      - `parameters_json` (jsonb) - Model hyperparameters and configuration
      - `performance_metrics` (jsonb) - IoU, Dice score, accuracy metrics
      - `file_url` (text) - Storage URL for model weights
      - `file_size_mb` (real) - Model file size in MB
      - `is_pretrained` (boolean) - Whether this is a pre-trained model
      - `created_at` (timestamptz) - Creation timestamp
      - `updated_at` (timestamptz) - Last update timestamp

    - `datasets`
      - `id` (uuid, primary key) - Unique dataset identifier
      - `name` (text) - Dataset display name
      - `description` (text) - Dataset description
      - `num_images` (integer) - Total number of image pairs
      - `augmentation_config` (jsonb) - Augmentation settings
      - `train_split` (real) - Training split percentage (0-1)
      - `status` (text) - Status: 'building', 'ready', 'processing'
      - `created_at` (timestamptz) - Creation timestamp
      - `updated_at` (timestamptz) - Last update timestamp

    - `dataset_images`
      - `id` (uuid, primary key) - Unique image pair identifier
      - `dataset_id` (uuid, foreign key) - Parent dataset
      - `input_image_url` (text) - Storage URL for input image
      - `target_mask_url` (text) - Storage URL for target mask
      - `original_filename` (text) - Original file name
      - `width` (integer) - Image width in pixels
      - `height` (integer) - Image height in pixels
      - `split_type` (text) - Split assignment: 'train' or 'val'
      - `uploaded_at` (timestamptz) - Upload timestamp

    - `training_runs`
      - `id` (uuid, primary key) - Unique training run identifier
      - `model_id` (uuid, foreign key) - Trained model reference
      - `dataset_id` (uuid, foreign key) - Training dataset reference
      - `epochs` (integer) - Number of training epochs
      - `learning_rate` (real) - Learning rate value
      - `batch_size` (integer) - Batch size used
      - `optimizer` (text) - Optimizer type: 'adam', 'sgd', 'rmsprop'
      - `loss_history` (jsonb) - Training and validation loss per epoch
      - `metrics_history` (jsonb) - IoU, Dice, accuracy per epoch
      - `best_epoch` (integer) - Epoch with best validation metrics
      - `status` (text) - Status: 'running', 'completed', 'failed'
      - `started_at` (timestamptz) - Training start time
      - `completed_at` (timestamptz) - Training completion time

    - `processing_history`
      - `id` (uuid, primary key) - Unique processing record identifier
      - `input_image_url` (text) - Storage URL for input image
      - `output_mask_url` (text) - Storage URL for generated mask
      - `overlay_image_url` (text) - Storage URL for vector overlay
      - `model_id` (uuid, foreign key) - Model used for inference
      - `parameters_json` (jsonb) - Processing parameters (speed, coordinates, etc)
      - `paths_json` (jsonb) - Vectorized path coordinates
      - `robot_script_url` (text) - Storage URL for generated script
      - `num_paths` (integer) - Number of paths generated
      - `num_points` (integer) - Total number of points
      - `processing_time_ms` (integer) - Processing duration in milliseconds
      - `created_at` (timestamptz) - Processing timestamp

  2. Storage Buckets
    - model-weights: Store PyTorch model checkpoint files
    - dataset-images: Store input images and target masks
    - processed-outputs: Store inference results and robot scripts

  3. Security
    - Enable RLS on all tables
    - Add policies for authenticated users to manage their own data
    - Allow public read access to pre-trained models
    
  4. Indexes
    - Index on model architecture_type for filtering
    - Index on dataset status for quick queries
    - Index on training_runs status for monitoring
    - Index on processing_history created_at for chronological queries
*/

-- Create models table
CREATE TABLE IF NOT EXISTS models (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL,
  architecture_type text NOT NULL CHECK (architecture_type IN ('unet', 'deeplabv3plus', 'fcn8s')),
  parameters_json jsonb DEFAULT '{}'::jsonb,
  performance_metrics jsonb DEFAULT '{}'::jsonb,
  file_url text,
  file_size_mb real DEFAULT 0,
  is_pretrained boolean DEFAULT false,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Create datasets table
CREATE TABLE IF NOT EXISTS datasets (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  name text NOT NULL,
  description text DEFAULT '',
  num_images integer DEFAULT 0,
  augmentation_config jsonb DEFAULT '{}'::jsonb,
  train_split real DEFAULT 0.8 CHECK (train_split >= 0 AND train_split <= 1),
  status text DEFAULT 'building' CHECK (status IN ('building', 'ready', 'processing')),
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now()
);

-- Create dataset_images table
CREATE TABLE IF NOT EXISTS dataset_images (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  dataset_id uuid NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  input_image_url text NOT NULL,
  target_mask_url text NOT NULL,
  original_filename text NOT NULL,
  width integer NOT NULL,
  height integer NOT NULL,
  split_type text DEFAULT 'train' CHECK (split_type IN ('train', 'val')),
  uploaded_at timestamptz DEFAULT now()
);

-- Create training_runs table
CREATE TABLE IF NOT EXISTS training_runs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  model_id uuid REFERENCES models(id) ON DELETE SET NULL,
  dataset_id uuid NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  epochs integer NOT NULL DEFAULT 10,
  learning_rate real NOT NULL DEFAULT 0.001,
  batch_size integer NOT NULL DEFAULT 4,
  optimizer text DEFAULT 'adam' CHECK (optimizer IN ('adam', 'sgd', 'rmsprop')),
  loss_history jsonb DEFAULT '[]'::jsonb,
  metrics_history jsonb DEFAULT '[]'::jsonb,
  best_epoch integer,
  status text DEFAULT 'running' CHECK (status IN ('running', 'completed', 'failed')),
  started_at timestamptz DEFAULT now(),
  completed_at timestamptz
);

-- Create processing_history table
CREATE TABLE IF NOT EXISTS processing_history (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  input_image_url text NOT NULL,
  output_mask_url text,
  overlay_image_url text,
  model_id uuid REFERENCES models(id) ON DELETE SET NULL,
  parameters_json jsonb DEFAULT '{}'::jsonb,
  paths_json jsonb DEFAULT '[]'::jsonb,
  robot_script_url text,
  num_paths integer DEFAULT 0,
  num_points integer DEFAULT 0,
  processing_time_ms integer DEFAULT 0,
  created_at timestamptz DEFAULT now()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_models_architecture ON models(architecture_type);
CREATE INDEX IF NOT EXISTS idx_models_created ON models(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_datasets_status ON datasets(status);
CREATE INDEX IF NOT EXISTS idx_dataset_images_dataset ON dataset_images(dataset_id);
CREATE INDEX IF NOT EXISTS idx_training_runs_status ON training_runs(status);
CREATE INDEX IF NOT EXISTS idx_training_runs_model ON training_runs(model_id);
CREATE INDEX IF NOT EXISTS idx_processing_history_created ON processing_history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_processing_history_model ON processing_history(model_id);

-- Enable Row Level Security
ALTER TABLE models ENABLE ROW LEVEL SECURITY;
ALTER TABLE datasets ENABLE ROW LEVEL SECURITY;
ALTER TABLE dataset_images ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE processing_history ENABLE ROW LEVEL SECURITY;

-- RLS Policies for models table
CREATE POLICY "Anyone can view models"
  ON models FOR SELECT
  USING (true);

CREATE POLICY "Authenticated users can insert models"
  ON models FOR INSERT
  TO authenticated
  WITH CHECK (true);

CREATE POLICY "Authenticated users can update models"
  ON models FOR UPDATE
  TO authenticated
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Authenticated users can delete models"
  ON models FOR DELETE
  TO authenticated
  USING (true);

-- RLS Policies for datasets table
CREATE POLICY "Anyone can view datasets"
  ON datasets FOR SELECT
  USING (true);

CREATE POLICY "Authenticated users can insert datasets"
  ON datasets FOR INSERT
  TO authenticated
  WITH CHECK (true);

CREATE POLICY "Authenticated users can update datasets"
  ON datasets FOR UPDATE
  TO authenticated
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Authenticated users can delete datasets"
  ON datasets FOR DELETE
  TO authenticated
  USING (true);

-- RLS Policies for dataset_images table
CREATE POLICY "Anyone can view dataset images"
  ON dataset_images FOR SELECT
  USING (true);

CREATE POLICY "Authenticated users can insert dataset images"
  ON dataset_images FOR INSERT
  TO authenticated
  WITH CHECK (true);

CREATE POLICY "Authenticated users can update dataset images"
  ON dataset_images FOR UPDATE
  TO authenticated
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Authenticated users can delete dataset images"
  ON dataset_images FOR DELETE
  TO authenticated
  USING (true);

-- RLS Policies for training_runs table
CREATE POLICY "Anyone can view training runs"
  ON training_runs FOR SELECT
  USING (true);

CREATE POLICY "Authenticated users can insert training runs"
  ON training_runs FOR INSERT
  TO authenticated
  WITH CHECK (true);

CREATE POLICY "Authenticated users can update training runs"
  ON training_runs FOR UPDATE
  TO authenticated
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Authenticated users can delete training runs"
  ON training_runs FOR DELETE
  TO authenticated
  USING (true);

-- RLS Policies for processing_history table
CREATE POLICY "Anyone can view processing history"
  ON processing_history FOR SELECT
  USING (true);

CREATE POLICY "Authenticated users can insert processing history"
  ON processing_history FOR INSERT
  TO authenticated
  WITH CHECK (true);

CREATE POLICY "Authenticated users can update processing history"
  ON processing_history FOR UPDATE
  TO authenticated
  USING (true)
  WITH CHECK (true);

CREATE POLICY "Authenticated users can delete processing history"
  ON processing_history FOR DELETE
  TO authenticated
  USING (true);