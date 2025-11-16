-- Create storage buckets for AI welding application

-- Create model-weights bucket
INSERT INTO storage.buckets (id, name, public)
VALUES ('model-weights', 'model-weights', true)
ON CONFLICT (id) DO NOTHING;

-- Create dataset-images bucket
INSERT INTO storage.buckets (id, name, public)
VALUES ('dataset-images', 'dataset-images', true)
ON CONFLICT (id) DO NOTHING;

-- Create processed-outputs bucket
INSERT INTO storage.buckets (id, name, public)
VALUES ('processed-outputs', 'processed-outputs', true)
ON CONFLICT (id) DO NOTHING;
