-- Storage buckets setup for AI Welding Path Generator
-- This file contains SQL commands to set up storage buckets
-- Run these commands in Supabase Dashboard > SQL Editor

-- Create model-weights bucket for storing PyTorch model checkpoint files
INSERT INTO storage.buckets (id, name, public)
VALUES ('model-weights', 'model-weights', true)
ON CONFLICT (id) DO NOTHING;

-- Create dataset-images bucket for storing input images and target masks
INSERT INTO storage.buckets (id, name, public)
VALUES ('dataset-images', 'dataset-images', true)
ON CONFLICT (id) DO NOTHING;

-- Create processed-outputs bucket for storing inference results and robot scripts
INSERT INTO storage.buckets (id, name, public)
VALUES ('processed-outputs', 'processed-outputs', true)
ON CONFLICT (id) DO NOTHING;

-- Storage policies for model-weights bucket
CREATE POLICY "Public can view model weights"
ON storage.objects FOR SELECT
USING (bucket_id = 'model-weights');

CREATE POLICY "Authenticated users can upload model weights"
ON storage.objects FOR INSERT
TO authenticated
WITH CHECK (bucket_id = 'model-weights');

CREATE POLICY "Authenticated users can update model weights"
ON storage.objects FOR UPDATE
TO authenticated
USING (bucket_id = 'model-weights');

CREATE POLICY "Authenticated users can delete model weights"
ON storage.objects FOR DELETE
TO authenticated
USING (bucket_id = 'model-weights');

-- Storage policies for dataset-images bucket
CREATE POLICY "Public can view dataset images"
ON storage.objects FOR SELECT
USING (bucket_id = 'dataset-images');

CREATE POLICY "Authenticated users can upload dataset images"
ON storage.objects FOR INSERT
TO authenticated
WITH CHECK (bucket_id = 'dataset-images');

CREATE POLICY "Authenticated users can update dataset images"
ON storage.objects FOR UPDATE
TO authenticated
USING (bucket_id = 'dataset-images');

CREATE POLICY "Authenticated users can delete dataset images"
ON storage.objects FOR DELETE
TO authenticated
USING (bucket_id = 'dataset-images');

-- Storage policies for processed-outputs bucket
CREATE POLICY "Public can view processed outputs"
ON storage.objects FOR SELECT
USING (bucket_id = 'processed-outputs');

CREATE POLICY "Authenticated users can upload processed outputs"
ON storage.objects FOR INSERT
TO authenticated
WITH CHECK (bucket_id = 'processed-outputs');

CREATE POLICY "Authenticated users can update processed outputs"
ON storage.objects FOR UPDATE
TO authenticated
USING (bucket_id = 'processed-outputs');

CREATE POLICY "Authenticated users can delete processed outputs"
ON storage.objects FOR DELETE
TO authenticated
USING (bucket_id = 'processed-outputs');
