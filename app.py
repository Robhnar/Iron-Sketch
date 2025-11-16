#!/usr/bin/env python3
"""
AI Welding Path Generator - IronSketch
Complete Streamlit application for CNN-based welding path generation.
Uses local CSV files for data persistence.

Usage:
    streamlit run app.py
"""

import streamlit as st
import torch
import cv2
import numpy as np
import time
from datetime import datetime
import io
import zipfile
import json
from PIL import Image
from pathlib import Path

from models import ModelFactory, get_device, save_model, load_model, get_model_info, model_to_bytes, load_model_from_bytes
from utils import ImageProcessor, Vectorizer, RobotScriptGenerator
from utils.csv_manager import CSVManager
from training import WeldingDataset, Trainer, CombinedLoss
from torch.utils.data import DataLoader

st.set_page_config(
    page_title="AI Welding Path Generator - IronSketch",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def get_csv_manager():
    """Get cached CSV manager."""
    try:
        return CSVManager("data")
    except Exception as e:
        st.error(f"Failed to initialize CSV manager: {e}")
        return None


def initialize_pretrained_models():
    """Initialize pre-trained models in the database."""
    csv_manager = get_csv_manager()
    if not csv_manager:
        return

    models = csv_manager.list_models()

    hf_exists = any(m['architecture_type'] == 'deeplabv3_hf' for m in models)
    google_exists = any(m['architecture_type'] == 'deeplabv3_google' for m in models)

    if not hf_exists:
        csv_manager.add_model(
            name="SegFormer Pre-trained",
            architecture_type="deeplabv3_hf",
            backbone="MiT-B0",
            parameters_json={},
            performance_metrics={'pretrained': 'ADE20K'},
            file_path="",
            file_size_mb=3.7,
            is_pretrained=True
        )
        st.success("Added Hugging Face SegFormer model")

    if not google_exists:
        csv_manager.add_model(
            name="DeepLabV3 MobileNetV2 (Google)",
            architecture_type="deeplabv3_google",
            backbone="MobileNetV2",
            parameters_json={},
            performance_metrics={'pretrained': 'PASCAL VOC 2012'},
            file_path="",
            file_size_mb=2.1,
            is_pretrained=True
        )
        st.success("Added Google DeepLabV3 MobileNetV2 model")


def tab_generate_paths():
    """Tab 1: Generate Paths - Main inference interface."""
    st.header("🎯 Generate Welding Paths")
    st.markdown("Upload an image and generate AI-powered welding paths with configurable robot parameters.")

    csv_manager = get_csv_manager()
    if not csv_manager:
        st.error("CSV Manager not initialized")
        return

    initialize_pretrained_models()

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload Image (JPG/PNG)",
            type=['jpg', 'jpeg', 'png'],
            help="Image will be auto-resized to 256x384 pixels"
        )

    with col2:
        models = csv_manager.list_models()

        if not models:
            st.warning("⚠️ No models available. Please train a model first in the 'Train Models' tab.")
            return

        model_options = {f"{m['name']} ({m['architecture_type']})": m for m in models}
        selected_model_name = st.selectbox("Select Model", options=list(model_options.keys()))
        selected_model = model_options[selected_model_name]

    with st.sidebar:
        st.header("⚙️ Configuration")

        st.subheader("Speed Control")
        speed_mm_s = st.slider(
            "Welding Speed (mm/s)",
            min_value=10.0,
            max_value=30.0,
            value=20.0,
            step=1.0,
            help="Lower speed = thicker lines, Higher speed = thinner lines"
        )
        st.caption(RobotScriptGenerator.map_speed_to_thickness(speed_mm_s))

        st.subheader("Coordinate System")
        scale_mm_per_px = st.number_input(
            "mm per pixel",
            min_value=0.1,
            max_value=10.0,
            value=0.5,
            step=0.1
        )
        origin_x = st.number_input("Origin X offset (mm)", value=0.0, step=10.0)
        origin_y = st.number_input("Origin Y offset (mm)", value=0.0, step=10.0)
        z_height = st.number_input("Z height (mm)", value=50.0, step=5.0)

        st.subheader("Post-Processing")
        close_kernel = st.slider("Closing kernel size", 1, 7, 3, 2)
        min_area = st.slider("Min contour area (px)", 50, 500, 100, 50)
        simplify_eps = st.slider("Simplification epsilon", 0.5, 5.0, 1.0, 0.5)
        use_skeleton = st.checkbox("Use skeletonization", value=False)

        process_button = st.button("🚀 Generate Paths", type="primary", use_container_width=True)

    if uploaded_file and process_button:
        start_time = time.time()

        with st.spinner("Processing..."):
            file_bytes = uploaded_file.read()
            image = ImageProcessor.load_image_from_bytes(file_bytes)
            image_rgb = ImageProcessor.convert_bgr_to_rgb(image)

            st.subheader("📸 Input Image")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(image_rgb, caption="Original Image", use_container_width=True)

            resized = ImageProcessor.resize_with_aspect_ratio(image)
            resized_rgb = ImageProcessor.convert_bgr_to_rgb(resized)

            with col2:
                st.image(resized_rgb, caption="Resized (256x384)", use_container_width=True)

            device = get_device()
            model = ModelFactory.create_model(selected_model['architecture_type'], pretrained=False)

            if selected_model['file_path'] and Path(selected_model['file_path']).exists():
                try:
                    model = load_model(selected_model['file_path'], model, device)
                except Exception as e:
                    st.error(f"Failed to load model: {e}")
                    return
            elif selected_model['is_pretrained']:
                model.to(device)
            else:
                st.error("Model file not found")
                return

            normalized = ImageProcessor.normalize_image(resized)
            input_tensor = torch.from_numpy(normalized).permute(2, 0, 1).float().unsqueeze(0).to(device)

            model.eval()
            with torch.no_grad():
                output = model(input_tensor)
                if output.shape[1] == 1:
                    prob_mask = torch.sigmoid(output).squeeze().cpu().numpy()
                else:
                    prob_mask = output.squeeze().cpu().numpy()

            binary_mask = ImageProcessor.binary_threshold(prob_mask)

            processed_mask = ImageProcessor.post_process_mask(
                binary_mask,
                close_kernel_size=close_kernel,
                min_area=min_area
            )

            if use_skeleton:
                processed_mask = ImageProcessor.skeletonize(processed_mask)

            with col3:
                st.image(processed_mask, caption="AI Generated Mask", use_container_width=True)

            st.subheader("🔧 Vectorization")

            vectorizer = Vectorizer()
            paths = vectorizer.vectorize_mask(processed_mask, simplify_epsilon=simplify_eps)

            if paths:
                paths = vectorizer.optimize_path_order(paths)

            stats = vectorizer.get_path_statistics(paths)

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Paths", stats['num_paths'])
            col2.metric("Total Points", stats['total_points'])
            col3.metric("Avg Points/Path", f"{stats['average_points_per_path']:.1f}")
            col4.metric("Total Distance", f"{stats['total_distance_pixels']:.0f} px")

            overlay = vectorizer.draw_paths_on_image(resized_rgb, paths, color=(0, 255, 0), thickness=2)
            st.image(overlay, caption="Vector Overlay", use_container_width=True)

            st.subheader("📜 Robot Script")

            transformed_paths = vectorizer.transform_coordinates(
                paths,
                scale_mm_per_px,
                origin_x,
                origin_y,
                invert_y=True,
                image_height=256
            )

            abb_script = RobotScriptGenerator.generate_abb_script(
                transformed_paths,
                z_height,
                speed_mm_s,
                scale_mm_per_px,
                origin_x,
                origin_y
            )

            gcode = RobotScriptGenerator.generate_gcode(
                transformed_paths,
                z_height,
                speed_mm_s * 60
            )

            params_file = RobotScriptGenerator.generate_parameter_file(
                selected_model['architecture_type'],
                speed_mm_s,
                scale_mm_per_px,
                origin_x,
                origin_y,
                z_height,
                stats['num_paths'],
                stats['total_points'],
                simplify_eps,
                {'close_kernel': close_kernel, 'min_area': min_area}
            )

            processing_time = int((time.time() - start_time) * 1000)

            col1, col2, col3 = st.columns(3)

            with col1:
                st.download_button(
                    "📄 Download ABB Script",
                    abb_script,
                    file_name="robot_script.script",
                    mime="text/plain"
                )

            with col2:
                st.download_button(
                    "📄 Download G-code",
                    gcode,
                    file_name="plasma_cut.nc",
                    mime="text/plain"
                )

            with col3:
                st.download_button(
                    "📋 Download Parameters",
                    params_file,
                    file_name="parameters.txt",
                    mime="text/plain"
                )

            with st.expander("Preview ABB Script"):
                st.code(abb_script[:1000] + ("..." if len(abb_script) > 1000 else ""), language="text")

            csv_manager.log_processing(
                model_id=selected_model['id'],
                parameters={'speed': speed_mm_s, 'scale': scale_mm_per_px},
                num_paths=stats['num_paths'],
                num_points=stats['total_points'],
                processing_time_ms=processing_time
            )

            st.success(f"✅ Processing complete in {processing_time}ms!")


def tab_train_models():
    """Tab 2: Train Models - Model training interface."""
    st.header("🎓 Train Models")
    st.markdown("Train CNN models on your custom datasets with real-time progress monitoring.")

    csv_manager = get_csv_manager()
    if not csv_manager:
        return

    datasets = csv_manager.list_datasets(status='ready')

    if not datasets:
        st.warning("⚠️ No datasets available. Create a dataset first in the 'Dataset Builder' tab.")
        return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Configuration")

        architecture = st.selectbox(
            "Architecture",
            options=['unet', 'deeplabv3plus', 'fcn8s', 'deeplabv3plus_resnet50'],
            format_func=lambda x: ModelFactory.get_architecture_info(x)['name']
        )

        arch_info = ModelFactory.get_architecture_info(architecture)
        st.info(f"**{arch_info['name']}**\n\n"
                f"Encoder: {arch_info['encoder']}\n\n"
                f"Parameters: {arch_info['params']}\n\n"
                f"Best for: {arch_info['best_for']}")

        model_name = st.text_input("Model Name", value=f"{architecture}_model_{datetime.now().strftime('%Y%m%d')}")

    with col2:
        st.subheader("Training Configuration")

        dataset_options = {d['name']: d for d in datasets}
        selected_dataset_name = st.selectbox("Select Dataset", options=list(dataset_options.keys()))
        selected_dataset = dataset_options[selected_dataset_name]

        epochs = st.number_input("Epochs", min_value=1, max_value=100, value=20)
        batch_size = st.selectbox("Batch Size", options=[1, 2, 4, 8], index=2)
        learning_rate = st.selectbox("Learning Rate", options=[0.0001, 0.001, 0.01], index=1)
        optimizer_type = st.selectbox("Optimizer", options=['adam', 'sgd', 'rmsprop'])
        early_stopping = st.number_input("Early Stopping Patience", min_value=3, max_value=20, value=5)

    train_button = st.button("🚀 Start Training", type="primary", use_container_width=True)

    if train_button:
        with st.spinner("Initializing training..."):
            device = get_device()
            st.info(f"Using device: {device}")

            model = ModelFactory.create_model(architecture, pretrained=True)
            model_info = get_model_info(model)
            st.success(f"✓ Model created with {model_info['parameter_count_formatted']} parameters")

            train_images = csv_manager.get_dataset_images(selected_dataset['id'], split_type='train')
            val_images = csv_manager.get_dataset_images(selected_dataset['id'], split_type='val')

            if not train_images or not val_images:
                st.error("Dataset has no images or missing train/val split")
                return

            st.info(f"Train: {len(train_images)} images, Val: {len(val_images)} images")

            augmentation = WeldingDataset.create_augmentation_pipeline()
            train_dataset = WeldingDataset(train_images, augmentation=augmentation)
            val_dataset = WeldingDataset(val_images)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            criterion = CombinedLoss()

            if optimizer_type == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            elif optimizer_type == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
            else:
                optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

            trainer = Trainer(model, device, criterion, optimizer)

            training_run = csv_manager.create_training_run(
                dataset_id=selected_dataset['id'],
                epochs=epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                optimizer=optimizer_type
            )

            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_placeholder = st.empty()

            def progress_callback(epoch, train_loss, val_metrics, epoch_time):
                progress = epoch / epochs
                progress_bar.progress(progress)
                status_text.text(f"Epoch {epoch}/{epochs} - {epoch_time:.2f}s")

                with metrics_placeholder.container():
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Train Loss", f"{train_loss:.4f}")
                    col2.metric("Val Loss", f"{val_metrics['loss']:.4f}")
                    col3.metric("IoU", f"{val_metrics['iou']:.4f}")
                    col4.metric("Dice", f"{val_metrics['dice']:.4f}")

            history = trainer.train(
                train_loader,
                val_loader,
                epochs,
                early_stopping_patience=early_stopping,
                progress_callback=progress_callback
            )

            st.success(f"✅ Training complete! Best epoch: {history['best_epoch']}")

            model_dir = csv_manager.models_dir
            model_dir.mkdir(exist_ok=True)
            model_filename = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
            model_path = model_dir / model_filename

            save_model(model, str(model_path), architecture, {
                'epochs_trained': len(history['train_losses']),
                'best_epoch': history['best_epoch']
            })

            file_size_mb = model_path.stat().st_size / (1024 * 1024)

            model_record = csv_manager.add_model(
                name=model_name,
                architecture_type=architecture,
                backbone=arch_info['encoder'],
                parameters_json={
                    'epochs': epochs,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'optimizer': optimizer_type
                },
                performance_metrics={
                    'best_val_loss': history['best_val_loss'],
                    'best_epoch': history['best_epoch'],
                    'final_iou': history['val_metrics'][-1]['iou'],
                    'final_dice': history['val_metrics'][-1]['dice']
                },
                file_path=str(model_path),
                file_size_mb=file_size_mb,
                is_pretrained=False
            )

            csv_manager.update_training_run(training_run['id'], {
                'model_id': model_record['id'],
                'status': 'completed',
                'loss_history': history['train_losses'] + history['val_losses'],
                'metrics_history': history['val_metrics'],
                'best_epoch': history['best_epoch'],
                'completed_at': datetime.utcnow().isoformat()
            })

            st.balloons()
            st.success(f"Model '{model_name}' saved successfully to {model_path}!")


def tab_dataset_builder():
    """Tab 3: Dataset Builder - Interactive dataset creation."""
    st.header("📚 Dataset Builder")
    st.markdown("Create and manage datasets for training CNN models.")

    csv_manager = get_csv_manager()
    if not csv_manager:
        return

    tab1, tab2 = st.tabs(["Create Dataset", "View Datasets"])

    with tab1:
        st.subheader("Create New Dataset")

        dataset_name = st.text_input("Dataset Name", value=f"dataset_{datetime.now().strftime('%Y%m%d')}")
        dataset_description = st.text_area("Description", value="Custom welding path dataset")

        col1, col2 = st.columns(2)

        with col1:
            input_images = st.file_uploader(
                "Upload Input Images",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                help="Original color images"
            )

        with col2:
            target_masks = st.file_uploader(
                "Upload Target Masks",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                help="Binary masks (white paths on black background)"
            )

        train_split = st.slider("Training Split %", 50, 95, 80, 5)

        if st.button("Create Dataset", type="primary"):
            if not input_images or not target_masks:
                st.error("Please upload both input images and target masks")
                return

            if len(input_images) != len(target_masks):
                st.error(f"Number of inputs ({len(input_images)}) must match targets ({len(target_masks)})")
                return

            with st.spinner("Creating dataset..."):
                dataset = csv_manager.create_dataset(
                    name=dataset_name,
                    description=dataset_description,
                    train_split=train_split / 100.0
                )

                progress_bar = st.progress(0)

                num_train = int(len(input_images) * train_split / 100.0)

                dataset_dir = csv_manager.datasets_dir / str(dataset['id'])
                input_dir = dataset_dir / "input"
                target_dir = dataset_dir / "target"

                for idx, (input_file, target_file) in enumerate(zip(input_images, target_masks)):
                    input_bytes = input_file.read()
                    target_bytes = target_file.read()

                    input_image = ImageProcessor.load_image_from_bytes(input_bytes)
                    target_image = ImageProcessor.load_image_from_bytes(target_bytes)

                    resized_input = ImageProcessor.resize_with_aspect_ratio(input_image)
                    resized_target = ImageProcessor.resize_with_aspect_ratio(target_image)

                    input_path = input_dir / f"input_{idx}.png"
                    target_path = target_dir / f"target_{idx}.png"

                    cv2.imwrite(str(input_path), resized_input)
                    cv2.imwrite(str(target_path), resized_target)

                    split_type = 'train' if idx < num_train else 'val'

                    csv_manager.add_dataset_image(
                        dataset_id=dataset['id'],
                        input_path=str(input_path),
                        target_path=str(target_path),
                        original_filename=input_file.name,
                        width=384,
                        height=256,
                        split_type=split_type
                    )

                    progress_bar.progress((idx + 1) / len(input_images))

                csv_manager.update_dataset(dataset['id'], {
                    'num_images': len(input_images),
                    'status': 'ready'
                })

                st.success(f"✅ Dataset '{dataset_name}' created with {len(input_images)} image pairs!")
                st.balloons()

    with tab2:
        st.subheader("Existing Datasets")

        datasets = csv_manager.list_datasets()

        if not datasets:
            st.info("No datasets yet. Create one in the 'Create Dataset' tab.")
            return

        for dataset in datasets:
            with st.expander(f"📁 {dataset['name']} ({dataset['num_images']} images)"):
                col1, col2, col3 = st.columns(3)
                col1.write(f"**Status:** {dataset['status']}")
                col2.write(f"**Train Split:** {int(dataset['train_split'] * 100)}%")
                col3.write(f"**Created:** {dataset['created_at'][:10]}")

                st.write(f"**Description:** {dataset['description']}")

                images = csv_manager.get_dataset_images(dataset['id'])
                if images:
                    st.write(f"**Sample Images:**")
                    cols = st.columns(min(4, len(images)))
                    for idx, img in enumerate(images[:4]):
                        with cols[idx]:
                            if Path(img['input_path']).exists():
                                st.image(str(img['input_path']), caption=f"Input {idx+1}", use_container_width=True)


def tab_batch_processing():
    """Tab 4: Batch Processing - Process multiple images."""
    st.header("⚡ Batch Processing")
    st.markdown("Process multiple images in batch with the same configuration.")

    csv_manager = get_csv_manager()
    if not csv_manager:
        return

    models = csv_manager.list_models()

    if not models:
        st.warning("⚠️ No models available.")
        return

    col1, col2 = st.columns(2)

    with col1:
        uploaded_files = st.file_uploader(
            "Upload Multiple Images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True
        )

    with col2:
        model_options = {f"{m['name']} ({m['architecture_type']})": m for m in models}
        selected_model_name = st.selectbox("Select Model", options=list(model_options.keys()))
        selected_model = model_options[selected_model_name]

        speed_mm_s = st.slider("Speed (mm/s)", 10.0, 30.0, 20.0, 1.0)
        scale_mm_per_px = st.number_input("mm/px", 0.1, 10.0, 0.5, 0.1)

    if uploaded_files and st.button("Process Batch", type="primary"):
        with st.spinner(f"Processing {len(uploaded_files)} images..."):
            device = get_device()
            model = ModelFactory.create_model(selected_model['architecture_type'], pretrained=False)

            if selected_model['file_path'] and Path(selected_model['file_path']).exists():
                model = load_model(selected_model['file_path'], model, device)
            elif selected_model['is_pretrained']:
                model.to(device)

            progress_bar = st.progress(0)
            results = []

            for idx, uploaded_file in enumerate(uploaded_files):
                file_bytes = uploaded_file.read()
                image = ImageProcessor.load_image_from_bytes(file_bytes)
                resized = ImageProcessor.resize_with_aspect_ratio(image)

                normalized = ImageProcessor.normalize_image(resized)
                input_tensor = torch.from_numpy(normalized).permute(2, 0, 1).float().unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(input_tensor)
                    if output.shape[1] == 1:
                        prob_mask = torch.sigmoid(output).squeeze().cpu().numpy()
                    else:
                        prob_mask = output.squeeze().cpu().numpy()

                binary_mask = ImageProcessor.binary_threshold(prob_mask)
                processed_mask = ImageProcessor.post_process_mask(binary_mask)

                vectorizer = Vectorizer()
                paths = vectorizer.vectorize_mask(processed_mask)
                stats = vectorizer.get_path_statistics(paths)

                results.append({
                    'filename': uploaded_file.name,
                    'num_paths': stats['num_paths'],
                    'num_points': stats['total_points']
                })

                progress_bar.progress((idx + 1) / len(uploaded_files))

            st.success(f"✅ Processed {len(uploaded_files)} images!")

            st.subheader("Results")
            for result in results:
                st.write(f"**{result['filename']}**: {result['num_paths']} paths, {result['num_points']} points")


def main():
    st.title("🤖 AI Welding Path Generator - IronSketch")
    st.markdown("Transform images into robot-executable welding paths using CNN architectures")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Generate Paths",
        "🎓 Train Models",
        "📚 Dataset Builder",
        "⚡ Batch Processing"
    ])

    with tab1:
        tab_generate_paths()

    with tab2:
        tab_train_models()

    with tab3:
        tab_dataset_builder()

    with tab4:
        tab_batch_processing()


if __name__ == "__main__":
    main()
