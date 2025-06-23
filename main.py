import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import rasterio

from config import config
from dataset import FullImageHyperspectralDataset
from srnet_model import SpectralReconstructionNet
from train import Trainer
from reference_manager import ReferenceManager
from spect_dict import SpectralDictionary


class HyperspectralProcessor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def load_data(self, num_images=None):
        """Load pre-generated augmented dataset (supports .npy and .tif/.tiff)."""
        # use the path from config
        self.data_dir = config.dataset_path

        # list all files once
        files = sorted(os.listdir(self.data_dir))

        # separate by extension
        npy_files = [f for f in files if f.lower().endswith('.npy')]
        tif_files = [f for f in files if f.lower().endswith(('.tif', '.tiff'))]

        use_npy = len(npy_files) > 0
        chosen = npy_files if use_npy else tif_files

        if num_images is not None:
            chosen = chosen[:num_images]

        all_data = []
        print(f"Loading {len(chosen)} {'.npy' if use_npy else 'TIFF'} files from {self.data_dir}…")

        for fname in chosen:
            path = os.path.join(self.data_dir, fname)
            try:
                if use_npy:
                    cube = np.load(path)                        # (bands, H, W)
                    cube = np.transpose(cube, (1, 2, 0))        # -> (H, W, bands)
                else:
                    import rasterio
                    with rasterio.open(path) as src:
                        data = src.read()                       # (bands, H, W)
                        cube = np.transpose(data, (1, 2, 0))    # -> (H, W, bands)

                all_data.append(cube)
                if len(all_data) % 50 == 0:
                    print(f"  loaded {len(all_data)} images")

            except Exception as e:
                print(f"Error loading {fname}: {e}")

        if not all_data:
            raise ValueError(f"No data could be loaded from {self.data_dir}")

        # stack into shape (N, H, W, C)
        return np.stack(all_data)



    def prepare_datasets(self, data): # 'data' is the stacked numpy array [N_total, H_orig, W_orig, C_orig]
        """Prepare training, validation, and test datasets."""
        total_samples = data.shape[0]
        train_size = int(0.8 * total_samples)
        val_size = int(0.1 * total_samples)

        # Split data (NumPy arrays)
        train_data_np = data[:train_size]
        val_data_np = data[train_size:train_size+val_size]
        test_data_np = data[train_size+val_size:] # This will be empty if train_size + val_size = total_samples
                                                  # Ensure total_samples - (train_size + val_size) >=0 for test_data

        print("\nDataset Information:")
        print(f"Total number of images: {total_samples}")
        # Use train_data_np.shape[0] etc. for actual sizes after splitting
        print(f"Training: {train_data_np.shape[0]} images")
        print(f"Validation: {val_data_np.shape[0]} images")
        print(f"Test: {test_data_np.shape[0]} images") # This will be correct
        print(f"Image dimensions: {data.shape[1]}×{data.shape[2]} pixels")
        print(f"Wavelength points (original input): {data.shape[3]}") # Clarify this is from original input

        # Create datasets
        print("\nCreating datasets...")
        # THE KEY CHANGE: Pass the NumPy arrays directly
        train_dataset = FullImageHyperspectralDataset(train_data_np)
        val_dataset = FullImageHyperspectralDataset(val_data_np)
        # Handle case where test_data_np might be empty if split is exactly 80/10/10 and total_samples is not large enough
        if test_data_np.shape[0] > 0:
            test_dataset = FullImageHyperspectralDataset(test_data_np)
        else:
            print("Warning: Test dataset is empty based on current split percentages and total samples.")
            test_dataset = None # Or an empty dataset if your framework requires it

        # Now print the number of selected wavelengths from the dataset, after processing
        if train_dataset.num_images > 0:
            print(f"Wavelength points (selected for model): {train_dataset.num_selected_wavelengths}")
        elif val_dataset.num_images > 0: # Fallback if train is empty for some reason
            print(f"Wavelength points (selected for model): {val_dataset.num_selected_wavelengths}")


        print("\nVisualizing filter arrangements...")
        if train_dataset.num_images > 0: # Ensure dataset is not empty
            train_dataset.visualize_filter_pattern(num_repeats=3)
            train_dataset.visualize_filter_transmissions()
        else:
            print("Train dataset is empty, skipping visualization.")

        # --- Build spectral dictionary ---
        spectral_dict = None
        # Option 1: Build from References
        print("\nAttempting to build spectral dictionary from reference file...")
        try: # Add try-except for robustness
            if hasattr(config, 'reference_spectra_path') and os.path.exists(config.reference_spectra_path):
                ref_manager = ReferenceManager(config.reference_spectra_path) # ReferenceManager uses config for wavelengths
                reference_data_for_dict = np.array(list(ref_manager.get_all_spectra().values()))
                
                # Determine num_selected_wavelengths from a dataset instance (if available) or config
                num_selected_wl_for_dict = train_dataset.num_selected_wavelengths if train_dataset.num_images > 0 else len(config.wavelength_indices)

                if reference_data_for_dict.shape[0] > 0 and \
                   reference_data_for_dict.shape[1] == num_selected_wl_for_dict:
                    n_comps_ref = min(20, reference_data_for_dict.shape[0], reference_data_for_dict.shape[1])
                    if n_comps_ref < 1: n_comps_ref = 1
                    spectral_dict = SpectralDictionary(n_components=n_comps_ref) # Uses config for wavelengths
                    spectral_dict.build_from_data(reference_data_for_dict, force_rebuild=True)
                    print(f"Built spectral dictionary from {reference_data_for_dict.shape[0]} references, {n_comps_ref} components.")
                else: 
                    print(f"Warning: Reference data for dictionary has {reference_data_for_dict.shape[0]} spectra with {reference_data_for_dict.shape[1]} bands. Expected {num_selected_wl_for_dict} bands matching model output.")
            else:
                print("Warning: Reference spectra path not configured or file not found for dictionary building.")
        except Exception as e:
            print(f"Error building spectral dictionary from references: {e}")


        # Option 2: Fallback to building from Training Data
        if spectral_dict is None:
             print("\nBuilding spectral dictionary from training data samples (fallback)...")
             if train_dataset.num_images > 0:
                 num_train_imgs_for_dict = train_dataset.hypercube.shape[0]
                 sample_img_indices = np.random.choice(
                     num_train_imgs_for_dict, 
                     size=min(num_train_imgs_for_dict, 5),
                     replace=False
                 )
                 
                 spectra_samples_list = []
                 for img_idx in sample_img_indices:
                     img_data_h_w_c = train_dataset.hypercube[img_idx] # [H_pad, W_pad, C_selected] tensor
                     img_data_flat = img_data_h_w_c.reshape(-1, train_dataset.num_selected_wavelengths)
                     
                     pixels_to_sample_count = min(img_data_flat.shape[0], 1000)
                     if pixels_to_sample_count > 0: # Ensure we can sample
                        pixel_sample_indices = np.random.choice(
                            img_data_flat.shape[0], 
                            size=pixels_to_sample_count, 
                            replace=False
                        )
                        spectra_samples_list.append(img_data_flat[pixel_sample_indices].cpu().numpy())
                 
                 if spectra_samples_list:
                     all_samples_np = np.vstack(spectra_samples_list)
                     n_comps_dict = min(20, all_samples_np.shape[0], all_samples_np.shape[1])
                     if n_comps_dict < 1 : n_comps_dict = 1
                     
                     spectral_dict = SpectralDictionary(n_components=n_comps_dict) # Uses config for wavelengths
                     spectral_dict.build_from_data(all_samples_np, force_rebuild=True)
                     print(f"Built spectral dictionary from training data, {n_comps_dict} components.")

                 else: # spectra_samples_list is empty
                     print("Warning: Not enough data to sample for spectral dictionary from training. Using default.")
                     spectral_dict = SpectralDictionary(n_components=min(20, train_dataset.num_selected_wavelengths)).build_default_dictionary()
             else: # train_dataset is empty
                 print("Warning: Training dataset is empty. Using default spectral dictionary.")
                 spectral_dict = SpectralDictionary(n_components=min(20, len(config.wavelength_indices))).build_default_dictionary()
        # --- End Build spectral dictionary ---


        # Create data loaders
        # Keep num_workers=0 and pin_memory=False for initial testing of dataset.py changes
        # Later, you can increase num_workers for POSIX systems.
        num_dataloader_workers = 0 # For initial testing
        # num_dataloader_workers = 4 if os.name == 'posix' else 0 # For production/speed
        
        print(f"Using {num_dataloader_workers} DataLoader workers.")
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                                  num_workers=num_dataloader_workers, pin_memory=(num_dataloader_workers > 0))
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, 
                                num_workers=num_dataloader_workers, pin_memory=(num_dataloader_workers > 0))
        
        if test_dataset: # Only create loader if test_dataset exists
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, 
                                     num_workers=num_dataloader_workers, pin_memory=(num_dataloader_workers > 0))
        else:
            test_loader = None


        return train_loader, val_loader, test_loader, spectral_dict

    def create_srnet_model(self, spectral_dictionary_from_data=None): # Add new argument
        num_wavelengths = config.num_output_wavelengths

        # --- Load Reference Manager ---
        reference_manager = None
        reference_spectra_tensor = None

        try:
            ref_file = config.reference_spectra_path
            if os.path.exists(ref_file):
                # This ReferenceManager is for the model's weighted loss, separate from dict building
                # but could use the same file. It interpolates to config.wavelength_indices.
                temp_ref_manager = ReferenceManager(ref_file)
                ref_dict_model = temp_ref_manager.get_all_spectra()
                if ref_dict_model:
                    ref_list_model = [torch.tensor(spec, dtype=torch.float32) for spec in ref_dict_model.values()]
                    if ref_list_model:
                        reference_spectra_tensor = torch.stack(ref_list_model)
                        ref_norms = torch.linalg.norm(reference_spectra_tensor, dim=1, keepdim=True) + 1e-8
                        reference_spectra_tensor = reference_spectra_tensor / ref_norms
                print(f"Loaded {len(ref_dict_model) if ref_dict_model else 0} reference spectra for model's weighted loss.")
            # ... (error handling for reference_spectra_path) ...
        except Exception as e:
            print(f"Warning: Could not load reference spectra for model loss: {e}")

        model = SpectralReconstructionNet(
            input_channels=1,
            out_channels=num_wavelengths,
            dim=64,
            deep_stage=3,
            num_blocks=[1, 2, 3],
            num_heads=[2, 4, 8],
            reference_spectra=reference_spectra_tensor, # For weighted MSE
            use_spectral_dict=True # To enable spectral_dict_loss
        )

        # Assign the externally built spectral dictionary to the model if provided and model uses it
        if model.use_spectral_dict:
            if spectral_dictionary_from_data:
                model.spectral_dict = spectral_dictionary_from_data
                print("Assigned externally built spectral dictionary to SRNet.")
            elif model.spectral_dict is None: # If model has internal slot but nothing was built/assigned
                print("SRNet has no spectral dictionary; building default internally.")
                # The model will try to build its own default if its self.spectral_dict is None
                # and use_spectral_dict is True. The internal build uses config.wavelength_indices.
                pass # Model's __init__ handles default creation if spectral_dictionary_from_data is None

            # Ensure the model's spectral_dict (whether assigned or internally built) is on the correct device
            if model.spectral_dict:
                target_device = self.device # Or next(model.parameters()).device once model is on device
                # The model's __init__ should handle moving its own dictionary components to its device.
                # If assigning externally, we need to ensure it's moved:
                if spectral_dictionary_from_data: # Only if we assigned it
                    if model.spectral_dict.components is not None:
                        model.spectral_dict.components = model.spectral_dict.components.to(target_device)
                    if model.spectral_dict.mean_spectrum is not None:
                        model.spectral_dict.mean_spectrum = model.spectral_dict.mean_spectrum.to(target_device)
                    model.spectral_dict.device = str(target_device)


        print(f"SRNet Configuration: ...")
        return model

    def visualize_reconstruction(self, model, test_loader, save_dir='results'):
        """Visualize reconstruction results."""
        os.makedirs(save_dir, exist_ok=True)
        model.eval()

        with torch.no_grad():
            # Get a test sample
            filtered_measurements, filter_pattern, original_spectrum = next(iter(test_loader))

            # Move to device
            filtered_measurements = filtered_measurements.to(self.device)
            filter_pattern = filter_pattern.to(self.device)
            original_spectrum = original_spectrum.to(self.device)

            # Perform reconstruction
            reconstructed_spectrum = model(filtered_measurements, filter_pattern)

            # Move to CPU for plotting
            original_spectrum = original_spectrum.cpu().numpy()[0]
            reconstructed_spectrum = reconstructed_spectrum.cpu().numpy()[0]

            # Plot results for wavelengths in range (800-1700nm)
            wavelengths = config.full_wavelengths[config.wavelength_indices]

            # Plot central pixel spectrum
            h, w = original_spectrum.shape[1:]
            center_h, center_w = h//2, w//2

            plt.figure(figsize=(10, 6))
            plt.plot(wavelengths, original_spectrum[:, center_h, center_w], 'b-', label='Original')
            plt.plot(wavelengths, reconstructed_spectrum[:, center_h, center_w], 'r--', label='Reconstructed')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Intensity')
            plt.title('Reconstruction Result (Center Pixel)')
            plt.legend()
            plt.grid(True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.savefig(os.path.join(save_dir, f'reconstruction_{timestamp}.png'))
            plt.close()

            # Plot full image comparison at middle wavelength
            middle_idx = len(wavelengths) // 2
            middle_wavelength = wavelengths[middle_idx]

            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(original_spectrum[middle_idx], cmap='viridis')
            plt.title(f'Original at {middle_wavelength:.0f}nm')
            plt.colorbar()

            plt.subplot(132)
            plt.imshow(reconstructed_spectrum[middle_idx], cmap='viridis')
            plt.title(f'Reconstructed at {middle_wavelength:.0f}nm')
            plt.colorbar()

            plt.subplot(133)
            difference = np.abs(original_spectrum[middle_idx] - reconstructed_spectrum[middle_idx])
            plt.imshow(difference, cmap='viridis')
            plt.title('Absolute Difference')
            plt.colorbar()

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'fullimage_comparison_{timestamp}.png'))
            plt.close()


def main():
    """Main training and evaluation pipeline."""
    num_images = 1000

    print("Starting Hyperspectral Neural Network Training Pipeline with SRNet...")
    print(f"\nConfiguration:")
    print(f"Number of filters: {config.num_filters}")
    print(f"Superpixel arrangement: {config.superpixel_height}×{config.superpixel_width}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Number of epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")

    try:
        processor = HyperspectralProcessor()

        print("\nLoading augmented dataset...")
        all_data = processor.load_data(num_images)

        print("\nPreparing datasets...")
        train_loader, val_loader, test_loader, built_spectral_dict = processor.prepare_datasets(all_data)
        print("\nInitializing SRNet model...")
        model = processor.create_srnet_model(spectral_dictionary_from_data=built_spectral_dict)
        print(f"Model created successfully")

        trainer = Trainer(model, train_loader, val_loader)

        print("\nStarting training...")
        trainer.train()

        print("\nEvaluating RECONSTRUCTION performance...")
        # Evaluation measures how well the trained model reconstructs the spectra
        # Note: 'outputs' and 'targets' from evaluate_model are lists of tensors, potentially large.
        # Consider if you need them directly or just the average metrics.
        test_metrics, _, _ = trainer.evaluate_model(test_loader)
        print(f"Final Test Metrics (Reconstruction):")
        for key, val in test_metrics.items():
             # Format based on metric type for better readability
             if key in ['psnr']:
                 print(f"  {key}: {val:.2f}")
             elif key in ['ssim', 'spectral_fidelity']:
                  print(f"  {key}: {val:.4f}")
             elif key in ['rmse', 'mrae']:
                  print(f"  {key}: {val:.6f}")
             else: # Loss components
                  print(f"  {key}: {val:.6f}")

        print("\nEvaluating model...")
        test_loss, outputs, targets = trainer.evaluate_model(test_loader)
        print(f"Final test loss: {test_loss['total_loss']:.6f}")


        print("\nGenerating visualizations...")
        processor.visualize_reconstruction(model, test_loader)

        print("\nPipeline completed successfully!")

    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
