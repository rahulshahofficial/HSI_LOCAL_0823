import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from config import config
import matplotlib.pyplot as plt

class FullImageHyperspectralDataset(Dataset):
    def __init__(self, hyperspectral_cube_numpy): # Expect NumPy array [N, H, W, C] or [H, W, C]
        self.num_filters = config.num_filters
        self.superpixel_height = config.superpixel_height
        self.superpixel_width = config.superpixel_width

        if len(hyperspectral_cube_numpy.shape) == 4:
            self.num_images, h_orig, w_orig, c_orig = hyperspectral_cube_numpy.shape
        else: # Assuming [H, W, C]
            h_orig, w_orig, c_orig = hyperspectral_cube_numpy.shape
            self.num_images = 1
            hyperspectral_cube_numpy = hyperspectral_cube_numpy.reshape(1, h_orig, w_orig, c_orig)

        max_filters = self.superpixel_height * self.superpixel_width
        if self.num_filters > max_filters:
            raise ValueError(f"Number of filters ({self.num_filters}) exceeds superpixel capacity ({max_filters})")

        # Padding
        pad_h = (self.superpixel_height - (h_orig % self.superpixel_height)) % self.superpixel_height
        pad_w = (self.superpixel_width - (w_orig % self.superpixel_width)) % self.superpixel_width

        self.h_padded = h_orig + pad_h
        self.w_padded = w_orig + pad_w

        if pad_h > 0 or pad_w > 0:
            # Pad using np.pad for clarity and efficiency for potentially large N
            # We pad each image in the batch individually
            padded_cubes_list = []
            for i in range(self.num_images):
                img_to_pad = hyperspectral_cube_numpy[i]
                padded_img = np.pad(
                    img_to_pad,
                    ((0, pad_h), (0, pad_w), (0, 0)), # Pad H, W; not C
                    mode='constant',
                    constant_values=0
                )
                padded_cubes_list.append(padded_img)
            hyperspectral_cube_padded_np = np.stack(padded_cubes_list)
        else:
            hyperspectral_cube_padded_np = hyperspectral_cube_numpy

        # Normalization (per image) and Wavelength Selection
        # Convert to PyTorch tensor early to do this on potentially GPU if data fits
        # For very large datasets, this might be an issue, but for num_images=100, it's usually fine.
        # If num_images is huge, consider doing norm/select in __getitem__ if memory is a constraint.

        # Select wavelengths first using NumPy indexing, then convert to tensor
        selected_data_np = hyperspectral_cube_padded_np[:, :, :, config.wavelength_indices]

        self.hypercube = torch.from_numpy(selected_data_np).float() # [N, H_pad, W_pad, C_selected]

        # Per-image normalization (important for diverse datasets)
        for i in range(self.num_images):
            img_slice = self.hypercube[i]
            max_val = torch.max(img_slice)
            if max_val > 0:
                self.hypercube[i] = img_slice / max_val
            # else: image is all zeros, remains zeros

        self.wavelengths = config.full_wavelengths[config.wavelength_indices]
        self.num_selected_wavelengths = len(self.wavelengths)

        # --- Load Filter Data (once) ---
        # self.filter_matrix will be [num_filters, C_selected]
        filters_pd_data = pd.read_csv(config.filter_path, header=None)
        filter_transmissions_np = filters_pd_data.iloc[:self.num_filters, 1:].values
        csv_wavelengths_np = np.linspace(800, 1700, filter_transmissions_np.shape[1])

        interpolated_filters_list = []
        for filter_spectrum_np in filter_transmissions_np:
            interp_np = np.interp(self.wavelengths, csv_wavelengths_np, filter_spectrum_np)
            interpolated_filters_list.append(interp_np)
        self.filter_matrix = torch.tensor(np.array(interpolated_filters_list), dtype=torch.float32)

        # --- Create Tiled Filter Index Map (once) ---
        # This map indicates which of the `num_filters` is applied at each pixel
        sp_h_template, sp_w_template = self.superpixel_height, self.superpixel_width
        superpixel_indices_template = torch.zeros((sp_h_template, sp_w_template), dtype=torch.long)
        for di_ in range(sp_h_template):
            for dj_ in range(sp_w_template):
                superpixel_indices_template[di_, dj_] = (di_ * sp_w_template + dj_) % self.num_filters

        num_h_tiles = self.h_padded // sp_h_template
        num_w_tiles = self.w_padded // sp_w_template
        # self.pixel_filter_indices_map is [H_pad, W_pad]
        self.pixel_filter_indices_map = superpixel_indices_template.tile(num_h_tiles, num_w_tiles)

        # --- Pre-compute Filter Pattern Base (once) ---
        # This gathers the full spectral curves for the tiled pattern
        # filter_pattern_base_h_w_c: [H_pad, W_pad, C_selected]
        self.filter_pattern_base_h_w_c = self.filter_matrix[self.pixel_filter_indices_map]
        # self.filter_pattern_for_model: [C_selected, H_pad, W_pad]
        self.filter_pattern_for_model = self.filter_pattern_base_h_w_c.permute(2, 0, 1)

    # load_filter_data and _interpolate_filters are effectively in __init__ now

    def create_filtered_measurements(self, idx): # idx is the image index
        """Generate filtered measurements for a specific image using pre-computed tensors."""
        # self.hypercube[idx] is [H_pad, W_pad, C_selected]
        # self.filter_pattern_base_h_w_c is [H_pad, W_pad, C_selected] (same for all images)

        # Element-wise multiplication and sum over the channel dimension
        measurements_h_w = torch.sum(
            self.hypercube[idx] * self.filter_pattern_base_h_w_c,
            dim=-1
        ) # Result is [H_pad, W_pad]

        return measurements_h_w.unsqueeze(0) # Add channel dim: [1, H_pad, W_pad]

    def create_filter_pattern(self, idx): # idx is unused as pattern is same for all images
        """Return the pre-computed filter pattern."""
        # self.filter_pattern_for_model is [C_selected, H_pad, W_pad]
        return self.filter_pattern_for_model

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # Retrieve pre-processed ground truth for this image
        # self.hypercube[idx] is [H_pad, W_pad, C_selected]
        # We need to return [C_selected, H_pad, W_pad] for ground_truth
        ground_truth = self.hypercube[idx].permute(2, 0, 1)

        # Generate filtered measurements for this specific image's hyperspectral data
        filtered_measurements = self.create_filtered_measurements(idx)

        # Get the (shared) filter pattern
        filter_pattern = self.create_filter_pattern(idx) # idx is not strictly needed here

        return filtered_measurements, filter_pattern, ground_truth

    # visualize_filter_pattern and visualize_filter_transmissions can remain as they are
    # but ensure they use self.h_padded, self.w_padded if visualizing full image pattern
    def visualize_filter_pattern(self, num_repeats=3): # num_repeats might be misleading now
        """Visualize the filter arrangement in superpixels"""
        base_pattern_np = np.zeros((self.superpixel_height, self.superpixel_width))
        for i in range(self.superpixel_height):
            for j in range(self.superpixel_width):
                filter_idx = (i * self.superpixel_width + j) % self.num_filters
                base_pattern_np[i, j] = filter_idx + 1 # 1-based for display

        # Use the pre-computed tiled map for the full pattern
        # self.pixel_filter_indices_map is 0-based, add 1 for display
        full_pattern_np = (self.pixel_filter_indices_map.numpy() + 1)

        # Dynamically import matplotlib only when needed for visualization
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 8))
        plt.subplot(121)
        plt.imshow(base_pattern_np, cmap='viridis', vmin=1, vmax=self.num_filters if self.num_filters > 0 else 1)
        plt.title(f'Single Superpixel Pattern ({self.superpixel_height}x{self.superpixel_width})')
        plt.colorbar(label='Filter Index (1-based)')

        for i in range(self.superpixel_height):
            for j in range(self.superpixel_width):
                plt.text(j, i, f'F{int(base_pattern_np[i,j])}',
                        ha='center', va='center', color='white')

        plt.subplot(122)
        plt.imshow(full_pattern_np, cmap='viridis', vmin=1, vmax=self.num_filters if self.num_filters > 0 else 1)
        num_repeats_h_actual = self.h_padded // self.superpixel_height
        num_repeats_w_actual = self.w_padded // self.superpixel_width
        plt.title(f'Full Padded Image Pattern ({num_repeats_h_actual}x{num_repeats_w_actual} superpixels)')
        plt.colorbar(label='Filter Index (1-based)')

        # Draw grid lines based on actual tiling, limit to `num_repeats` if it's small for clarity
        h_lines_to_draw = min(num_repeats_h_actual + 1, num_repeats + 1 if num_repeats else num_repeats_h_actual + 1)
        w_lines_to_draw = min(num_repeats_w_actual + 1, num_repeats + 1 if num_repeats else num_repeats_w_actual + 1)

        for i in range(h_lines_to_draw):
            plt.axhline(y=i*self.superpixel_height - 0.5, color='w', linestyle='-', alpha=0.5)
        for i in range(w_lines_to_draw):
            plt.axvline(x=i*self.superpixel_width - 0.5, color='w', linestyle='-', alpha=0.5)

        plt.tight_layout()
        plt.show()

    def visualize_filter_transmissions(self):
        """Visualize the spectral transmission curves of all filters"""
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        for i in range(self.num_filters):
            # self.filter_matrix is [num_filters, C_selected] tensor
            plt.plot(self.wavelengths, self.filter_matrix[i].cpu().numpy(), # Move to CPU for numpy
                    label=f'Filter {i+1}', linewidth=2)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Transmission')
        plt.ylim(0, 1)
        plt.title('Filter Transmission Spectra')
        plt.grid(True)
        plt.legend()
        plt.show()
