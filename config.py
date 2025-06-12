import os
import platform
import numpy as np


class Config:
    def __init__(self):
        # Set base paths based on operating system
        if platform.system() == 'Windows':
            self.base_path = r"V:\SimulationData\Rahul\Hyperspectral Imaging Project"
        else:  # macOS
            self.base_path = '/Volumes/ValentineLab/SimulationData/Rahul/Hyperspectral Imaging Project'

        # Set model path to the central Models folder
        if platform.system() == 'Windows':
            self.models_path = os.path.join(self.base_path, "Machine Learning Codes", "Models")
        else:  # macOS
            self.models_path = os.path.join(self.base_path, "Machine Learning Codes", "Models")

        # # Dataset paths
        self.dataset_path = os.path.join(self.base_path, 'HSI Data Sets',
                                       'Plastics HSI Dataset',
                                       # 'Wavelength 16',
                                       # # '043025_SyntheticHSI_Images_512x512_16wl_grids')
                                       # 'Wavelength 32',
                                       # '043025_SyntheticHSI_Images_256x256_32wl_grids')
                                       # "051225_SyntheticHSI_Images_512p_32wl_grids")
                                       # "051225_SyntheticHSI_Images_1024p_32wl_grids")
                                       # "051225_SyntheticHSI_Images_2048p_32wl_grids")
                                       # "051225_SyntheticHSI_Images_256_32wl_overlap")
                                       # "051225_SyntheticHSI_Images_256_32wl_bottles_and_random_grid")
                                       'Wavelength 64',
                                       '051225_SyntheticHSI_Images_256_64wl_bottles_and_random_grid')
                                       # '052725_SyntheticHSI_Images_512_64wl_bottles_and_random_grid')


        # # Dataset paths
        # self.dataset_path = os.path.join(self.base_path, 'HSI Data Sets',
        #                                'AVIRIS_augmented_dataset_2_npy)
        self.wavelength_path = os.path.join(self.dataset_path, 'wavelengths.csv')

        self.reference_spectra_path = os.path.join(self.base_path, 'HSI Data Sets',
                                       'Plastics HSI Dataset', 
                                       'reference_spectra.csv'),
                                       # 'reference_spectra_modified_64pts.csv')


        # Filter paths
        self.filter_path = os.path.join(self.base_path, 'Machine Learning Codes',
                                      'Filter CSV files',
                                      # 'TransmissionTable_NIR_smooth.csv')
                                      'Correlation Selected Filters', 
                                      # '16 Filters', '16Filters_smoothened.csv')
                                      '9 Filters', '9Filters_smoothened.csv')


        # Model parameters
        self.batch_size = 16
        self.num_epochs = 10
        self.learning_rate = 1e-5
        # self.num_filters = 9
        self.num_filters = 9
        self.superpixel_height = 3
        self.superpixel_width = 3 # Change superpixel size here
        

        # Modified parameters for AVIRIS dataset
        self.image_height = 256  # For our cropped images
        self.image_width = 256
        # Modified parameters for 100-wavelength synthetic data
        self.num_wavelengths = 64  # Changed from 220 or 900
        self.wavelength_range = (800, 1700)  # nm, keep same range

        # Synthetic bands run 800,801,…,1700 nm
        # Update wavelength indices and setup
        self.full_wavelengths = np.linspace(800, 1700, self.num_wavelengths)  # n evenly spaced points
        self.wavelength_indices = np.arange(len(self.full_wavelengths))  # All n indices
        self.num_output_wavelengths = len(self.wavelength_indices)  # n bands


        self.input_channels = 1
        self.kernel_size = 3
        self.padding = 1
        self.use_batch_norm = True

        self.conv_channels = [1, 128, 256]  # 3D Conv channels

        # Updated model save path to use centralized location
        # Model save information: Date_Size_WL_Spectial_Test#

        # self.model_save_path = os.path.join(self.models_path, '050825_256p_32wl_Optimized_Main_Dataset_Plots_py_gemini_Test3.pth')
        self.model_save_path = os.path.join(self.models_path, '061225_256p_64wl_16filters_SharpSpectraTest1.pth')
        # self.model_save_path = os.path.join(self.models_path, 'test.pth')

        self.results_path = 'results/061225'

config = Config()
print(f"Number of wavelength indices: {len(config.wavelength_indices)}")
print(f"Range of indices: {min(config.wavelength_indices)} to {max(config.wavelength_indices)}")
print(f"Actual wavelengths: {config.full_wavelengths[config.wavelength_indices[0]]} to {config.full_wavelengths[config.wavelength_indices[-1]]}")
