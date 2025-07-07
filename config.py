import os
import platform
import numpy as np


class Config:
    def __init__(self):

        # ## CLOUD CODE
        # # Set base paths based on operating system
        # if platform.system() == 'Windows':
        #     self.base_path = r"V:\SimulationData\Rahul\Hyperspectral Imaging Project"
        # else:  # macOS
        #     self.base_path = '/Volumes/ValentineLab/SimulationData/Rahul/Hyperspectral Imaging Project'

        ## LOCAL CODE PC Workstation 4    
            # Set base paths based on operating system
        if platform.system() == 'Windows':
            self.base_path = r"D:\Rahul"
        else:  # macOS
            self.base_path = '/Volumes/ValentineLab/SimulationData/Rahul/Hyperspectral Imaging Project'

        # Set model path to the central Models folder
        if platform.system() == 'Windows':
            self.models_path = os.path.join(self.base_path, "Models")
        else:  # macOS
            self.models_path = os.path.join(self.base_path, "Machine Learning Codes", "Models")

        # # # Dataset paths CLOUD
        # self.dataset_path = os.path.join(self.base_path, 'HSI Data Sets',
        #                                'Plastics HSI Dataset',
        #                                # 'Wavelength 16',
        #                                # # '043025_SyntheticHSI_Images_512x512_16wl_grids')
        #                                # 'Wavelength 32',
        #                                # '043025_SyntheticHSI_Images_256x256_32wl_grids')
        #                                # "051225_SyntheticHSI_Images_512p_32wl_grids")
        #                                # "051225_SyntheticHSI_Images_1024p_32wl_grids")
        #                                # "051225_SyntheticHSI_Images_2048p_32wl_grids")
        #                                # "051225_SyntheticHSI_Images_256_32wl_overlap")
        #                                # "051225_SyntheticHSI_Images_256_32wl_bottles_and_random_grid")
        #                                # 'Wavelength 64',
        #                                'Wavelength 96',
        #                                '061425_SyntheticHSI_Images_256_96wl_SHARP_2Peaks_bottles_and_random_grid')
        #                                # '061225_SyntheticHSI_Images_256_64wl_SHARP_bottles_and_random_grid')
        #                                # '052725_SyntheticHSI_Images_512_64wl_bottles_and_random_grid')

        # # Dataset paths LOCAL CODE PC Workstation 4
        self.dataset_path = os.path.join(self.base_path, 'Datasets',
                                        '070725NPY Dataset')
                                       # '061425_SyntheticHSI_Images_256_96wl_SHARP_2Peaks_bottles_and_random_grid')
                                       # '062325_SyntheticHSI_Images_256_96wl_SHARP_2Peaks_bottles_and_random_grid_with_random_objects')

        self.reference_spectra_path = os.path.join(self.base_path,'Datasets',
                                                  'sharper_reference_spectra_2Peaks_96pts.csv')

                                       


        # # Dataset paths
        # self.dataset_path = os.path.join(self.base_path, 'HSI Data Sets',
        #                                'AVIRIS_augmented_dataset_2_npy)
        self.wavelength_path = os.path.join(self.dataset_path, 'wavelengths_arranged_complex.csv')

        # self.reference_spectra_path = os.path.join(self.base_path, 'HSI Data Sets',
        #                                'Plastics HSI Dataset',
        #                                # 'reference_spectra.csv')
        #                                # 'reference_spectra_modified_64pts.csv')
        #                                'sharper_reference_spectra_2Peaks_96pts.csv')


        # # Filter paths
        # self.filter_path = os.path.join(self.base_path, 'Machine Learning Codes',
        #                               'Filter CSV files',
        #                               # 'TransmissionTable_NIR_smooth.csv')
        #                               'Correlation Selected Filters', 
        #                               # '16 Filters', '16Filters_smoothened.csv')
        #                               '16 Filters', '16Filters.csv')
        #                               # '9 Filters', '9Filters_smoothened.csv')

        # Filter paths LOCAL PC WS 4
        self.filter_path = os.path.join(self.base_path, 'Filters',
                                         # '16Filters.csv')
                                         '16Filters_98_95_40%.csv')
                                         # '16Filters_98_95_40%_theta20.csv')
                                      


        # Model parameters
        self.batch_size = 16
        self.num_epochs = 5
        self.learning_rate = 1e-5
        # self.num_filters = 9
        self.num_filters = 16
        self.superpixel_height = 4
        self.superpixel_width = 4 # Change superpixel size here
        # self.superpixel_size = 4 # Change superpixel size here

        # Modified parameters for AVIRIS dataset
        self.image_height = 256  # For our cropped images
        self.image_width = 256
        # Modified parameters for 100-wavelength synthetic data
        self.num_wavelengths = 96  # Changed from 220 or 900
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

        # self.model_save_path = os.path.join(self.models_path, '061825_256p_96wl_16filters_2PEAKS_SharpSpectraTest5.pth')
        self.model_save_path = os.path.join(self.models_path, '070725_256p_96wl_AVIRIS_FILTER_98_95_40_test1.pth')
        # self.model_save_path = os.path.join(self.models_path, 'test.pth')

        self.results_path = 'results/070725'

config = Config()
print(f"Number of wavelength indices: {len(config.wavelength_indices)}")
print(f"Range of indices: {min(config.wavelength_indices)} to {max(config.wavelength_indices)}")
print(f"Actual wavelengths: {config.full_wavelengths[config.wavelength_indices[0]]} to {config.full_wavelengths[config.wavelength_indices[-1]]}")
