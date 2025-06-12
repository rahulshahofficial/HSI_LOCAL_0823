import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from config import config # Assuming config holds the correct wavelength info

class ReferenceManager:
    """
    Manages loading and accessing reference spectral data.
    Ensures reference spectra are aligned with the model's output wavelengths.
    """
    def __init__(self, reference_file_path):
        self.reference_file_path = reference_file_path
        self.target_wavelengths = config.full_wavelengths[config.wavelength_indices]
        self.num_wavelengths = len(self.target_wavelengths)
        self.reference_spectra = {} # Dict: material_name -> spectrum (numpy array)
        self.material_names = []

        self._load_and_interpolate()

    def _load_and_interpolate(self):
        """Loads reference spectra and interpolates them to match target wavelengths."""
        try:
            df = pd.read_csv(self.reference_file_path)
        except FileNotFoundError:
            print(f"Error: Reference file not found at {self.reference_file_path}")
            raise
        except Exception as e:
            print(f"Error loading reference file {self.reference_file_path}: {e}")
            raise

        # Assume first column is 'Wavelength'
        if df.columns[0].lower() != 'wavelength':
            print(f"Warning: First column expected to be 'Wavelength', found '{df.columns[0]}'")
        
        source_wavelengths = df.iloc[:, 0].values
        self.material_names = df.columns[1:].tolist()

        print(f"Loading reference spectra for materials: {self.material_names}")
        print(f"Source wavelengths range: {source_wavelengths.min():.1f} - {source_wavelengths.max():.1f} nm")
        print(f"Target wavelengths range: {self.target_wavelengths.min():.1f} - {self.target_wavelengths.max():.1f} nm ({self.num_wavelengths} points)")

        for material_name in self.material_names:
            source_spectrum = df[material_name].values

            # Create interpolation function
            interp_func = interp1d(source_wavelengths, source_spectrum, 
                                   kind='linear', # Or 'cubic' for smoother
                                   bounds_error=False, 
                                   fill_value="extrapolate") # Handle out-of-range

            # Interpolate to target wavelengths
            interpolated_spectrum = interp_func(self.target_wavelengths)
            
            # Normalize the reference spectrum (optional, but often helpful)
            # interpolated_spectrum = interpolated_spectrum / np.max(interpolated_spectrum) 
            interpolated_spectrum = np.clip(interpolated_spectrum, 0, None) # Ensure non-negative

            self.reference_spectra[material_name] = interpolated_spectrum
            
        print("Reference spectra loaded and interpolated successfully.")

    def get_spectrum(self, material_name):
        """Returns the interpolated spectrum for a given material name."""
        return self.reference_spectra.get(material_name, None)

    def get_all_spectra(self):
        """Returns a dictionary of all reference spectra."""
        return self.reference_spectra
        
    def get_material_names(self):
        """Returns a list of material names."""
        return self.material_names

# Example Usage (add to main or test script)
# reference_manager = ReferenceManager('path/to/your/reference_file.csv')
# pet_spectrum = reference_manager.get_spectrum('PET') 
# all_spectra = reference_manager.get_all_spectra()

# --- END OF FILE reference_manager.py ---
