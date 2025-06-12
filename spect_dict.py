import torch
import numpy as np
import os
import pickle
from config import config
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d

class SpectralDictionary:
    """
    A class to manage a dictionary of typical spectral signatures for use in
    regularizing the reconstruction of hyperspectral data.
    
    This dictionary can be constructed from real spectral data and used to
    guide the reconstruction process towards more realistic and smoother 
    spectral signatures.
    """
    
    def __init__(self, n_components=20, device='cpu'):
        """
        Initialize the spectral dictionary.

        Args:
            n_components: Number of principal components to use for the dictionary
            device: Device to store the dictionary on ('cpu' or 'cuda')
        """
        self.n_components = n_components
        self.device = device
        self.pca = None
        self.mean_spectrum = None
        self.components = None

        # Use subset wavelengths corresponding to 800-1700nm range
        self.wavelengths = config.full_wavelengths[config.wavelength_indices]  # Now 100 points


        # Print initialization info for debugging
        print(f"Initializing spectral dictionary for {len(self.wavelengths)} wavelengths ({self.wavelengths[0]:.1f}-{self.wavelengths[-1]:.1f}nm)")

        # Make sure save directory exists
        results_path = config.results_path
        os.makedirs(results_path, exist_ok=True)
        self.dictionary_path = os.path.join(results_path, 'spectral_dictionary.pkl')
        
    def build_from_data(self, spectral_data, force_rebuild=False):
        """
        Build the spectral dictionary from a dataset of spectral signatures.

        Args:
            spectral_data: Tensor or array of shape [N, C] where N is number of spectral signatures
                          and C is number of spectral bands
            force_rebuild: Whether to force rebuilding the dictionary even if it exists

        Returns:
            self: The dictionary object
        """
        # Check if dictionary already exists
        if os.path.exists(self.dictionary_path) and not force_rebuild:
            try:
                self.load_dictionary()
                print(f"Loaded existing spectral dictionary with {self.n_components} components")

                # Add a check for correct dimensions
                if hasattr(self, 'components') and self.components is not None:
                    expected_dim = len(config.wavelength_indices)
                    actual_dim = self.components.shape[1]
                    if actual_dim != expected_dim:
                        print(f"Warning: Loaded dictionary has {actual_dim} wavelengths but expected {expected_dim}")
                        print("Forcing dictionary rebuild with correct dimensions")
                        # Continue to rebuild instead of returning
                    else:
                        return self
            except Exception as e:
                print(f"Failed to load existing dictionary: {str(e)}, rebuilding...")

        print(f"Building spectral dictionary with {self.n_components} components from {spectral_data.shape[0]} samples")

        # Verify input dimensions
        if spectral_data.shape[1] != len(config.wavelength_indices):
            print(f"Warning: Input spectral data has {spectral_data.shape[1]} wavelengths but expected {len(config.wavelength_indices)}")
            print("This may cause dimension mismatches later")

        # Convert to numpy if tensor
        if torch.is_tensor(spectral_data):
            spectral_data = spectral_data.cpu().numpy()

        # Compute mean spectrum
        self.mean_spectrum = np.mean(spectral_data, axis=0)

        # Center the data
        centered_data = spectral_data - self.mean_spectrum

        # Fit PCA
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(centered_data)

        # Store components and convert to tensor
        self.components = torch.tensor(
            self.pca.components_,
            dtype=torch.float32,
            device=self.device
        )
        self.mean_spectrum = torch.tensor(
            self.mean_spectrum,
            dtype=torch.float32,
            device=self.device
        )

        # Save dictionary
        self.save_dictionary()

        print(f"Dictionary built - explained variance: {self.pca.explained_variance_ratio_.sum()*100:.2f}%")
        print(f"Dictionary shape: Components {self.components.shape}, Mean {self.mean_spectrum.shape}")
        return self
    
    def save_dictionary(self):
        """Save the dictionary to disk."""
        os.makedirs(os.path.dirname(self.dictionary_path), exist_ok=True)
        
        # Move tensors to CPU for saving
        components_cpu = self.components.cpu().numpy() if self.components is not None else None
        mean_spectrum_cpu = self.mean_spectrum.cpu().numpy() if self.mean_spectrum is not None else None
        
        with open(self.dictionary_path, 'wb') as f:
            pickle.dump({
                'components': components_cpu,
                'mean_spectrum': mean_spectrum_cpu,
                'pca': self.pca,
                'n_components': self.n_components,
                'wavelengths': self.wavelengths
            }, f)
    
    def load_dictionary(self):
        """Load the dictionary from disk."""
        with open(self.dictionary_path, 'rb') as f:
            data = pickle.load(f)
            
        self.pca = data['pca']
        self.n_components = data['n_components']
        self.wavelengths = data['wavelengths']
        
        # Move components to the right device
        self.components = torch.tensor(
            data['components'], 
            dtype=torch.float32, 
            device=self.device
        )
        self.mean_spectrum = torch.tensor(
            data['mean_spectrum'], 
            dtype=torch.float32, 
            device=self.device
        )
    
    def project(self, spectra):
        """
        Project the spectra onto the dictionary and reconstruct.
        This gives a smoothed version of the input spectra.
        
        Args:
            spectra: Tensor of shape [..., C] where C is the number of spectral bands
            
        Returns:
            smooth_spectra: Smoothed version of the input spectra
        """
        if not torch.is_tensor(spectra):
            spectra = torch.tensor(spectra, dtype=torch.float32, device=self.device)
        
        # Get original shape and reshape to 2D
        original_shape = spectra.shape
        spectra_2d = spectra.reshape(-1, original_shape[-1])
        
        # Center the data
        centered = spectra_2d - self.mean_spectrum
        
        # Project onto components
        coefficients = torch.matmul(centered, self.components.T)
        
        # Reconstruct
        reconstructed = torch.matmul(coefficients, self.components) + self.mean_spectrum
        
        # Reshape back to original shape
        return reconstructed.reshape(original_shape)
    
    def spectral_prior_loss(self, predicted_spectra):
        """
        Calculate a loss based on how well the predicted spectra
        can be represented by the dictionary.

        Args:
            predicted_spectra: Tensor of shape [..., C] where C is number of spectral bands

        Returns:
            loss: A scalar loss value
        """
        try:
            # Make sure dictionary is initialized
            if self.components is None:
                self.build_default_dictionary(force_rebuild=True)

            # Check if the dictionary dimensions match the predicted spectra
            pred_dim = predicted_spectra.shape[-1]
            dict_dim = self.components.shape[-1]

            if pred_dim != dict_dim:
                print(f"Warning: Dictionary dimension mismatch. Predicted: {pred_dim}, Dictionary: {dict_dim}")
                print("Rebuilding dictionary with correct dimensions...")

                # We need to rebuild the dictionary with the correct dimensions
                # This is a temporary approach to handle the mismatch
                # For production, you should ensure dictionary is built with correct dimensions initially

                # Create a simple dataset with correct dimensions for rebuilding
                # This is a fallback approach - ideally you would use real spectral data
                sample_data = torch.rand(1000, pred_dim).cpu().numpy()
                self.build_from_data(sample_data, force_rebuild=True)

                # If rebuilding fails, return zero loss
                if self.components.shape[-1] != pred_dim:
                    print("Failed to rebuild dictionary with correct dimensions.")
                    return torch.tensor(0.0, device=predicted_spectra.device)

            # Get original shape and reshape to 2D if needed
            original_shape = predicted_spectra.shape
            if len(original_shape) > 2:
                spectra_2d = predicted_spectra.reshape(-1, original_shape[-1])
            else:
                spectra_2d = predicted_spectra

            # Ensure tensors are on the same device
            if self.components.device != spectra_2d.device:
                self.components = self.components.to(spectra_2d.device)
                self.mean_spectrum = self.mean_spectrum.to(spectra_2d.device)

            # Project onto dictionary (get smoothed version)
            smoothed_spectra = self.project(spectra_2d)

            # Calculate reconstruction error
            error = torch.mean((spectra_2d - smoothed_spectra) ** 2)

            return error

        except Exception as e:
            print(f"Warning: Failed to compute spectral dictionary loss: {str(e)}")
            # Return a zero tensor as fallback
            return torch.tensor(0.0, device=predicted_spectra.device)
        
    def build_default_dictionary(self, force_rebuild=False):
        """
        Build a default dictionary using some common spectral signatures
        when no external data is available.
        
        Args:
            force_rebuild: Whether to force rebuilding the dictionary
            
        Returns:
            self: The dictionary object
        """
        if os.path.exists(self.dictionary_path) and not force_rebuild:
            try:
                self.load_dictionary()
                print("Loaded existing default dictionary")
                return self
            except:
                print("Failed to load existing dictionary, creating default...")
                
        print("Creating default spectral dictionary...")
        
        # Generate some synthetic spectral signatures based on common patterns
        num_samples = 1000
        num_bands = len(self.wavelengths)
        synthetic_data = np.zeros((num_samples, num_bands))
        
        # Generate random smooth spectra using different methods
        x = np.linspace(0, 1, num_bands)
        
        for i in range(num_samples):
            if i % 4 == 0:
                # Gaussian peaks
                num_peaks = np.random.randint(1, 4)
                spectrum = np.zeros(num_bands)
                for _ in range(num_peaks):
                    center = np.random.rand()
                    width = 0.05 + 0.1 * np.random.rand()
                    height = 0.5 + np.random.rand()
                    spectrum += height * np.exp(-((x - center) ** 2) / (2 * width ** 2))
            elif i % 4 == 1:
                # Sigmoid-based transitions
                num_transitions = np.random.randint(1, 3)
                spectrum = np.zeros(num_bands)
                base = np.random.rand() * 0.3
                for _ in range(num_transitions):
                    center = 0.2 + 0.6 * np.random.rand()
                    steepness = 10 + 20 * np.random.rand()
                    direction = 1 if np.random.rand() > 0.5 else -1
                    height = 0.2 + 0.6 * np.random.rand()
                    spectrum += height / (1 + np.exp(-steepness * direction * (x - center)))
                spectrum += base
            elif i % 4 == 2:
                # Polynomial-based smooth curves
                degree = np.random.randint(2, 5)
                coeffs = np.random.randn(degree + 1)
                spectrum = np.polyval(coeffs, x)
                # Normalize
                spectrum = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min() + 1e-8)
            else:
                # Combination of sine waves
                num_waves = np.random.randint(2, 5)
                spectrum = np.zeros(num_bands)
                for _ in range(num_waves):
                    freq = 1 + 5 * np.random.rand()
                    phase = 2 * np.pi * np.random.rand()
                    amp = 0.2 + 0.8 * np.random.rand()
                    spectrum += amp * np.sin(freq * x * np.pi + phase)
                # Normalize
                spectrum = (spectrum - spectrum.min()) / (spectrum.max() - spectrum.min() + 1e-8)
            
            # Add a small amount of noise
            noise = 0.02 * np.random.randn(num_bands)
            spectrum += noise
            
            # Ensure values are positive
            spectrum = np.maximum(0, spectrum)
            
            # Normalize
            if spectrum.max() > 0:
                spectrum /= spectrum.max()
                
            synthetic_data[i] = spectrum
        
        return self.build_from_data(synthetic_data, force_rebuild=True)
