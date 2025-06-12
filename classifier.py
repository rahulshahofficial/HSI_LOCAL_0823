import numpy as np
import torch
from reference_manager import ReferenceManager
from tqdm import tqdm

class SpectralClassifier:
    """
    Classifies pixels based on spectral similarity to reference spectra.
    Uses Spectral Angle Mapper (SAM) by default.
    """
    def __init__(self, reference_manager: ReferenceManager):
        self.reference_manager = reference_manager
        self.reference_spectra = self.reference_manager.get_all_spectra()
        self.material_names = self.reference_manager.get_material_names()
        
        # Pre-normalize reference spectra for efficiency
        self.normalized_references = {
            name: spec / (np.linalg.norm(spec) + 1e-8) 
            for name, spec in self.reference_spectra.items()
        }
        self.num_classes = len(self.material_names)
        
        print(f"Classifier initialized with {self.num_classes} reference materials.")

    def _calculate_sam(self, spectrum1, spectrum2_normalized):
        """Calculates the spectral angle between two spectra."""
        # Normalize spectrum1
        norm1 = np.linalg.norm(spectrum1) + 1e-8
        spectrum1_normalized = spectrum1 / norm1
        
        # Calculate dot product (cosine similarity)
        dot_product = np.dot(spectrum1_normalized, spectrum2_normalized)
        
        # Clip for numerical stability (essential!)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        
        # Calculate angle in radians
        angle = np.arccos(dot_product)
        return angle

    def classify_pixel(self, pixel_spectrum):
        """
        Classifies a single pixel spectrum.

        Args:
            pixel_spectrum (np.ndarray): 1D array of spectral intensities.

        Returns:
            tuple: (best_match_name, min_angle)
                   Returns (None, float('inf')) if no match found or error.
        """
        min_angle = float('inf')
        best_match_name = None

        if pixel_spectrum is None or pixel_spectrum.sum() < 1e-6: # Handle near-zero spectra
             return "Background", min_angle # Or a dedicated background class


        for name, ref_spec_normalized in self.normalized_references.items():
            angle = self._calculate_sam(pixel_spectrum, ref_spec_normalized)
            
            if angle < min_angle:
                min_angle = angle
                best_match_name = name
                
        return best_match_name, min_angle

    def classify_image(self, hsi_cube, sam_threshold=0.04):
        """
        Classifies an entire hyperspectral image cube.

        Args:
            hsi_cube (torch.Tensor or np.ndarray): Reconstructed cube (C, H, W).
            sam_threshold (float): Maximum spectral angle (radians) to consider a match. 
                                   Pixels with angles above this for all references are 
                                   classified as 'Unknown' or 'Background'.

        Returns:
            np.ndarray: Classification map (H, W) with integer labels.
                        0: Background/Unknown
                        1+: Index corresponding to material_names list + 1
            np.ndarray: SAM map (H, W) containing the minimum SAM angle for the best match.
        """
        if torch.is_tensor(hsi_cube):
            hsi_cube = hsi_cube.cpu().numpy()

        C, H, W = hsi_cube.shape
        classification_map = np.zeros((H, W), dtype=int)
        sam_map = np.full((H, W), float('inf'), dtype=float)

        # Reshape for faster processing: (H*W, C)
        hsi_flat = hsi_cube.reshape(C, -1).T 

        # Precompute reference matrix (NumRefs, C)
        reference_matrix_normalized = np.array([
            self.normalized_references[name] for name in self.material_names
        ]) # Shape: (NumClasses, C)

        # Normalize the input spectra batch: (H*W, C)
        norms_flat = np.linalg.norm(hsi_flat, axis=1, keepdims=True) + 1e-8
        hsi_flat_normalized = hsi_flat / norms_flat

        # Calculate all dot products (cosine similarities) at once: (H*W, NumClasses)
        # (H*W, C) @ (C, NumClasses) -> (H*W, NumClasses)
        dot_products = hsi_flat_normalized @ reference_matrix_normalized.T
        dot_products = np.clip(dot_products, -1.0, 1.0) # Clip for stability

        # Calculate all angles: (H*W, NumClasses)
        angles = np.arccos(dot_products)

        # Find the minimum angle and corresponding class index for each pixel
        min_angles_flat = np.min(angles, axis=1) # Shape: (H*W,)
        best_match_indices_flat = np.argmin(angles, axis=1) # Shape: (H*W,)

        # Reshape back to image dimensions
        sam_map = min_angles_flat.reshape(H, W)
        
        # Assign class labels based on threshold
        # Indices are 0 to NumClasses-1. We want 1 to NumClasses for materials.
        class_labels_flat = best_match_indices_flat + 1 
        
        # Apply threshold: Pixels where min_angle > threshold are set to 0 (Background/Unknown)
        mask = min_angles_flat > sam_threshold
        class_labels_flat[mask] = 0
        
        # Assign labels for near-zero spectra (handled earlier conceptually, apply here)
        zero_spectrum_mask = norms_flat.flatten() < 1e-7
        class_labels_flat[zero_spectrum_mask] = 0
        sam_map[zero_spectrum_mask.reshape(H,W)] = float('inf') # Indicate no meaningful angle

        classification_map = class_labels_flat.reshape(H, W)

        return classification_map, sam_map

    def get_class_name(self, class_id):
        """ Get material name from class ID (1-based index) """
        if class_id <= 0 or class_id > self.num_classes:
            return "Background/Unknown"
        return self.material_names[class_id - 1]
