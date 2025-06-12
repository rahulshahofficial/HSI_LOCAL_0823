import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
import time
from thop import profile

class HyperspectralMetrics:
    """
    A class for computing various metrics for hyperspectral image reconstruction evaluation.
    
    This class provides methods to calculate:
    - Reconstruction precision metrics: PSNR, RMSE, MRAE, SSIM, spectral fidelity
    - Efficiency metrics: parameter count, FLOPs, running time per frame
    """
    
    @staticmethod
    def calculate_psnr(prediction, target, max_val=1.0):
        """
        Calculate Peak Signal-to-Noise Ratio.
        
        Args:
            prediction: Predicted hyperspectral image tensor
            target: Ground truth hyperspectral image tensor
            max_val: Maximum value of the signal (default: 1.0 for normalized data)
            
        Returns:
            Float: PSNR value in dB
        """
        # Ensure tensors are on CPU and convert to numpy if needed
        if torch.is_tensor(prediction):
            prediction = prediction.detach().cpu().numpy()
        if torch.is_tensor(target):
            target = target.detach().cpu().numpy()
        
        # Calculate MSE
        mse = np.mean((prediction - target) ** 2)
        if mse == 0:
            return float('inf')  # Perfect prediction
        
        # Calculate PSNR
        psnr = 20 * np.log10(max_val / np.sqrt(mse))
        return psnr
    
    @staticmethod
    def calculate_rmse(prediction, target):
        """
        Calculate Root Mean Square Error.
        
        Args:
            prediction: Predicted hyperspectral image tensor
            target: Ground truth hyperspectral image tensor
            
        Returns:
            Float: RMSE value
        """
        # Ensure tensors are on CPU and convert to numpy if needed
        if torch.is_tensor(prediction):
            prediction = prediction.detach().cpu().numpy()
        if torch.is_tensor(target):
            target = target.detach().cpu().numpy()
        
        # Calculate RMSE
        mse = np.mean((prediction - target) ** 2)
        rmse = np.sqrt(mse)
        return rmse
    
    @staticmethod
    def calculate_mrae(prediction, target, epsilon=1e-6):
        """
        Calculate Mean Relative Average Error.
        
        Args:
            prediction: Predicted hyperspectral image tensor
            target: Ground truth hyperspectral image tensor
            epsilon: Small constant to avoid division by zero
            
        Returns:
            Float: MRAE value
        """
        # Ensure tensors are on CPU and convert to numpy if needed
        if torch.is_tensor(prediction):
            prediction = prediction.detach().cpu().numpy()
        if torch.is_tensor(target):
            target = target.detach().cpu().numpy()
        
        # Calculate MRAE
        abs_diff = np.abs(prediction - target)
        rel_error = abs_diff / (np.abs(target) + epsilon)
        mrae = np.mean(rel_error)
        return mrae
    
    @staticmethod
    def calculate_ssim(prediction, target):
        """
        Calculate Structural Similarity Index Metric averaged across all spectral bands.
        
        Args:
            prediction: Predicted hyperspectral image tensor of shape [C, H, W]
            target: Ground truth hyperspectral image tensor of shape [C, H, W]
            
        Returns:
            Float: Average SSIM value across all bands
        """
        # Ensure tensors are on CPU and convert to numpy if needed
        if torch.is_tensor(prediction):
            prediction = prediction.detach().cpu().numpy()
        if torch.is_tensor(target):
            target = target.detach().cpu().numpy()
        
        # Calculate SSIM for each wavelength band
        num_bands = prediction.shape[0]
        ssim_values = []
        
        for i in range(num_bands):
            # Handle data range for each band
            data_range = max(
                np.max(prediction[i]) - np.min(prediction[i]),
                np.max(target[i]) - np.min(target[i])
            )
            if data_range == 0:
                data_range = 1  # Avoid division by zero
                
            # Calculate SSIM for this band
            ssim_val = ssim(
                prediction[i], 
                target[i],
                data_range=data_range
            )
            ssim_values.append(ssim_val)
        
        # Return average SSIM across all bands
        return np.mean(ssim_values)
    
    @staticmethod
    def calculate_spectral_fidelity(prediction, target, epsilon=1e-6):
        """
        Calculate spectral fidelity (similarity between spectral signatures).
        
        This metric measures how well the spectral signatures are preserved in the reconstruction
        by calculating the cosine similarity between predicted and target spectral vectors
        at each spatial location.
        
        Args:
            prediction: Predicted hyperspectral image tensor of shape [C, H, W]
            target: Ground truth hyperspectral image tensor of shape [C, H, W]
            epsilon: Small constant to avoid division by zero
            
        Returns:
            Float: Average spectral fidelity across all spatial locations
        """
        # Ensure tensors are on CPU and convert to numpy if needed
        if torch.is_tensor(prediction):
            prediction = prediction.detach().cpu().numpy()
        if torch.is_tensor(target):
            target = target.detach().cpu().numpy()
        
        # Get dimensions
        channels, height, width = prediction.shape
        
        # Reshape to have spectral dimension first for each pixel
        pred_reshaped = prediction.reshape(channels, -1)  # [C, H*W]
        target_reshaped = target.reshape(channels, -1)  # [C, H*W]
        
        # Calculate cosine similarity for each pixel location
        # Sum(pred_i * target_i) / (sqrt(Sum(pred_i^2)) * sqrt(Sum(target_i^2)))
        
        # Numerator: Sum of element-wise products
        dot_product = np.sum(pred_reshaped * target_reshaped, axis=0)  # [H*W]
        
        # Denominators: L2 norms (with epsilon to avoid division by zero)
        pred_norm = np.sqrt(np.sum(pred_reshaped**2, axis=0) + epsilon)  # [H*W]
        target_norm = np.sqrt(np.sum(target_reshaped**2, axis=0) + epsilon)  # [H*W]
        
        # Cosine similarity for each pixel
        cos_sim = dot_product / (pred_norm * target_norm)  # [H*W]
        
        # Average over all spatial locations
        spectral_fidelity = np.mean(cos_sim)
        
        return spectral_fidelity
    
    @staticmethod
    def compute_all_metrics(prediction, target):
        """
        Compute all reconstruction quality metrics at once.
        
        Args:
            prediction: Predicted hyperspectral image tensor
            target: Ground truth hyperspectral image tensor
            
        Returns:
            Dict: Dictionary containing all computed metrics
        """
        metrics = {
            'psnr': HyperspectralMetrics.calculate_psnr(prediction, target),
            'rmse': HyperspectralMetrics.calculate_rmse(prediction, target),
            'mrae': HyperspectralMetrics.calculate_mrae(prediction, target),
            'ssim': HyperspectralMetrics.calculate_ssim(prediction, target),
            'spectral_fidelity': HyperspectralMetrics.calculate_spectral_fidelity(prediction, target)
        }
        return metrics
    
    @staticmethod
    def analyze_model_efficiency(model, input_shape, filter_shape, device="cpu"):
        """
        Analyze model efficiency in terms of parameters, FLOPs, and inference time.
        
        Args:
            model: Neural network model
            input_shape: Shape of the input tensor (B, C, H, W)
            filter_shape: Shape of the filter pattern tensor (B, C, H, W)
            device: Device to run inference on ("cpu" or "cuda")
            
        Returns:
            Dict: Dictionary containing efficiency metrics
        """
        # Create dummy inputs with the specified shapes
        dummy_input = torch.randn(input_shape, device=device)
        dummy_filter = torch.randn(filter_shape, device=device)
        
        # Count parameters
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate FLOPs
        try:
            macs, _ = profile(model, inputs=(dummy_input, dummy_filter))
            flops = macs * 2  # Approximately 2 operations per MAC
        except:
            # Fallback if thop fails
            flops = -1
            print("Warning: Could not calculate FLOPs. Make sure 'thop' package is installed.")
        
        # Measure inference time
        model.eval()
        
        # Warm-up runs (to avoid measuring initialization overhead)
        for _ in range(5):
            with torch.no_grad():
                _ = model(dummy_input, dummy_filter)
        
        # Actual timing
        torch.cuda.synchronize() if device == "cuda" else None
        start_time = time.time()
        
        num_runs = 10
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(dummy_input, dummy_filter)
        
        torch.cuda.synchronize() if device == "cuda" else None
        end_time = time.time()
        
        # Average inference time per frame
        avg_time_per_frame = (end_time - start_time) / num_runs
        
        efficiency_metrics = {
            'num_params': num_params,
            'flops': flops,
            'time_per_frame': avg_time_per_frame
        }
        
        return efficiency_metrics
