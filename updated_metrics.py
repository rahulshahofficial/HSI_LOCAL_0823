import torch
import torch.nn.functional as F
# import numpy as np
# from skimage.metrics import structural_similarity as ssim
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
    def calculate_rmse_torch(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Root Mean Square Error using PyTorch tensors.
        Args:
            prediction: Predicted hyperspectral image tensor (GPU)
            target: Ground truth hyperspectral image tensor (GPU)
        Returns:
            Scalar tensor: RMSE value
        """
        if prediction.shape != target.shape:
            raise ValueError(f"Input shapes must match. Got {prediction.shape} and {target.shape}")
        mse = torch.mean((prediction - target) ** 2)
        rmse = torch.sqrt(mse)
        return rmse

    @staticmethod
    def calculate_psnr_torch(prediction: torch.Tensor, target: torch.Tensor, max_val: float = 1.0, epsilon: float = 1e-12) -> torch.Tensor:
        """
        Calculate Peak Signal-to-Noise Ratio using PyTorch tensors.
        Args:
            prediction: Predicted tensor (GPU)
            target: Ground truth tensor (GPU)
            max_val: Maximum possible pixel value (default: 1.0)
            epsilon: Small value to avoid log10(0)
        Returns:
            Scalar tensor: PSNR value in dB
        """
        if prediction.shape != target.shape:
            raise ValueError(f"Input shapes must match. Got {prediction.shape} and {target.shape}")

        mse = torch.mean((prediction - target) ** 2)
        if mse == 0:
            # PSNR is infinite for identical images
            # Return a very large number or positive infinity
            return torch.tensor(float('inf'), device=prediction.device)

        psnr = 10.0 * torch.log10(max_val**2 / (mse + epsilon))
        return psnr

    @staticmethod
    def calculate_mrae_torch(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        """
        Calculate Mean Relative Average Error using PyTorch tensors.
        Args:
            prediction: Predicted tensor (GPU)
            target: Ground truth tensor (GPU)
            epsilon: Small constant to avoid division by zero
        Returns:
            Scalar tensor: MRAE value
        """
        if prediction.shape != target.shape:
            raise ValueError(f"Input shapes must match. Got {prediction.shape} and {target.shape}")

        abs_diff = torch.abs(prediction - target)
        # Ensure denominator is safe
        denominator = torch.abs(target) + epsilon
        # Avoid potential issues if target is exactly zero in some places where pred is not
        denominator = torch.max(denominator, torch.tensor(epsilon, device=target.device))

        rel_error = abs_diff / denominator
        mrae = torch.mean(rel_error)
        return mrae

    @staticmethod
    def calculate_spectral_fidelity_torch(prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
        """
        Calculate spectral fidelity (average cosine similarity) using PyTorch tensors.
        Args:
            prediction: Predicted tensor (GPU) - Expected shape [B, C, H, W] or [C, H, W]
            target: Ground truth tensor (GPU) - Expected shape [B, C, H, W] or [C, H, W]
            epsilon: Small constant to avoid division by zero
        Returns:
            Scalar tensor: Average spectral fidelity
        """
        if prediction.shape != target.shape:
            raise ValueError(f"Input shapes must match. Got {prediction.shape} and {target.shape}")

        # Ensure shape is [B, C, H*W] or [C, H*W]
        if prediction.ndim == 4: # Batched input B, C, H, W
            B, C, H, W = prediction.shape
            pred_flat = prediction.reshape(B, C, -1)
            target_flat = target.reshape(B, C, -1)
            dim_to_normalize = 1 # Channel dimension
            dim_to_sum = 1      # Channel dimension
        elif prediction.ndim == 3: # Single image C, H, W
             C, H, W = prediction.shape
             pred_flat = prediction.reshape(C, -1)
             target_flat = target.reshape(C, -1)
             dim_to_normalize = 0 # Channel dimension
             dim_to_sum = 0      # Channel dimension
        else:
            raise ValueError(f"Unsupported input dimensions: {prediction.ndim}. Expected 3 or 4.")

        # Normalize along the spectral dimension (C) for each pixel vector
        pred_norm = torch.linalg.norm(pred_flat, dim=dim_to_normalize, keepdim=True)
        target_norm = torch.linalg.norm(target_flat, dim=dim_to_normalize, keepdim=True)

        pred_normalized = pred_flat / (pred_norm + epsilon)
        target_normalized = target_flat / (target_norm + epsilon)

        # Calculate dot product (cosine similarity) along the spectral dimension
        # Result shape: [B, H*W] or [H*W]
        cos_sim = torch.sum(pred_normalized * target_normalized, dim=dim_to_sum)

        # Average over all pixels (and batch if present)
        spectral_fidelity = torch.mean(cos_sim)

        return spectral_fidelity

    @staticmethod
    def calculate_ssim_torch(prediction: torch.Tensor, target: torch.Tensor, window_size: int = 3, data_range: float = 1.0, size_average: bool = True, C1: float = None, C2: float = None) -> torch.Tensor:
        """
        Calculate Structural Similarity Index Metric (SSIM) using PyTorch.
        Adapted from loss implementation. Assumes input is [B, C, H, W].
        Args:
            prediction: Predicted tensor (GPU) [B, C, H, W]
            target: Ground truth tensor (GPU) [B, C, H, W]
            window_size: Size of the Gaussian window (must be odd). Default: 3x3 patch means window_size=3.
            data_range: The dynamic range of the images (Maximum value - Minimum value). Default: 1.0.
            size_average: If True, compute the mean SSIM over the batch. Default: True.
            C1: Constant for luminance stabilization. Default: (0.01 * data_range)^2.
            C2: Constant for contrast stabilization. Default: (0.03 * data_range)^2.
        Returns:
            Scalar tensor: Mean SSIM value
        """
        if prediction.shape != target.shape:
            raise ValueError(f"Input shapes must match. Got {prediction.shape} and {target.shape}")
        if prediction.ndim != 4:
             raise ValueError(f"Input tensors must be 4D [B, C, H, W]. Got {prediction.ndim}D")

        if C1 is None:
            C1 = (0.01 * data_range) ** 2
        if C2 is None:
            C2 = (0.03 * data_range) ** 2

        B, C, H, W = prediction.shape
        padd = window_size // 2 # Padding to keep dimensions constant

        # Calculate means using average pooling
        mu_x = F.avg_pool2d(prediction, kernel_size=window_size, stride=1, padding=padd)
        mu_y = F.avg_pool2d(target, kernel_size=window_size, stride=1, padding=padd)

        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_xy = mu_x * mu_y

        # Calculate variances and covariances
        sigma_x_sq = F.avg_pool2d(prediction * prediction, kernel_size=window_size, stride=1, padding=padd) - mu_x_sq
        sigma_y_sq = F.avg_pool2d(target * target, kernel_size=window_size, stride=1, padding=padd) - mu_y_sq
        sigma_xy = F.avg_pool2d(prediction * target, kernel_size=window_size, stride=1, padding=padd) - mu_xy

        # SSIM formula
        ssim_numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        ssim_denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
        ssim_map = ssim_numerator / ssim_denominator

        if size_average:
            return ssim_map.mean() # Average over batch, channels, spatial dims
        else:
            return ssim_map.mean(dim=[1, 2, 3]) # Average over channels, spatial dims per batch item


    @staticmethod
    def compute_all_metrics_torch(prediction_gpu: torch.Tensor, target_gpu: torch.Tensor, max_val: float = 1.0):
        """
        Computes all metrics using PyTorch on the GPU.
        Args:
            prediction_gpu: Predicted tensor on GPU (B, C, H, W) or (C, H, W)
            target_gpu: Target tensor on GPU (B, C, H, W) or (C, H, W)
            max_val: Maximum possible pixel value (for PSNR).
        Returns:
            Dict: Dictionary containing computed metrics as Python floats.
        """
        if prediction_gpu.device != target_gpu.device:
            raise ValueError("Prediction and target tensors must be on the same device.")
        if prediction_gpu.shape != target_gpu.shape:
             raise ValueError(f"Prediction shape {prediction_gpu.shape} != Target shape {target_gpu.shape}")

        # Add batch dimension if input is single image (C, H, W)
        pred_in = prediction_gpu.unsqueeze(0) if prediction_gpu.ndim == 3 else prediction_gpu
        targ_in = target_gpu.unsqueeze(0) if target_gpu.ndim == 3 else target_gpu

        # --- Calculate metrics ---
        rmse = HyperspectralMetrics.calculate_rmse_torch(pred_in, targ_in)
        psnr = HyperspectralMetrics.calculate_psnr_torch(pred_in, targ_in, max_val=max_val)
        mrae = HyperspectralMetrics.calculate_mrae_torch(pred_in, targ_in)
        # SSIM expects B, C, H, W
        ssim = HyperspectralMetrics.calculate_ssim_torch(pred_in, targ_in, data_range=max_val)
        # Spectral fidelity can handle B, C, H, W directly
        sf = HyperspectralMetrics.calculate_spectral_fidelity_torch(pred_in, targ_in)

        metrics = {
            'psnr': psnr.item(),
            'rmse': rmse.item(),
            'mrae': mrae.item(),
            'ssim': ssim.item(),
            'spectral_fidelity': sf.item()
        }
        return metrics

    # @staticmethod
    # def calculate_psnr(prediction, target, max_val=1.0):
    #     """
    #     Calculate Peak Signal-to-Noise Ratio.
        
    #     Args:
    #         prediction: Predicted hyperspectral image tensor
    #         target: Ground truth hyperspectral image tensor
    #         max_val: Maximum value of the signal (default: 1.0 for normalized data)
            
    #     Returns:
    #         Float: PSNR value in dB
    #     """
    #     # Ensure tensors are on CPU and convert to numpy if needed
    #     if torch.is_tensor(prediction):
    #         prediction = prediction.detach().cpu().numpy()
    #     if torch.is_tensor(target):
    #         target = target.detach().cpu().numpy()
        
    #     # Calculate MSE
    #     mse = np.mean((prediction - target) ** 2)
    #     if mse == 0:
    #         return float('inf')  # Perfect prediction
        
    #     # Calculate PSNR
    #     psnr = 20 * np.log10(max_val / np.sqrt(mse))
    #     return psnr
    
    # @staticmethod
    # def calculate_rmse(prediction, target):
    #     """
    #     Calculate Root Mean Square Error.
        
    #     Args:
    #         prediction: Predicted hyperspectral image tensor
    #         target: Ground truth hyperspectral image tensor
            
    #     Returns:
    #         Float: RMSE value
    #     """
    #     # Ensure tensors are on CPU and convert to numpy if needed
    #     if torch.is_tensor(prediction):
    #         prediction = prediction.detach().cpu().numpy()
    #     if torch.is_tensor(target):
    #         target = target.detach().cpu().numpy()
        
    #     # Calculate RMSE
    #     mse = np.mean((prediction - target) ** 2)
    #     rmse = np.sqrt(mse)
    #     return rmse
    
    # @staticmethod
    # def calculate_mrae(prediction, target, epsilon=1e-6):
    #     """
    #     Calculate Mean Relative Average Error.
        
    #     Args:
    #         prediction: Predicted hyperspectral image tensor
    #         target: Ground truth hyperspectral image tensor
    #         epsilon: Small constant to avoid division by zero
            
    #     Returns:
    #         Float: MRAE value
    #     """
    #     # Ensure tensors are on CPU and convert to numpy if needed
    #     if torch.is_tensor(prediction):
    #         prediction = prediction.detach().cpu().numpy()
    #     if torch.is_tensor(target):
    #         target = target.detach().cpu().numpy()
        
    #     # Calculate MRAE
    #     abs_diff = np.abs(prediction - target)
    #     rel_error = abs_diff / (np.abs(target) + epsilon)
    #     mrae = np.mean(rel_error)
    #     return mrae
    
    # @staticmethod
    # def calculate_ssim(prediction, target):
    #     """
    #     Calculate Structural Similarity Index Metric averaged across all spectral bands.
        
    #     Args:
    #         prediction: Predicted hyperspectral image tensor of shape [C, H, W]
    #         target: Ground truth hyperspectral image tensor of shape [C, H, W]
            
    #     Returns:
    #         Float: Average SSIM value across all bands
    #     """
    #     # Ensure tensors are on CPU and convert to numpy if needed
    #     if torch.is_tensor(prediction):
    #         prediction = prediction.detach().cpu().numpy()
    #     if torch.is_tensor(target):
    #         target = target.detach().cpu().numpy()
        
    #     # Calculate SSIM for each wavelength band
    #     num_bands = prediction.shape[0]
    #     ssim_values = []
        
    #     for i in range(num_bands):
    #         # Handle data range for each band
    #         data_range = max(
    #             np.max(prediction[i]) - np.min(prediction[i]),
    #             np.max(target[i]) - np.min(target[i])
    #         )
    #         if data_range == 0:
    #             data_range = 1  # Avoid division by zero
                
    #         # Calculate SSIM for this band
    #         ssim_val = ssim(
    #             prediction[i],
    #             target[i],
    #             data_range=data_range
    #         )
    #         ssim_values.append(ssim_val)
        
    #     # Return average SSIM across all bands
    #     return np.mean(ssim_values)
    
    # @staticmethod
    # def calculate_spectral_fidelity(prediction, target, epsilon=1e-6):
    #     """
    #     Calculate spectral fidelity (similarity between spectral signatures).
        
    #     This metric measures how well the spectral signatures are preserved in the reconstruction
    #     by calculating the cosine similarity between predicted and target spectral vectors
    #     at each spatial location.
        
    #     Args:
    #         prediction: Predicted hyperspectral image tensor of shape [C, H, W]
    #         target: Ground truth hyperspectral image tensor of shape [C, H, W]
    #         epsilon: Small constant to avoid division by zero
            
    #     Returns:
    #         Float: Average spectral fidelity across all spatial locations
    #     """
    #     # Ensure tensors are on CPU and convert to numpy if needed
    #     if torch.is_tensor(prediction):
    #         prediction = prediction.detach().cpu().numpy()
    #     if torch.is_tensor(target):
    #         target = target.detach().cpu().numpy()
        
    #     # Get dimensions
    #     channels, height, width = prediction.shape
        
    #     # Reshape to have spectral dimension first for each pixel
    #     pred_reshaped = prediction.reshape(channels, -1)  # [C, H*W]
    #     target_reshaped = target.reshape(channels, -1)  # [C, H*W]
        
    #     # Calculate cosine similarity for each pixel location
    #     # Sum(pred_i * target_i) / (sqrt(Sum(pred_i^2)) * sqrt(Sum(target_i^2)))
        
    #     # Numerator: Sum of element-wise products
    #     dot_product = np.sum(pred_reshaped * target_reshaped, axis=0)  # [H*W]
        
    #     # Denominators: L2 norms (with epsilon to avoid division by zero)
    #     pred_norm = np.sqrt(np.sum(pred_reshaped**2, axis=0) + epsilon)  # [H*W]
    #     target_norm = np.sqrt(np.sum(target_reshaped**2, axis=0) + epsilon)  # [H*W]
        
    #     # Cosine similarity for each pixel
    #     cos_sim = dot_product / (pred_norm * target_norm)  # [H*W]
        
    #     # Average over all spatial locations
    #     spectral_fidelity = np.mean(cos_sim)
        
    #     return spectral_fidelity
    
    # @staticmethod
    # def compute_all_metrics(prediction, target):
    #     """
    #     Compute all reconstruction quality metrics at once.
        
    #     Args:
    #         prediction: Predicted hyperspectral image tensor
    #         target: Ground truth hyperspectral image tensor
            
    #     Returns:
    #         Dict: Dictionary containing all computed metrics
    #     """
    #     metrics = {
    #         'psnr': HyperspectralMetrics.calculate_psnr(prediction, target),
    #         'rmse': HyperspectralMetrics.calculate_rmse(prediction, target),
    #         'mrae': HyperspectralMetrics.calculate_mrae(prediction, target),
    #         'ssim': HyperspectralMetrics.calculate_ssim(prediction, target),
    #         'spectral_fidelity': HyperspectralMetrics.calculate_spectral_fidelity(prediction, target)
    #     }
    #     return metrics
    
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
