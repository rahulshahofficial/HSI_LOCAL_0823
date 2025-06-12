import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
import numbers

from reference_manager import ReferenceManager
from spect_dict import SpectralDictionary

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class Spectral_Atten(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.num_heads = heads
        self.to_q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_k = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.to_v = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.k_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x_in):
        """
        x_in: [b,c,h,w]
        return out: [b,c,h,w]
        """
        b, c, h, w = x_in.shape
        q_in = self.q_dwconv(self.to_q(x_in))
        k_in = self.k_dwconv(self.to_k(x_in))
        v_in = self.v_dwconv(self.to_v(x_in))

        q = rearrange(q_in, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k_in, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v_in, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (q @ k.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.proj(out)

        return out


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class PreNorm(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net1 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.net2 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.relu = nn.GELU()
        self.out_conv = nn.Conv2d(dim * 2, dim, 1, 1, bias=False)

    def forward(self, x):
        out1 = self.net1(x)
        out2 = self.net2(x)
        out = torch.cat((out1, out2), dim=1)
        return self.out_conv(self.relu(out))


class SAM_Spectral(nn.Module):
    def __init__(self, dim, heads, num_blocks):
        super().__init__()

        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                LayerNorm(dim),
                Spectral_Atten(dim=dim, heads=heads),
                LayerNorm(dim),
                PreNorm(dim, mult=4)
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        for (norm1, attn, norm2, ffn) in self.blocks:
            x = attn(norm1(x)) + x
            x = ffn(norm2(x)) + x
        return x


class SpectralReconstructionNet(nn.Module):
    def __init__(self, input_channels=1, out_channels=100, dim=32, deep_stage=3,
                num_blocks=[1, 1, 1], num_heads=[1, 2, 4], reference_spectra=None, use_spectral_dict=True):
        super(SpectralReconstructionNet, self).__init__()
        """
        SRNet for hyperspectral reconstruction

        Args:
            input_channels: Number of input measurement channels
            out_channels: Number of spectral bands to reconstruct
            dim: Base feature dimension
            deep_stage: Number of encoder/decoder stages
            num_blocks: Number of SAM blocks at each stage
            num_heads: Number of attention heads at each stage
            reference_spectra: Pre-normalized tensor of reference spectra [NumRefs, C] or None.
            use_spectral_dict: Whether to use the spectral dictionary for regularization
        """
        self.dim = dim
        self.out_channels = out_channels
        self.stage = deep_stage
        self.reference_spectra = reference_spectra
        if self.reference_spectra is not None:
            print(f"SRNet initialized with reference spectra of shape: {self.reference_spectra.shape}")
        else:
            print("SRNet initialized *without* reference spectra for weighted loss.")

        self.use_spectral_dict = use_spectral_dict
        self.spectral_dict = None # Initialize
        if self.use_spectral_dict:
            try:
                from spect_dict import SpectralDictionary
                # Initialize dictionary - build it later or pass data now
                self.spectral_dict = SpectralDictionary(n_components=min(20, reference_spectra.shape[0] if reference_spectra is not None else 20))
                # # Option: Build dictionary here if reference_spectra are available
                if self.reference_spectra is not None:
                    print("Building spectral dictionary from provided reference spectra...")
                    # Note: build_from_data expects numpy array
                    self.spectral_dict.build_from_data(self.reference_spectra.cpu().numpy(), force_rebuild=True)
                else:
                    print("Building default spectral dictionary (no references provided).")
                    self.spectral_dict.build_default_dictionary() # Fallback
            except ImportError:
                print("Warning: Could not import SpectralDictionary. Spectral dictionary regularization will be disabled.")
                self.use_spectral_dict = False
            except Exception as e:
                 print(f"Warning: Failed to initialize/build spectral dictionary: {e}")
                 self.use_spectral_dict = False

        # Input embeddings - one for measurements, one for filter patterns
        self.embedding1 = nn.Conv2d(input_channels, dim, kernel_size=3, padding=1, bias=False)
        self.embedding2 = nn.Conv2d(out_channels, dim, kernel_size=3, padding=1, bias=False)
        self.embedding = nn.Conv2d(dim * 2, dim, kernel_size=3, padding=1, bias=False)

        # Spatial down/up sampling operations
        self.down_sample = nn.Conv2d(dim, dim, 4, 2, 1, bias=False)
        self.up_sample = nn.ConvTranspose2d(dim, dim, stride=2, kernel_size=2, padding=0, output_padding=0)

        # Final mapping to output spectral bands
        self.mapping = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=False)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(deep_stage):
            self.encoder_layers.append(nn.ModuleList([
                SAM_Spectral(dim=dim_stage, heads=num_heads[i], num_blocks=num_blocks[i]),
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2

        # Bottleneck
        self.bottleneck = SAM_Spectral(
            dim=dim_stage, heads=num_heads[-1], num_blocks=num_blocks[-1])

        # Decoder layers with adaptive dimension matching for skip connections
        self.decoder_layers = nn.ModuleList([])
        for i in range(deep_stage):
            # Transposed convolution to upsample feature maps
            decoder_up = nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, padding=0, output_padding=0)

            # Adaptive dimension matching for skip connection
            # This allows for flexibility in case encoder features have slightly different dimensions
            decoder_fusion = nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False)

            # SAM block for spectral processing
            decoder_sam = SAM_Spectral(
                dim=dim_stage // 2,
                heads=num_heads[deep_stage - 1 - i],
                num_blocks=num_blocks[deep_stage - 1 - i]
            )

            self.decoder_layers.append(nn.ModuleList([decoder_up, decoder_fusion, decoder_sam]))
            dim_stage //= 2

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, filter_pattern):
        """
        Forward pass with explicit filter pattern input
        Args:
            x: Input filtered measurements [B, C, H, W]
            filter_pattern: Filter pattern tensor [B, num_wavelengths, H, W]
                           representing the spectral transmission of filters
        """
        # Process input measurements and filter pattern
        x = self.embedding1(x)
        mask = self.embedding2(filter_pattern)

        # Combine both feature maps
        x = torch.cat((x, mask), dim=1)
        fea = self.embedding(x)

        # Save initial features for residual connection
        residual = fea
        fea = self.down_sample(fea)

        # Encoder forward pass - store intermediate features for skip connections
        fea_encoder = []
        for (Attention, FeaDownSample) in self.encoder_layers:
            fea = Attention(fea)
            fea_encoder.append(fea)  # Save for skip connection
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder forward pass with skip connections
        for i, (FeaUpSample, Fusion, Attention) in enumerate(self.decoder_layers):
            # Upsample feature maps
            fea = FeaUpSample(fea)

            # Ensure matching spatial dimensions for the skip connection
            skip_connection = fea_encoder[self.stage - 1 - i]

            # Adjust dimensions if necessary (resizing feature map)
            if fea.shape[-2:] != skip_connection.shape[-2:]:
                skip_connection = F.interpolate(
                    skip_connection,
                    size=fea.shape[-2:],
                    mode='bilinear',
                    align_corners=False
                )

            # Concatenate along channel dimension for the skip connection
            combined = torch.cat([fea, skip_connection], dim=1)

            # Apply 1x1 convolution to fuse features and reduce channels
            fea = Fusion(combined)

            # Apply spectral attention module
            fea = Attention(fea)

        # Final upsampling and residual connection
        fea = self.up_sample(fea)

        # Ensure matching dimensions for residual connection
        if fea.shape != residual.shape:
            fea = F.interpolate(fea, size=residual.shape[-2:], mode='bilinear', align_corners=False)

        out = fea + residual

        # Map to output spectral dimensions
        out = self.mapping(out)

        # Apply sigmoid to force outputs between 0 and 1
        out = torch.sigmoid(out) #(hard sigmoid)
        # out = torch.clamp(out, 0.0, 1.0) #(soft clamp)

        return out

    def compute_loss(self, outputs, targets, criterion):
        """
        Compute total loss including weighted reconstruction loss based on reference similarity.
        Enhanced with perceptual metrics (SSIM-based) to improve visual quality.

        Args:
            outputs: Predicted hyperspectral images
            targets: Ground truth hyperspectral images
            criterion: Reconstruction loss function (e.g., MSELoss)

        Returns:
            tuple: (total_loss, loss_components)
                - total_loss: The weighted sum of all loss components
                - loss_components: Dictionary containing individual loss values
        """

        B, C, H, W = outputs.shape
        device = outputs.device

        # --- Calculate Weights based on Ground Truth Similarity to References ---
        pixel_weights = torch.ones((B, H, W), device=device) # Default weight is 1

        if self.reference_spectra is not None:
            # Ensure references are on the correct device
            if self.reference_spectra.device != device:
                self.reference_spectra = self.reference_spectra.to(device)

            num_refs = self.reference_spectra.shape[0]

            # Reshape targets for batch processing: [B*H*W, C]
            targets_flat = targets.permute(0, 2, 3, 1).reshape(-1, C)

            # Normalize target spectra
            targets_norm = torch.linalg.norm(targets_flat, dim=1, keepdim=True) + 1e-8
            targets_flat_normalized = targets_flat / targets_norm

            # Calculate cosine similarity (dot product) between targets and references
            # targets_flat_normalized: [B*H*W, C]
            # self.reference_spectra.T: [C, NumRefs]
            # Result: [B*H*W, NumRefs]
            cos_sim = torch.matmul(targets_flat_normalized, self.reference_spectra.T)

            # Clip for numerical stability when calculating acos
            cos_sim = torch.clamp(cos_sim, -1.0 + 1e-7, 1.0 - 1e-7)

            # Calculate spectral angles (SAM) in radians
            angles = torch.acos(cos_sim) # Shape: [B*H*W, NumRefs]

            # Find the minimum angle (best match) for each pixel
            min_sam_gt, _ = torch.min(angles, dim=1) # Shape: [B*H*W]

            # --- Define the weighting function ---
            # High weight for low SAM (good match), low weight for high SAM (poor match)
            # Option: Exponential decay weight = base + scale * exp(-alpha * sam)
            alpha = 15.0 # Controls how quickly weight drops off. Higher alpha = faster drop.
            base_weight = 0.2 # Minimum weight for pixels dissimilar to references
            scale = 1.0 - base_weight # Scale of the exponential part (0.8 here)

            weights_flat = base_weight + scale * torch.exp(-alpha * min_sam_gt) # Shape: [B*H*W]

            # Reshape weights back to [B, H, W]
            pixel_weights = weights_flat.reshape(B, H, W)
        # --- End Weight Calculation ---

        # --- Weighted Reconstruction Loss ---
        # Calculate standard pixel-wise squared error
        pixel_wise_sq_error = (outputs - targets)**2 # Shape: [B, C, H, W]

        # Average squared error across channels for each pixel
        pixel_wise_mse = torch.mean(pixel_wise_sq_error, dim=1) # Shape: [B, H, W]

        # Apply the weights
        weighted_pixel_mse = pixel_weights * pixel_wise_mse # Shape: [B, H, W]

        # Calculate the final mean weighted reconstruction loss
        weighted_recon_loss = torch.mean(weighted_pixel_mse)
        # --- End Weighted Reconstruction Loss ---

        # --- Standard (Unweighted) Reconstruction Loss (for reporting) ---
        unweighted_recon_loss = criterion(outputs, targets)
        # --- End Standard ---

        # Spectral Smoothness Loss: Enhanced version with multi-order derivatives
        # First-order smoothness (already implemented)
        first_order_diff = outputs[:, 1:, :, :] - outputs[:, :-1, :, :]
        first_order_smoothness = torch.mean(first_order_diff ** 2)

        # Second-order smoothness (penalizes sudden changes in slope - very effective for reducing oscillations)
        if outputs.shape[1] > 2:
            second_order_diff = outputs[:, 2:, :, :] - 2 * outputs[:, 1:-1, :, :] + outputs[:, :-2, :, :]
            second_order_smoothness = torch.mean(second_order_diff ** 2)
        else:
            second_order_smoothness = 0.0

        # Combined spectral smoothness with higher weight on second-order
        spectral_smoothness_loss = first_order_smoothness + 2.0 * second_order_smoothness

        # Spatial Consistency Loss: Encourages smooth changes between adjacent pixels
        dx = outputs[:, :, 1:, :] - outputs[:, :, :-1, :]
        dy = outputs[:, :, :, 1:] - outputs[:, :, :, :-1]
        spatial_consistency_loss = torch.mean(dx ** 2) + torch.mean(dy ** 2)

        # Spectral Angle Mapper loss: Measures similarity between spectral signatures
        # Reshape to (batch_size, num_wavelengths, height*width)
        b, c, h, w = outputs.shape
        outputs_flat = outputs.reshape(b, c, -1)  # [B, C, H*W]
        targets_flat = targets.reshape(b, c, -1)  # [B, C, H*W]

        # Normalize along spectral dimension
        outputs_norm = torch.nn.functional.normalize(outputs_flat, dim=1, p=2)
        targets_norm = torch.nn.functional.normalize(targets_flat, dim=1, p=2)

        # Compute dot product between normalized vectors (cosine similarity)
        cos_sim = torch.sum(outputs_norm * targets_norm, dim=1)  # [B, H*W]

        # Convert to angle and average (lower is better)
        spectral_angle_loss = torch.mean(torch.acos(torch.clamp(cos_sim, -0.9999, 0.9999)))

        # SSIM-based loss component (1-SSIM, since we minimize loss)
        # We implement a simplified version that works with batched tensors
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        # Calculate local means and variances using average pooling
        # For simplicity, we compute this across all spectral bands simultaneously
        mu_x = torch.nn.functional.avg_pool2d(outputs, kernel_size=3, stride=1, padding=1)
        mu_y = torch.nn.functional.avg_pool2d(targets, kernel_size=3, stride=1, padding=1)

        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x_sq = torch.nn.functional.avg_pool2d(outputs ** 2, kernel_size=3, stride=1, padding=1) - mu_x_sq
        sigma_y_sq = torch.nn.functional.avg_pool2d(targets ** 2, kernel_size=3, stride=1, padding=1) - mu_y_sq
        sigma_xy = torch.nn.functional.avg_pool2d(outputs * targets, kernel_size=3, stride=1, padding=1) - mu_xy

        # Calculate SSIM
        ssim_numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
        ssim_denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
        ssim = ssim_numerator / ssim_denominator

        # Convert to loss (1-SSIM)
        ssim_loss = 1.0 - torch.mean(ssim)

        # Spectral Total Variation Loss: More robust smoothness regularization
        # It penalizes total absolute differences while preserving important edges/transitions
        spectral_tv = torch.abs(outputs[:, 1:, :, :] - outputs[:, :-1, :, :])
        # Use a soft approximation to L1 norm (Huber-like loss) to be differentiable
        epsilon = 1e-3  # Small constant to avoid numerical issues
        spectral_tv_loss = torch.mean(torch.sqrt(spectral_tv ** 2 + epsilon))

        # Spectral Dictionary Prior Loss (Optional, can complement the reference loss)
        spectral_dict_loss = torch.tensor(0.0, device=device)
        if self.use_spectral_dict and self.spectral_dict is not None:
            # Ensure dictionary is built and on the correct device
            try:
                # Lazy building/loading if not already done
                if self.spectral_dict.components is None:
                     if self.reference_spectra is not None:
                         print("Building spectral dictionary from references inside compute_loss...")
                         self.spectral_dict.build_from_data(self.reference_spectra.cpu().numpy(), force_rebuild=True)
                     else:
                         print("Building default spectral dictionary inside compute_loss...")
                         self.spectral_dict.build_default_dictionary()

                # Ensure device consistency
                if self.spectral_dict.device != str(device):
                    self.spectral_dict.device = str(device)
                    if self.spectral_dict.components is not None:
                        self.spectral_dict.components = self.spectral_dict.components.to(device)
                    if self.spectral_dict.mean_spectrum is not None:
                        self.spectral_dict.mean_spectrum = self.spectral_dict.mean_spectrum.to(device)

                # Calculate loss (using reshaped outputs)
                outputs_for_dict = outputs.permute(0, 2, 3, 1).reshape(-1, C)
                spectral_dict_loss = self.spectral_dict.spectral_prior_loss(outputs_for_dict)

            except Exception as e:
                print(f"Warning: Failed to compute spectral dictionary loss: {str(e)}")
                spectral_dict_loss = torch.tensor(0.0, device=device)

        # Store all loss components in a dictionary
        loss_components = {
            'mse_loss': unweighted_recon_loss.item(), # Report standard MSE
            'weighted_mse_loss': weighted_recon_loss.item(), # Report the weighted version
            'spectral_smoothness_loss': spectral_smoothness_loss.item(),
            'spatial_consistency_loss': spatial_consistency_loss.item(),
            'spectral_angle_loss': spectral_angle_loss.item(),
            'spectral_tv_loss': spectral_tv_loss.item(),
            'spectral_dict_loss': spectral_dict_loss.item() if isinstance(spectral_dict_loss, torch.Tensor) else spectral_dict_loss,
            'ssim_loss': ssim_loss.item()
        }

        # --- Weighted Sum of Losses ---
        # Use weighted_recon_loss instead of the original recon_loss
        # Adjust coefficients as needed based on experimentation
        # total_loss = (
        #     0.5 * unweighted_recon_loss +
        #     2.0 * weighted_recon_loss +             # <--- Key change: Use weighted loss
        #     0.3 * spatial_consistency_loss +
        #     0.3 * spectral_smoothness_loss +
        #     0.5 * spectral_tv_loss +
        #     0.8 * spectral_angle_loss +
        #     0.5 * spectral_dict_loss +
        #     0.5 * ssim_loss
        # )


        # OG LOSS function used in AVIRIS Dataset
        total_loss = (
            2.0 * unweighted_recon_loss +
            0.3 * spatial_consistency_loss +
            0.3 * spectral_smoothness_loss +  # Increased from 0.1 to 0.3
            0.8 * spectral_tv_loss +         # Added TV loss for robust smoothness
            0.8 * spectral_angle_loss +
            0.0 * spectral_dict_loss +        # No dict learning
            0.5 * ssim_loss
        )
        # Increased weight on spectral smoothness to encourage smoother reconstructions

        return total_loss, loss_components


