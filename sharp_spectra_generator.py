import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# --- Configuration for Synthetic Spectra with Dynamic Baseline (Method 1) ---
# This part remains the same if you still want to generate purely synthetic ones
NUM_SYNTHETIC_SPECTRA = 2 # Set to 0 if not needed
SYNTHETIC_NUM_SHARP_FEATURES = 7
SYNTHETIC_SHARP_FEATURE_DEPTH_RANGE = (0.1, 0.5)
SYNTHETIC_BASE_SHARP_WIDTH_NM = 15
SYNTHETIC_SHARPNESS_FACTOR = 5.0

SYNTHETIC_BASELINE_NUM_COMPONENTS = 3
SYNTHETIC_BASELINE_AMPLITUDE_RANGE = (-0.4, 0.4)
SYNTHETIC_BASELINE_WIDTH_RANGE_NM = (150, 400)
SYNTHETIC_BASELINE_OFFSET_MEAN = 0.7
SYNTHETIC_BASELINE_OFFSET_STD = 0.1

# --- Configuration for Modifying ALL Existing Spectra by Adding Gaussian Features ---
# MODIFY_NUM_ADDED_FEATURES_RANGE: (min_features, max_features) to add to each spectrum.
# A random number of features within this range will be chosen for each spectrum.
MODIFY_NUM_ADDED_FEATURES_RANGE = (2, 3) # e.g., add between 3 and 7 features
MODIFY_ADDED_FEATURE_AMPLITUDE_RANGE = (0.05, 0.5) # Amplitude (depth/height)
MODIFY_ADDED_FEATURE_BASE_WIDTH_NM = 20
MODIFY_ADDED_FEATURE_SHARPNESS_FACTOR = 4 # Higher = sharper added features
MODIFY_ADDED_FEATURE_TYPE = 'dip' # 'dip' or 'peak'

# --- File Paths ---
INPUT_EXCEL_FILE = '/Volumes/ValentineLab/SimulationData/Rahul/Hyperspectral Imaging Project/HSI Data Sets/Plastics HSI Dataset/reference_spectra.xlsx'
OUTPUT_SYNTHETIC_DB_EXCEL_FILE = '/Volumes/ValentineLab/SimulationData/Rahul/Hyperspectral Imaging Project/HSI Data Sets/Plastics HSI Dataset/sharp_spectra.xlsx'
OUTPUT_MODIFIED_ALL_EXISTING_EXCEL_FILE = '/Volumes/ValentineLab/SimulationData/Rahul/Hyperspectral Imaging Project/HSI Data Sets/Plastics HSI Dataset/sharper_reference_spectra.xlsx'

def generate_gaussian(wavelengths, center, amplitude, width):
    """Generates a Gaussian shape."""
    return amplitude * np.exp(-((wavelengths - center)**2) / (2 * (width**2)))

def generate_dynamic_baseline(wavelengths, num_components, amplitude_range, width_range_nm, offset_mean, offset_std):
    """Generates a smoothly varying dynamic baseline using a sum of broad Gaussians."""
    min_wl, max_wl = wavelengths.min(), wavelengths.max()
    baseline = np.ones_like(wavelengths) * (np.random.normal(offset_mean, offset_std))
    for _ in range(num_components):
        center_wl = random.uniform(min_wl, max_wl)
        amplitude = random.uniform(amplitude_range[0], amplitude_range[1])
        width = random.uniform(width_range_nm[0], width_range_nm[1])
        baseline += generate_gaussian(wavelengths, center_wl, amplitude, width)
    baseline = np.clip(baseline, 0.2, 1.0)
    return baseline

def create_synthetic_spectrum_with_dynamic_baseline(
    wavelengths, dynamic_baseline, num_sharp_features,
    sharp_feature_depth_range, base_sharp_width_nm, sharpness_factor):
    min_wl, max_wl = wavelengths.min(), wavelengths.max()
    spectrum = dynamic_baseline.copy()
    actual_sharp_width = base_sharp_width_nm / sharpness_factor
    for _ in range(num_sharp_features):
        center_wl = random.uniform(min_wl + actual_sharp_width * 3, max_wl - actual_sharp_width * 3)
        idx_center = np.argmin(np.abs(wavelengths - center_wl))
        if dynamic_baseline[idx_center] < sharp_feature_depth_range[0] + 0.1: continue
        depth = random.uniform(sharp_feature_depth_range[0], sharp_feature_depth_range[1])
        depth = min(depth, dynamic_baseline[idx_center] - 0.05)
        dip = generate_gaussian(wavelengths, center_wl, depth, actual_sharp_width)
        spectrum -= dip
    return np.clip(spectrum, 0, 1)

def add_sharp_features_to_spectrum(
    original_spectrum, wavelengths, num_features_to_add,
    feature_amplitude_range, base_width_nm, sharpness_factor, feature_type='dip'):
    min_wl, max_wl = wavelengths.min(), wavelengths.max()
    modified_spectrum = original_spectrum.copy()
    actual_feature_width = base_width_nm / sharpness_factor

    for _ in range(num_features_to_add):
        center_wl = random.uniform(min_wl + actual_feature_width * 3, max_wl - actual_feature_width * 3)
        amplitude = random.uniform(feature_amplitude_range[0], feature_amplitude_range[1])
        feature = generate_gaussian(wavelengths, center_wl, amplitude, actual_feature_width)
        if feature_type == 'dip':
            modified_spectrum -= feature
        elif feature_type == 'peak':
            modified_spectrum += feature
        else:
            raise ValueError("feature_type must be 'dip' or 'peak'")
    return np.clip(modified_spectrum, 0, 1)

if __name__ == "__main__":
    try:
        df_input = pd.read_excel(INPUT_EXCEL_FILE)
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_EXCEL_FILE}' not found.")
        print("Creating a dummy 'plastic_spectra.xlsx' for demonstration...")
        dummy_wavelengths_arr = np.linspace(800, 1700, 181)
        dummy_plastic1_vals = 0.8 - 0.3 * generate_gaussian(dummy_wavelengths_arr, 950, 1, 40) - 0.2 * generate_gaussian(dummy_wavelengths_arr, 1300, 1, 60)
        dummy_plastic2_vals = 0.7 - 0.4 * generate_gaussian(dummy_wavelengths_arr, 1100, 1, 50) - 0.25 * generate_gaussian(dummy_wavelengths_arr, 1500, 1, 30)
        dummy_plastic1_vals = np.clip(dummy_plastic1_vals, 0.1, 0.9)
        dummy_plastic2_vals = np.clip(dummy_plastic2_vals, 0.1, 0.9)
        dummy_df = pd.DataFrame({
            'Wavelength': dummy_wavelengths_arr,
            'DummyPlastic1': dummy_plastic1_vals,
            'DummyPlastic2': dummy_plastic2_vals
        })
        dummy_df.to_excel(INPUT_EXCEL_FILE, index=False)
        df_input = pd.read_excel(INPUT_EXCEL_FILE)

    wavelengths = df_input.iloc[:, 0].values
    plastic_names_input = df_input.columns[1:].tolist()
    reference_spectra = {name: df_input[name].values for name in plastic_names_input}
    print(f"Loaded {len(reference_spectra)} reference spectra. Wavelengths from {wavelengths.min()}nm to {wavelengths.max()}nm.")

    num_plots = (1 if NUM_SYNTHETIC_SPECTRA > 0 else 0) + (1 if reference_spectra else 0)
    if num_plots == 0:
        print("No spectra to generate or modify. Exiting.")
        exit()

    fig, axes = plt.subplots(num_plots, 1, figsize=(14, 7 * num_plots), squeeze=False)
    current_plot_idx = 0

    # # --- 1. Generate Synthetic Spectra with Dynamic Baselines (Optional) ---
    # if NUM_SYNTHETIC_SPECTRA > 0:
    #     ax1 = axes[current_plot_idx, 0]
    #     current_plot_idx +=1
    #     synthetic_db_data = {'Wavelength': wavelengths}
    #     for i in range(NUM_SYNTHETIC_SPECTRA):
    #         spec_name = f"Synthetic_DB_{i+1}_SF{SYNTHETIC_SHARPNESS_FACTOR}"
    #         dynamic_baseline = generate_dynamic_baseline(
    #             wavelengths, SYNTHETIC_BASELINE_NUM_COMPONENTS, SYNTHETIC_BASELINE_AMPLITUDE_RANGE,
    #             SYNTHETIC_BASELINE_WIDTH_RANGE_NM, SYNTHETIC_BASELINE_OFFSET_MEAN, SYNTHETIC_BASELINE_OFFSET_STD
    #         )
    #         synth_spec = create_synthetic_spectrum_with_dynamic_baseline(
    #             wavelengths, dynamic_baseline, SYNTHETIC_NUM_SHARP_FEATURES,
    #             SYNTHETIC_SHARP_FEATURE_DEPTH_RANGE, SYNTHETIC_BASE_SHARP_WIDTH_NM, SYNTHETIC_SHARPNESS_FACTOR
    #         )
    #         synthetic_db_data[spec_name] = synth_spec
    #         ax1.plot(wavelengths, synth_spec, label=spec_name)

    #     ax1.set_title(f"Synthetic Spectra with Dynamic Baselines (SF: {SYNTHETIC_SHARPNESS_FACTOR})")
    #     ax1.set_xlabel("Wavelength (nm)")
    #     ax1.set_ylabel("Reflectance (Normalized)")
    #     ax1.legend(fontsize='small', loc='upper right')
    #     ax1.grid(True)
    #     ax1.set_ylim(-0.05, 1.1)

    #     df_synthetic_db = pd.DataFrame(synthetic_db_data)
    #     df_synthetic_db.to_excel(OUTPUT_SYNTHETIC_DB_EXCEL_FILE, index=False)
    #     print(f"\nSaved {NUM_SYNTHETIC_SPECTRA} synthetic spectra with dynamic baselines to '{OUTPUT_SYNTHETIC_DB_EXCEL_FILE}'")


    # --- 2. Modify ALL Existing Spectra by Adding Sharp Gaussian Features ---
    if reference_spectra:
        ax2 = axes[current_plot_idx, 0]
        current_plot_idx += 1
        modified_all_data = {'Wavelength': wavelengths}

        print(f"\nModifying {len(reference_spectra)} existing spectra...")

        for name, original_spectrum in reference_spectra.items():
            # Determine number of features to add for this specific spectrum
            num_features_to_add = random.randint(
                MODIFY_NUM_ADDED_FEATURES_RANGE[0],
                MODIFY_NUM_ADDED_FEATURES_RANGE[1]
            )

            print(f"  Modifying '{name}', adding {num_features_to_add} features.")

            modified_spectrum = add_sharp_features_to_spectrum(
                original_spectrum,
                wavelengths,
                num_features_to_add, # Use the per-spectrum random number
                MODIFY_ADDED_FEATURE_AMPLITUDE_RANGE,
                MODIFY_ADDED_FEATURE_BASE_WIDTH_NM,
                MODIFY_ADDED_FEATURE_SHARPNESS_FACTOR,
                MODIFY_ADDED_FEATURE_TYPE
            )

            # Sanitize original name for use in modified name if it contains problematic characters
            clean_name = "".join(c if c.isalnum() else "_" for c in name)
            modified_col_name = f"Modified_{clean_name}_Add{num_features_to_add}Feat_SF{MODIFY_ADDED_FEATURE_SHARPNESS_FACTOR}"

            # Store original and modified for easy comparison in Excel
            modified_all_data[name] = original_spectrum
            modified_all_data[modified_col_name] = modified_spectrum

            # Plot original and modified
            # To avoid overly crowded plots if many spectra, consider plotting only a few or one at a time
            # For now, let's plot all, but this might need adjustment for > 5-10 spectra
            p_orig = ax2.plot(wavelengths, original_spectrum, linestyle='--', alpha=0.7, label=f"Original {name}")
            ax2.plot(wavelengths, modified_spectrum, color=p_orig[0].get_color(), label=modified_col_name)

        ax2.set_title(f"All Existing Spectra Modified by Adding {MODIFY_ADDED_FEATURE_TYPE}s (SF: {MODIFY_ADDED_FEATURE_SHARPNESS_FACTOR})")
        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Reflectance (Normalized)")
        ax2.legend(fontsize='x-small', loc='upper right', ncol=2) # Adjust legend for many items
        ax2.grid(True)
        ax2.set_ylim(-0.05, 1.1)

        df_modified_all = pd.DataFrame(modified_all_data)
        df_modified_all.to_excel(OUTPUT_MODIFIED_ALL_EXISTING_EXCEL_FILE, index=False)
        print(f"\nSaved all {len(reference_spectra)} original and modified spectra to '{OUTPUT_MODIFIED_ALL_EXISTING_EXCEL_FILE}'")
    else:
        if current_plot_idx == 0 : # Only if no synthetic spectra were generated either
             print("\nNo reference spectra found to modify.")

    if num_plots > 0:
        plt.tight_layout()
        plt.show()
