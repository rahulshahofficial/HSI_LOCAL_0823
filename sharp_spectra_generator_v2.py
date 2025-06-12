import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# --- Configuration for Downsampling Output ---
# Set to None or a number >= original points to skip downsampling.
# Example: If original spectra have 181 points (800-1700nm every 5nm),
# setting TARGET_NUM_DATAPOINTS = 16 will downsample the output.
TARGET_NUM_DATAPOINTS = 64 # Or None to use original resolution

# --- Configuration for Synthetic Spectra with Dynamic Baseline (Method 1) ---
NUM_SYNTHETIC_SPECTRA = 2
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
    return amplitude * np.exp(-((wavelengths - center)**2) / (2 * (width**2)))

def generate_dynamic_baseline(wavelengths, num_components, amplitude_range, width_range_nm, offset_mean, offset_std):
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
        if feature_type == 'dip': modified_spectrum -= feature
        elif feature_type == 'peak': modified_spectrum += feature
        else: raise ValueError("feature_type must be 'dip' or 'peak'")
    return np.clip(modified_spectrum, 0, 1)

if __name__ == "__main__":
    try:
        df_input = pd.read_excel(INPUT_EXCEL_FILE)
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_EXCEL_FILE}' not found.")
        print("Creating a dummy 'plastic_spectra.xlsx' for demonstration...")
        # Dummy data uses a fixed reasonable resolution, downsampling applies to output
        dummy_wavelengths_arr_high_res = np.linspace(800, 1700, 181) # Approx 5nm steps
        dummy_plastic1_vals = 0.8 - 0.3 * generate_gaussian(dummy_wavelengths_arr_high_res, 950, 1, 40) - 0.2 * generate_gaussian(dummy_wavelengths_arr_high_res, 1300, 1, 60)
        dummy_plastic2_vals = 0.7 - 0.4 * generate_gaussian(dummy_wavelengths_arr_high_res, 1100, 1, 50) - 0.25 * generate_gaussian(dummy_wavelengths_arr_high_res, 1500, 1, 30)
        dummy_plastic1_vals = np.clip(dummy_plastic1_vals, 0.1, 0.9)
        dummy_plastic2_vals = np.clip(dummy_plastic2_vals, 0.1, 0.9)
        dummy_df = pd.DataFrame({
            'Wavelength': dummy_wavelengths_arr_high_res,
            'DummyPlastic1': dummy_plastic1_vals,
            'DummyPlastic2': dummy_plastic2_vals
        })
        dummy_df.to_excel(INPUT_EXCEL_FILE, index=False)
        df_input = pd.read_excel(INPUT_EXCEL_FILE)

    # --- Prepare Wavelengths ---
    # `original_full_res_wavelengths` is used for all generation steps
    original_full_res_wavelengths = df_input.iloc[:, 0].values
    num_original_points = len(original_full_res_wavelengths)

    # `output_wavelengths` is what will be used for plots and Excel files
    if TARGET_NUM_DATAPOINTS is not None and TARGET_NUM_DATAPOINTS < num_original_points:
        output_wavelengths = np.linspace(original_full_res_wavelengths.min(),
                                         original_full_res_wavelengths.max(),
                                         TARGET_NUM_DATAPOINTS)
        res_label = f"{TARGET_NUM_DATAPOINTS}pts"
        print(f"Output will be downsampled to {TARGET_NUM_DATAPOINTS} data points.")
    else:
        output_wavelengths = original_full_res_wavelengths
        res_label = f"{num_original_points}pts" # Original resolution
        print(f"Output will use original {num_original_points} data points.")

    # Update output filenames to include resolution info
    OUTPUT_SYNTHETIC_DB_EXCEL_FILE = OUTPUT_SYNTHETIC_DB_EXCEL_FILE.format(res_label)
    OUTPUT_MODIFIED_ALL_EXISTING_EXCEL_FILE = OUTPUT_MODIFIED_ALL_EXISTING_EXCEL_FILE.format(res_label)


    # Load reference spectra at full original resolution
    plastic_names_input = df_input.columns[1:].tolist()
    reference_spectra_full_res = {name: df_input[name].values for name in plastic_names_input}
    print(f"Loaded {len(reference_spectra_full_res)} reference spectra with {num_original_points} points each.")
    print(f"Wavelengths from {original_full_res_wavelengths.min()}nm to {original_full_res_wavelengths.max()}nm.")


    num_plots = (1 if NUM_SYNTHETIC_SPECTRA > 0 else 0) + (1 if reference_spectra_full_res else 0)
    if num_plots == 0:
        print("No spectra to generate or modify. Exiting.")
        exit()
        
    fig, axes = plt.subplots(num_plots, 1, figsize=(14, 7 * num_plots), squeeze=False)
    current_plot_idx = 0

    # # --- 1. Generate Synthetic Spectra with Dynamic Baselines (Optional) ---
    # if NUM_SYNTHETIC_SPECTRA > 0:
    #     ax1 = axes[current_plot_idx, 0]
    #     current_plot_idx +=1
    #     synthetic_db_data_for_df = {'Wavelength': output_wavelengths} # For DataFrame

    #     for i in range(NUM_SYNTHETIC_SPECTRA):
    #         spec_name = f"Synthetic_DB_{i+1}_SF{SYNTHETIC_SHARPNESS_FACTOR}"
            
    #         # Generate at full resolution
    #         dynamic_baseline_full_res = generate_dynamic_baseline(
    #             original_full_res_wavelengths, SYNTHETIC_BASELINE_NUM_COMPONENTS, SYNTHETIC_BASELINE_AMPLITUDE_RANGE,
    #             SYNTHETIC_BASELINE_WIDTH_RANGE_NM, SYNTHETIC_BASELINE_OFFSET_MEAN, SYNTHETIC_BASELINE_OFFSET_STD
    #         )
    #         synth_spec_full_res = create_synthetic_spectrum_with_dynamic_baseline(
    #             original_full_res_wavelengths, dynamic_baseline_full_res, SYNTHETIC_NUM_SHARP_FEATURES,
    #             SYNTHETIC_SHARP_FEATURE_DEPTH_RANGE, SYNTHETIC_BASE_SHARP_WIDTH_NM, SYNTHETIC_SHARPNESS_FACTOR
    #         )
            
    #         # Interpolate to output resolution
    #         final_synth_spec = np.interp(output_wavelengths, original_full_res_wavelengths, synth_spec_full_res)
            
    #         synthetic_db_data_for_df[spec_name] = final_synth_spec
    #         ax1.plot(output_wavelengths, final_synth_spec, label=spec_name, marker='o' if len(output_wavelengths) < 50 else None, markersize=4)

    #     ax1.set_title(f"Synthetic Spectra (Output Res: {len(output_wavelengths)}pts, SF: {SYNTHETIC_SHARPNESS_FACTOR})")
    #     ax1.set_xlabel("Wavelength (nm)")
    #     ax1.set_ylabel("Reflectance (Normalized)")
    #     ax1.legend(fontsize='small', loc='upper right')
    #     ax1.grid(True)
    #     ax1.set_ylim(-0.05, 1.1)

    #     df_synthetic_db = pd.DataFrame(synthetic_db_data_for_df)
    #     df_synthetic_db.to_excel(OUTPUT_SYNTHETIC_DB_EXCEL_FILE, index=False)
    #     print(f"\nSaved {NUM_SYNTHETIC_SPECTRA} synthetic spectra to '{OUTPUT_SYNTHETIC_DB_EXCEL_FILE}'")


    # --- 2. Modify ALL Existing Spectra by Adding Sharp Gaussian Features ---
    if reference_spectra_full_res:
        ax2 = axes[current_plot_idx, 0]
        current_plot_idx += 1
        modified_all_data_for_df = {'Wavelength': output_wavelengths} # For DataFrame
        
        print(f"\nModifying {len(reference_spectra_full_res)} existing spectra...")

        for name, original_spectrum_full_res in reference_spectra_full_res.items():
            num_features_to_add = random.randint(MODIFY_NUM_ADDED_FEATURES_RANGE[0], MODIFY_NUM_ADDED_FEATURES_RANGE[1])
            print(f"  Modifying '{name}', adding {num_features_to_add} features (orig res), then downsampling.")

            # Modify at full resolution
            modified_spectrum_full_res = add_sharp_features_to_spectrum(
                original_spectrum_full_res,
                original_full_res_wavelengths,
                num_features_to_add,
                MODIFY_ADDED_FEATURE_AMPLITUDE_RANGE,
                MODIFY_ADDED_FEATURE_BASE_WIDTH_NM,
                MODIFY_ADDED_FEATURE_SHARPNESS_FACTOR,
                MODIFY_ADDED_FEATURE_TYPE
            )
            
            # Interpolate both original and modified to output resolution
            final_original_spectrum_for_output = np.interp(output_wavelengths, original_full_res_wavelengths, original_spectrum_full_res)
            final_modified_spectrum_for_output = np.interp(output_wavelengths, original_full_res_wavelengths, modified_spectrum_full_res)
            
            clean_name = "".join(c if c.isalnum() else "_" for c in name)
            modified_col_name = f"Modified_{clean_name}_Add{num_features_to_add}Feat_SF{MODIFY_ADDED_FEATURE_SHARPNESS_FACTOR}"

            modified_all_data_for_df[name] = final_original_spectrum_for_output
            modified_all_data_for_df[modified_col_name] = final_modified_spectrum_for_output

            p_orig = ax2.plot(output_wavelengths, final_original_spectrum_for_output, linestyle='--', alpha=0.7, label=f"Original {name}", marker='.' if len(output_wavelengths) < 50 else None, markersize=4)
            ax2.plot(output_wavelengths, final_modified_spectrum_for_output, color=p_orig[0].get_color(), label=modified_col_name, marker='o' if len(output_wavelengths) < 50 else None, markersize=4)
        
        ax2.set_title(f"Modified Spectra (Output Res: {len(output_wavelengths)}pts, Added {MODIFY_ADDED_FEATURE_TYPE}s, SF: {MODIFY_ADDED_FEATURE_SHARPNESS_FACTOR})")
        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Reflectance (Normalized)")
        ax2.legend(fontsize='x-small', loc='upper right', ncol=max(1, len(reference_spectra_full_res) // 5))
        ax2.grid(True)
        ax2.set_ylim(-0.05, 1.1)

        df_modified_all = pd.DataFrame(modified_all_data_for_df)
        df_modified_all.to_excel(OUTPUT_MODIFIED_ALL_EXISTING_EXCEL_FILE, index=False)
        print(f"\nSaved all {len(reference_spectra_full_res)} original and modified spectra to '{OUTPUT_MODIFIED_ALL_EXISTING_EXCEL_FILE}'")
    else:
        if current_plot_idx == 0:
             print("\nNo reference spectra found to modify.")

    if num_plots > 0:
        plt.tight_layout()
        plt.suptitle(f"Spectral Data Generation (Output Resolution: {len(output_wavelengths)} points)", fontsize=16)
        fig.subplots_adjust(top=0.94) # Adjust top to make space for suptitle
        plt.show()
