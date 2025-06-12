import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import platform

# --- Enhanced CuPy Availability Check (Keep as is) ---
CUPY_AVAILABLE = False
CUPY_INIT_ERROR_MSG = ""
try:
    import cupy as cp
    _test_arr = cp.array([1, 2, 3]) 
    _test_arr_sum = _test_arr.sum() 
    CUPY_AVAILABLE = True
    print("CuPy successfully imported and initialized. GPU support is available.")
except ImportError:
    CUPY_INIT_ERROR_MSG = "CuPy library not found."
except Exception as e: 
    CUPY_INIT_ERROR_MSG = (
        f"CuPy imported but failed to initialize properly. GPU support will be disabled.\n"
        f"Error details: {type(e).__name__}: {e}\n"
    )

# --- Keep previous functions: load_data, calculate_sam_matrix, plot_selected_filters, save_selected_to_excel ---
# --- Make sure these are the correct versions from our previous iterations ---
def load_data(excel_path):
    """Loads filter transmission data from an Excel file. (Keep previous version)"""
    try:
        df_read = pd.read_excel(excel_path, header=0, index_col=0)
    except FileNotFoundError:
        raise
    except Exception as e:
        raise ValueError(f"Error reading Excel file '{excel_path}'. Details: {e}")

    if df_read.empty:
        raise ValueError(f"Excel file '{excel_path}' is empty or could not be parsed correctly.")

    filter_names = df_read.index.tolist()
    if not filter_names: 
        raise ValueError("No filter names found in the first column.")

    wavelength_headers = df_read.columns.tolist()
    if not wavelength_headers:
        raise ValueError("No wavelength column headers found.")

    parsed_wavelengths_list = []
    try:
        parsed_wavelengths_list = pd.to_numeric(wavelength_headers, errors='raise').tolist()
    except ValueError:
        cleaned_headers_for_wl = [str(h).lower().replace('nm', '').strip() for h in wavelength_headers]
        try:
            parsed_wavelengths_list = pd.to_numeric(cleaned_headers_for_wl, errors='raise').tolist()
        except ValueError:
            num_wl_points_from_cols = len(wavelength_headers)
            parsed_wavelengths_list = np.linspace(800, 1700, num_wl_points_from_cols).tolist()
            print(f"Warning: Could not parse wavelengths. Generated {num_wl_points_from_cols} linear points.")
    
    wavelengths = np.array(parsed_wavelengths_list)
    if wavelengths.size == 0:
         raise ValueError("Resulting wavelength array is empty.")

    processed_df_filters_as_rows = df_read.copy()
    any_nan_introduced = False
    
    for wl_col_header_idx, wl_col_header in enumerate(processed_df_filters_as_rows.columns):
        original_col_data = df_read.iloc[:, wl_col_header_idx].copy()
        converted_col_data = pd.to_numeric(processed_df_filters_as_rows[wl_col_header], errors='coerce')
        processed_df_filters_as_rows[wl_col_header] = converted_col_data
        newly_nan_mask = converted_col_data.isnull() & ~original_col_data.isnull()
        if newly_nan_mask.any():
            any_nan_introduced = True
            for filter_name_with_nan in newly_nan_mask[newly_nan_mask].index:
                original_problem_value = original_col_data[filter_name_with_nan]
                print(f"  Warning: Non-numeric '{str(original_problem_value)[:30]}' for '{filter_name_with_nan}' at '{wl_col_header}'. NaN.")
    if any_nan_introduced:
        print("NOTE: Non-numeric values converted to NaN. NaNs are ignored for min/SAM.")

    spectra_data_for_sam = processed_df_filters_as_rows.T.values.astype(np.float64)
    all_nan_filters_mask = processed_df_filters_as_rows.isnull().all(axis=1)
    if all_nan_filters_mask.any():
        print(f"Warning: Filters all NaNs: {processed_df_filters_as_rows.index[all_nan_filters_mask].tolist()}")

    if processed_df_filters_as_rows.shape[0] == 0: raise ValueError("No filter rows.")
    if processed_df_filters_as_rows.shape[1] == 0: raise ValueError("No wavelength columns.")
        
    return processed_df_filters_as_rows, filter_names, spectra_data_for_sam, wavelengths

def calculate_sam_matrix(spectra_matrix_with_nans_transposed, use_gpu=False):
    """Calculates SAM matrix. (Keep previous version)"""
    xp = np 
    input_matrix_gpu_ready = spectra_matrix_with_nans_transposed 
    if use_gpu and CUPY_AVAILABLE:
        xp = cp
        try:
            input_matrix_gpu_ready = xp.asarray(input_matrix_gpu_ready)
        except Exception as e:
            print(f"Warning: CuPy asarray failed: {e}. Fallback NumPy.")
            xp = np
            input_matrix_gpu_ready = spectra_matrix_with_nans_transposed
            use_gpu = False
    
    input_matrix_cleaned = xp.nan_to_num(input_matrix_gpu_ready, nan=0.0, posinf=0.0, neginf=0.0)
    num_wavelengths, num_filters = input_matrix_cleaned.shape
    if num_filters == 0: return np.array([])
    if num_wavelengths == 0: return np.full((num_filters, num_filters), np.pi / 2) if num_filters > 0 else np.array([])

    norm_filters = xp.linalg.norm(input_matrix_cleaned, axis=0, keepdims=True)
    is_zero_norm_filter = (norm_filters == 0)
    norm_filters_for_division = xp.where(is_zero_norm_filter, 1.0, norm_filters)
    spectra_normalized = input_matrix_cleaned / norm_filters_for_division
    cosine_similarity_matrix = spectra_normalized.T @ spectra_normalized
    
    both_filters_zero_norm_mask = is_zero_norm_filter.T & is_zero_norm_filter
    cosine_similarity_matrix = xp.where(both_filters_zero_norm_mask, 1.0, cosine_similarity_matrix)
    cosine_similarity_matrix = xp.clip(cosine_similarity_matrix, -1.0, 1.0)
    sam_matrix_xp = xp.arccos(cosine_similarity_matrix)
    
    return cp.asnumpy(sam_matrix_xp) if use_gpu and CUPY_AVAILABLE and xp is cp else sam_matrix_xp


def select_least_correlated_filters(
    sam_matrix, 
    filter_names, 
    n_to_select_user_requested,
    all_avg_transmissions_np, # NumPy array of AVERAGE transmissions for ALL filters
    minimum_required_avg_transmission # New parameter for the minimum threshold
    ):
    """
    Selects filters based on SAM and a MINIMUM AVERAGE transmission constraint.
    """
    num_total_filters = sam_matrix.shape[0]

    if n_to_select_user_requested <= 0:
        print("Warning: n_to_select is 0 or negative. No filters will be selected.")
        return []
    if num_total_filters == 0:
        print("Warning: No filters available in SAM matrix to select from.")
        return []

    # 1. Identify eligible filters based on MINIMUM AVERAGE transmission
    eligible_filter_indices = []
    for i in range(num_total_filters):
        avg_t = all_avg_transmissions_np[i]
        # A filter is eligible if its average transmission is not NaN AND is >= required minimum average
        if not np.isnan(avg_t) and avg_t >= minimum_required_avg_transmission:
            eligible_filter_indices.append(i)

    if not eligible_filter_indices:
        print(f"Warning: No filters meet the minimum average transmission criterion (Avg T >= {minimum_required_avg_transmission:.3f}). Cannot select any filters.")
        return []

    print(f"Found {len(eligible_filter_indices)} filters meeting the minimum average transmission criterion.")

    n_to_select_actual = min(n_to_select_user_requested, len(eligible_filter_indices))
    if n_to_select_actual < n_to_select_user_requested:
        print(f"Warning: Requested {n_to_select_user_requested} filters, but only {len(eligible_filter_indices)} meet the criteria. "
              f"Will select {n_to_select_actual} filters.")
    if n_to_select_actual == 0 :
        return []

    selected_original_indices = []

    # 2. Initial Seed from eligible filters
    if n_to_select_actual > 0:
        if len(eligible_filter_indices) == 1:
            initial_filter_original_idx = eligible_filter_indices[0]
        else:
            best_initial_original_idx = -1
            max_avg_sams_within_eligible = -np.inf
            for i_orig_idx in eligible_filter_indices:
                current_sum_sams = 0
                num_valid_sams_for_sum = 0
                for j_orig_idx in eligible_filter_indices:
                    if i_orig_idx == j_orig_idx: continue
                    sam_val = sam_matrix[i_orig_idx, j_orig_idx]
                    if not np.isnan(sam_val):
                        current_sum_sams += sam_val
                        num_valid_sams_for_sum +=1
                if num_valid_sams_for_sum > 0 :
                    avg_sam_to_eligible_peers = current_sum_sams / num_valid_sams_for_sum
                    if avg_sam_to_eligible_peers > max_avg_sams_within_eligible :
                        max_avg_sams_within_eligible = avg_sam_to_eligible_peers
                        best_initial_original_idx = i_orig_idx
                elif best_initial_original_idx == -1:
                    best_initial_original_idx = i_orig_idx
            if best_initial_original_idx != -1:
                initial_filter_original_idx = best_initial_original_idx
            else:
                initial_filter_original_idx = eligible_filter_indices[0] # Fallback
        selected_original_indices.append(initial_filter_original_idx)

    # 3. Greedily add remaining filters
    while len(selected_original_indices) < n_to_select_actual:
        best_next_candidate_original_idx = -1
        max_of_min_sams_to_selected_set = -1.0
        for candidate_original_idx in eligible_filter_indices:
            if candidate_original_idx in selected_original_indices: continue
            current_min_sam_to_selected = float('inf')
            valid_sam_found_for_this_candidate = False
            for sel_original_idx in selected_original_indices:
                sam_value = sam_matrix[candidate_original_idx, sel_original_idx]
                if not np.isnan(sam_value):
                    current_min_sam_to_selected = min(current_min_sam_to_selected, sam_value)
                    valid_sam_found_for_this_candidate = True
            if not valid_sam_found_for_this_candidate : current_min_sam_to_selected = -np.inf
            if current_min_sam_to_selected > max_of_min_sams_to_selected_set:
                max_of_min_sams_to_selected_set = current_min_sam_to_selected
                best_next_candidate_original_idx = candidate_original_idx
        if best_next_candidate_original_idx != -1:
            selected_original_indices.append(best_next_candidate_original_idx)
        else:
            break
    return [filter_names[i] for i in selected_original_indices]

def plot_selected_filters(
    df_filters_as_rows,
    selected_filter_names,
    all_original_filter_names,
    wavelengths,
    output_plot_path
    ):
    """Plots selected filters, using their original index and AVERAGE transmission in the legend."""
    if not selected_filter_names: return
    plt.figure(figsize=(12, 7))
    for name in selected_filter_names:
        if name in df_filters_as_rows.index:
            spectrum_data = df_filters_as_rows.loc[name].values
            try:
                original_idx_0_based = all_original_filter_names.index(name)
                legend_label_idx = original_idx_0_based + 1
            except ValueError:
                legend_label_idx = "UnkIdx"

            avg_trans = np.nanmean(spectrum_data)
            label = f"Filter Index {legend_label_idx} (Avg T: {avg_trans:.3f})"
            plt.plot(wavelengths, spectrum_data, label=label)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Transmission")
    plt.title(f"Spectra of {len(selected_filter_names)} Selected Filters")
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    try:
        plt.savefig(output_plot_path)
        print(f"Plot saved: {output_plot_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close()

def save_selected_to_excel(
    df_processed_filters_as_rows, # This DataFrame has filter names as index, wavelengths as columns
    selected_filter_names,
    wavelengths_np_array, # NumPy array of wavelength values
    output_excel_path
    ):
    """
    Saves the selected filters to an Excel sheet in a format where:
    - Column 1: Filter Names
    - Row 1 (Header): "FilterName", Wavelength_1, Wavelength_2, ...
    - Subsequent rows: FilterName_i, T_i_WL1, T_i_WL2, ...
    """
    if not selected_filter_names:
        print("No filters selected to save to Excel.")
        return

    # 1. Filter the original DataFrame to get only the selected filters
    # df_processed_filters_as_rows has filter names as its index.
    # .loc[selected_filter_names] will select rows corresponding to these names,
    # preserving the order of selected_filter_names if they are all present.
    selected_filters_df = df_processed_filters_as_rows.loc[selected_filter_names]

    # 2. Prepare the data for the desired output format
    # The index of selected_filters_df already contains the filter names.
    # We need to reset the index to make "FilterName" a regular column.
    output_df = selected_filters_df.reset_index()

    # Rename the new 'index' column (which contains filter names) to "FilterName"
    # The column name for filter names in your MATLAB code was implicitly taken from the first column.
    # Let's make it explicit.
    output_df.rename(columns={'index': 'FilterName'}, inplace=True) # 'index' is the default name after reset_index

    # 3. Set the column headers for wavelengths
    # The current column headers of selected_filters_df (and thus output_df after reset_index,
    # except for the 'FilterName' column) are already the wavelength values (as strings or numbers).
    # If they are numbers, Excel will handle them fine. If they are strings like "800nm",
    # readtable in MATLAB might read them as strings. It's often better if they are purely numeric
    # for wavelength headers if possible, or consistent strings like 'WL_800'.
    # The current script already tries to parse/generate numeric wavelengths.
    # The column headers of `selected_filters_df` (which become columns 1 to N in `output_df` after `reset_index`)
    # are taken from `df_processed_filters_as_rows.columns`, which are derived from `wavelengths_np_array`.
    # Let's ensure these wavelength columns are named clearly, perhaps just the numbers.

    # Create new column names for the wavelength part:
    # First column is 'FilterName', rest are wavelength values as strings
    new_column_names = ['FilterName'] + [str(int(w)) if w.is_integer() else str(w) for w in wavelengths_np_array]

    # Before assigning new names, check if the number of columns matches.
    # output_df has 1 (for FilterName) + num_wavelengths columns.
    # new_column_names also has 1 + num_wavelengths elements.
    if len(output_df.columns) == len(new_column_names):
        output_df.columns = new_column_names
    else:
        print(f"Warning: Mismatch in column count for output Excel. Expected {len(new_column_names)}, got {len(output_df.columns)}. Using default headers.")


    # 4. Save to Excel
    try:
        # index=False because we don't want pandas' default 0,1,2... index written to the file.
        # header=True is default and writes the column names (FilterName, 800, 810, ...)
        output_df.to_excel(output_excel_path, index=False, header=True)
        print(f"Selected filters data saved to: {output_excel_path} (MATLAB-friendly format)")
    except Exception as e:
        print(f"Error saving MATLAB-friendly Excel file: {e}")


def main():
    current_os = platform.system()
    if current_os == 'Windows':
        base_path = r"V:\SimulationData\Rahul\Hyperspectral Imaging Project"
        dummy_file_base_path_for_creation = "."
    elif current_os == 'Darwin':
        base_path = '/Volumes/ValentineLab/SimulationData/Rahul/Hyperspectral Imaging Project'
        dummy_file_base_path_for_creation = "."
    else:
        base_path = './SimulationData/Rahul/Hyperspectral Imaging Project'
        dummy_file_base_path_for_creation = "."
        print(f"Unrecognized OS '{current_os}'. Using default base path: '{base_path}'")

    # --- Configuration ---
    input_excel_file_relative = os.path.join("Machine Learning Codes", "Filter CSV files", "Feb1125_FilterList.xlsx")
    input_excel_file = os.path.join(base_path, input_excel_file_relative)

    num_filters_to_select_user = 16
    use_gpu_if_available = True

    # MINIMUM AVERAGE Transmission Constraint Parameter
    apply_min_avg_transmission_constraint = True # Changed variable name for clarity
    minimum_required_average_transmission_value = 0.6 # The threshold

    output_dir_relative = os.path.join("Machine Learning Codes", "Filter CSV files", "Correlation Selected Filters")
    output_dir = os.path.join(base_path, output_dir_relative)

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating output directory '{output_dir}': {e}")

    output_plot_file = os.path.join(output_dir, f"selected_{num_filters_to_select_user}_filters_plot.png")
    output_excel_file = os.path.join(output_dir, f"selected_{num_filters_to_select_user}_filters_data.xlsx")
    # --- End Configuration ---

    print(f"Input Excel file path: {input_excel_file}")
    print(f"Output directory path: {output_dir}")

    print(f"\nLoading data from: {input_excel_file}")
    try:
        df_processed_filters_as_rows, all_original_filter_names_list, spectra_for_sam, wavelengths = load_data(input_excel_file)
        print(f"Successfully loaded: {len(all_original_filter_names_list)} filters, "
              f"{len(wavelengths)} wavelength points ({wavelengths[0]:.0f}nm to {wavelengths[-1]:.0f}nm).")
    except FileNotFoundError:
        print(f"Error: Input Excel file not found: '{input_excel_file}'.")
        dummy_excel_filename = "optical_filters_dummy.xlsx"
        dummy_excel_full_path = os.path.join(dummy_file_base_path_for_creation, dummy_excel_filename)
        print(f"Attempting to create dummy: '{dummy_excel_full_path}'")
        num_dummy_filters, num_dummy_wl = 30, 50
        dummy_names = [f"DF_{i+1}" for i in range(num_dummy_filters)]
        dummy_wl_cols = [f"{int(wl)}" for wl in np.linspace(800, 1700, num_dummy_wl)]
        dummy_spectra = np.random.rand(num_dummy_filters, num_dummy_wl)
        for i in range(num_dummy_filters // 3): # Some filters with higher average T
            dummy_spectra[i, :] = np.random.rand(num_dummy_wl) * 0.4 + 0.5 # Avg T likely >= 0.5
        if num_dummy_filters > 0 and num_dummy_wl > 0:
             dummy_spectra[num_dummy_filters-1,0] = "BAD_DATA"
        dummy_df = pd.DataFrame(dummy_spectra, index=dummy_names, columns=dummy_wl_cols)
        try:
            dummy_df.to_excel(dummy_excel_full_path, index=True, header=True)
            print(f"Dummy file '{dummy_excel_full_path}' created.")
        except Exception as e_df_save:
             print(f"Failed to create dummy file: {e_df_save}")
        return
    except ValueError as e:
        print(f"Error loading/parsing: {e}")
        return

    if spectra_for_sam.shape[1] == 0:
        print("No filter data processed. Exiting.")
        return

    # Calculate AVERAGE transmissions for all filters
    all_avg_transmissions_series = df_processed_filters_as_rows.mean(axis=1, skipna=True)
    all_avg_transmissions_np = all_avg_transmissions_series.values

    print(f"\n--- Average Transmission Stats (across {len(all_original_filter_names_list)} filters) ---")
    if all_avg_transmissions_np.size > 0 and not np.all(np.isnan(all_avg_transmissions_np)):
        print(f"Min of average transmissions: {np.nanmin(all_avg_transmissions_np):.3f}")
        print(f"Max of average transmissions: {np.nanmax(all_avg_transmissions_np):.3f}")
        print(f"Mean of average transmissions: {np.nanmean(all_avg_transmissions_np):.3f}")
    else:
        print("No valid average transmission data to display stats for.")
    if apply_min_avg_transmission_constraint: # Updated condition name
        print(f"Required minimum average transmission for selection: >= {minimum_required_average_transmission_value:.3f}")
    print("------------------------------------\n")

    actual_use_gpu = use_gpu_if_available and CUPY_AVAILABLE
    print("\n--- Computation Backend ---")
    if use_gpu_if_available:
        if CUPY_AVAILABLE: print("GPU use requested and CuPy is available. Using GPU (CuPy).")
        else:
            print("GPU use requested, BUT CuPy is not available or failed to initialize.")
            if CUPY_INIT_ERROR_MSG: print(f"CuPy Init Details: {CUPY_INIT_ERROR_MSG}")
            print("Falling back to CPU (NumPy).")
            actual_use_gpu = False
    else: print("GPU use not requested. Using CPU (NumPy).")
    print("---------------------------\n")

    print("Calculating SAM matrix...")
    sam_matrix = calculate_sam_matrix(spectra_for_sam, use_gpu=actual_use_gpu)

    if sam_matrix.size == 0 and spectra_for_sam.shape[1] > 0 :
        print("Error: SAM matrix is empty.")
        return

    print(f"Selecting up to {num_filters_to_select_user} filters...")
    if apply_min_avg_transmission_constraint: # Updated condition name
        print(f"Applying constraint: minimum average transmission >= {minimum_required_average_transmission_value:.3f}")
        selected_filter_names = select_least_correlated_filters(
            sam_matrix,
            all_original_filter_names_list,
            num_filters_to_select_user,
            all_avg_transmissions_np, # Pass the array of averages
            minimum_required_average_transmission_value # Pass the threshold
        )
    else:
        print("No minimum average transmission constraint applied.")
        selected_filter_names = select_least_correlated_filters(
            sam_matrix,
            all_original_filter_names_list,
            num_filters_to_select_user,
            all_avg_transmissions_np,
            -np.inf # Makes all filters (with non-NaN avg_t) eligible
        )

    if not selected_filter_names:
        print("No filters were selected that meet all criteria.")
        return
    print(f"\nSelected filters (by name): {', '.join(selected_filter_names)}")
    print("Selected filters (by 1-based original index):")
    for name in selected_filter_names:
        try:
            idx_1_based = all_original_filter_names_list.index(name) + 1
            print(f"  - Index {idx_1_based} (Name: {name})")
        except ValueError:
            print(f"  - Name: {name} (Original index not found)")
    print(f"Number of filters selected: {len(selected_filter_names)}")

    print("\nGenerating spectral plot...")
    plot_selected_filters(
        df_processed_filters_as_rows,
        selected_filter_names,
        all_original_filter_names_list,
        wavelengths, 
        output_plot_file
    )

    print("Saving selected filter data to Excel...")
    save_selected_to_excel(df_processed_filters_as_rows, selected_filter_names, wavelengths, output_excel_file)

    print("\nProcess completed!")

if __name__ == "__main__":
    main()
