import time
import numpy as np
import torch
import json
import sys
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QComboBox, QPushButton, QLabel, QSlider, QLineEdit,
                           QTabWidget, QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView,
                           QFileDialog, QMessageBox, QProgressBar, QGroupBox, QSplitter)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import os
import rasterio
import pandas as pd

from config import config
from srnet_model import SpectralReconstructionNet
from dataset import FullImageHyperspectralDataset
from metrics import HyperspectralMetrics
from spect_dict import SpectralDictionary
import classifier
from reference_manager import ReferenceManager
from spect_dict import SpectralDictionary

class ReconstructionViewer(QMainWindow):
    """
    Enhanced GUI application for hyperspectral image reconstruction evaluation.

    Features:
    - Interactive image reconstruction and visualization
    - Comprehensive metrics calculation and display
    - Spectral and spatial analysis tools
    - Model efficiency evaluation
    - Spectral smoothness analysis
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Plastic HSI Reconstruction Viewer (SRNet) - Enhanced Metrics')
        self.setGeometry(100, 100, 1800, 1000)

        # Device configuration (early)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Viewer using device: {self.device}")

        # Initialize variables
        self.full_reconstruction = None
        self.original_image_chw_numpy = None # Will store C,H,W numpy array
        # self.wavelengths = None # Will be initialized below
        self.current_wavelength_idx = None
        self.current_metrics = None
        self.current_loss_components = None

        # --- Pre-compute/Cache Wavelengths ---
        self.wavelengths = config.full_wavelengths[config.wavelength_indices]
        self.num_selected_wavelengths = len(self.wavelengths)

        # --- Pre-compute/Cache Filter Matrix ---
        try:
            filters_pd = pd.read_csv(config.filter_path, header=None)
            filter_transmissions_np = filters_pd.iloc[:config.num_filters, 1:].values
            # Wavelengths from the CSV file
            csv_wavelengths_np = np.linspace(800, 1700, filter_transmissions_np.shape[1])

            interpolated_list = []
            for filter_spectrum_np in filter_transmissions_np:
                # Interpolate to the target wavelengths (self.wavelengths)
                interp_np = np.interp(self.wavelengths, csv_wavelengths_np, filter_spectrum_np)
                interpolated_list.append(interp_np)

            # self.filter_matrix will be [num_filters, num_selected_wavelengths]
            self.filter_matrix = torch.tensor(np.array(interpolated_list), dtype=torch.float32).to(self.device)
            print(f"Filter matrix loaded and interpolated: {self.filter_matrix.shape} on device {self.filter_matrix.device}")
        except Exception as e:
            QMessageBox.critical(self, "Filter Load Error", f"Failed to load or interpolate filters: {e}")
            raise # Stop if filters can't be loaded

        # --- Pre-compute Superpixel Filter Index Template ---
        # This template is for a single superpixel
        sp_h, sp_w = config.superpixel_height, config.superpixel_width
        self.superpixel_filter_indices_template = torch.zeros((sp_h, sp_w), dtype=torch.long)
        for di_ in range(sp_h):
            for dj_ in range(sp_w):
                self.superpixel_filter_indices_template[di_, dj_] = (di_ * sp_w + dj_) % config.num_filters


        self.data_dir = config.dataset_path

        # Make sure the directory exists, otherwise use a default location
        if not os.path.exists(self.data_dir):
            self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)

        # Create main widget with tab layout
        main_widget = QTabWidget()
        self.setCentralWidget(main_widget)

        # Create tabs
        self.reconstruction_tab = QWidget()
        self.metrics_tab = QWidget()
        self.analysis_tab = QWidget()

        main_widget.addTab(self.reconstruction_tab, "Reconstruction Viewer")
        main_widget.addTab(self.metrics_tab, "Metrics & Performance")
        main_widget.addTab(self.analysis_tab, "Analysis Tools")

        # Setup Reconstruction Tab
        self._setup_reconstruction_tab()

        # Setup Metrics Tab
        self._setup_metrics_tab()

        # Setup Analysis Tab
        self._setup_analysis_tab()

        # --- Add Reference Manager and Classifier ---
        self.reference_manager = None
        self.classifier = None
        # --- Load references early for model initialization too ---
        reference_spectra_tensor = None # Initialize as None
        try:
            ref_file = config.reference_spectra_path
            if os.path.exists(ref_file):
                self.reference_manager = ReferenceManager(ref_file)
                self.classifier = classifier.SpectralClassifier(self.reference_manager) # Keep classifier init here

                # --- Pre-process references for the model ---
                ref_dict = self.reference_manager.get_all_spectra()
                if ref_dict:
                    ref_list = [torch.tensor(spec, dtype=torch.float32) for spec in ref_dict.values()]
                    if ref_list:
                        temp_tensor = torch.stack(ref_list)
                        ref_norms = torch.linalg.norm(temp_tensor, dim=1, keepdim=True) + 1e-8
                        reference_spectra_tensor = temp_tensor / ref_norms
                        print(f"Test.py: Loaded and pre-processed {len(ref_dict)} reference spectra for model.")
            else:
                 QMessageBox.warning(self, "Reference File Missing",
                                     f"Reference file not found, classification/weighted loss might be affected:\n{ref_file}")
                 self.reference_manager = None # Ensure it's None if file missing
                 self.classifier = None
        except Exception as e:
             QMessageBox.critical(self, "Reference File Error",
                                   f"Could not load or process reference file:\n{ref_file}\n{e}")
             self.reference_manager = None
             self.classifier = None
        # --- End Reference Loading for Model ---


        # Load the trained model
        self.model = SpectralReconstructionNet(
            input_channels=1,
            out_channels=len(config.wavelength_indices),
            dim=64,
            deep_stage=3,
            num_blocks=[1, 2, 3],
            num_heads=[2, 4, 8],
            # --- Pass the loaded references here ---
            reference_spectra=reference_spectra_tensor,
            use_spectral_dict=True  # Keep this consistent with training
        )
        # Now load the saved weights
        state_dict = torch.load(config.model_save_path, map_location='cpu')
        # Filter out potential non-model keys if they exist (safer loading)
        filtered_state_dict = {k: v for k, v in state_dict.items() if not (k.endswith("total_ops") or k.endswith("total_params"))}
        # Load the state dict - use strict=False if you encounter issues, but True is preferred
        try:
            self.model.load_state_dict(filtered_state_dict, strict=True)
            print("Model state dictionary loaded successfully.")
        except RuntimeError as e:
             print(f"Error loading state dict (strict=True): {e}")
             print("Attempting to load with strict=False...")
             try:
                 self.model.load_state_dict(filtered_state_dict, strict=False)
                 print("Model state dictionary loaded with strict=False (some keys might have been ignored).")
             except Exception as final_e:
                  QMessageBox.critical(self, "Model Load Error", f"Failed to load model weights:\n{final_e}")
                  raise final_e # Stop execution if model cannot be loaded


        self.model = self.model.to(self.device)
        self.model.eval()


        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Populate band comparison dropdowns (wavelengths are now in self.wavelengths)
        for i in range(self.num_selected_wavelengths):
            wavelength_val = self.wavelengths[i]
            self.band1_combo.addItem(f"{wavelength_val:.1f} nm")
            self.band2_combo.addItem(f"{wavelength_val:.1f} nm")
        self.band1_combo.setCurrentIndex(0)
        self.band2_combo.setCurrentIndex(self.num_selected_wavelengths // 2)

        # Wavelength slider (max value based on num_selected_wavelengths)
        self.wavelength_slider.setMaximum(self.num_selected_wavelengths - 1)
        self.wavelength_slider.setValue(self.num_selected_wavelengths // 2)


        self.canvas.mpl_connect('button_press_event', self.on_click)
        self._analyze_model_efficiency()
        if self.classifier:
            self.target_material_combo.addItems(self.classifier.material_names)
        if not os.path.exists(config.results_path):
            os.makedirs(config.results_path, exist_ok=True)

        # # Initialize band comparison dropdown values after model is loaded
        # for i in range(len(config.wavelength_indices)):
        #     wavelength = config.full_wavelengths[config.wavelength_indices[i]]
        #     self.band1_combo.addItem(f"{wavelength:.1f} nm")
        #     self.band2_combo.addItem(f"{wavelength:.1f} nm")

        # # Add default values
        # self.band1_combo.setCurrentIndex(0)
        # self.band2_combo.setCurrentIndex(len(config.wavelength_indices) // 2)

        # # Connect click events
        # self.canvas.mpl_connect('button_press_event', self.on_click)

        # # Analyze model efficiency
        # self._analyze_model_efficiency()

        # # Populate analysis tab dropdowns if classifier loaded
        # if self.classifier:
        #     self.target_material_combo.addItems(self.classifier.material_names)


        # # Create results directory if it doesn't exist
        # if not os.path.exists(config.results_path):
        #     os.makedirs(config.results_path, exist_ok=True)

    def _setup_reconstruction_tab(self):
        """Set up the reconstruction visualization tab."""
        recon_layout = QVBoxLayout(self.reconstruction_tab)

        # Create control panel for reconstruction tab
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)

        # Add file selection
        self.file_combo = QComboBox()
        self.file_combo.addItems(self._get_available_files())
        control_layout.addWidget(QLabel("Select File:"))
        control_layout.addWidget(self.file_combo)

        # Add reconstruct button
        self.reconstruct_btn = QPushButton("Reconstruct")
        self.reconstruct_btn.clicked.connect(self.reconstruct_image)
        control_layout.addWidget(self.reconstruct_btn)

        # Add wavelength slider
        self.wavelength_slider = QSlider(Qt.Horizontal)
        self.wavelength_slider.setMinimum(0)
        self.wavelength_slider.setMaximum(len(config.wavelength_indices) - 1)
        self.wavelength_slider.setValue(len(config.wavelength_indices) // 2)
        self.wavelength_slider.valueChanged.connect(self.update_wavelength_display)
        control_layout.addWidget(QLabel("Wavelength:"))
        control_layout.addWidget(self.wavelength_slider)

        # Add wavelength display label
        self.wavelength_label = QLabel("Wavelength: N/A")
        control_layout.addWidget(self.wavelength_label)

        # Add coordinate inputs
        control_layout.addWidget(QLabel("X:"))
        self.x_coord = QLineEdit()
        self.x_coord.setFixedWidth(50)
        control_layout.addWidget(self.x_coord)

        control_layout.addWidget(QLabel("Y:"))
        self.y_coord = QLineEdit()
        self.y_coord.setFixedWidth(50)
        control_layout.addWidget(self.y_coord)

        # Add plot spectrum button
        self.plot_spectrum_btn = QPushButton("Plot Spectrum")
        self.plot_spectrum_btn.clicked.connect(self.plot_coordinates)
        control_layout.addWidget(self.plot_spectrum_btn)

        # Add time display
        self.time_label = QLabel("Time: N/A")
        control_layout.addWidget(self.time_label)

        recon_layout.addWidget(control_panel)

        # Create matplotlib figures for reconstruction tab
        self.figure = plt.figure(figsize=(16, 8))
        self.gs = self.figure.add_gridspec(2, 2)
        self.ax_orig = self.figure.add_subplot(self.gs[0, 0])
        self.ax_recon = self.figure.add_subplot(self.gs[0, 1])
        self.ax_spectrum = self.figure.add_subplot(self.gs[1, :])
        self.canvas = FigureCanvas(self.figure)
        recon_layout.addWidget(self.canvas)

        # Add save button for current visualization
        save_layout = QHBoxLayout()
        self.save_visualization_btn = QPushButton("Save Current Visualization")
        self.save_visualization_btn.clicked.connect(self.save_current_visualization)
        save_layout.addWidget(self.save_visualization_btn)

        # Add export metrics button
        self.export_metrics_btn = QPushButton("Export Metrics")
        self.export_metrics_btn.clicked.connect(self.export_current_metrics)
        save_layout.addWidget(self.export_metrics_btn)

        # Add save results to layout
        recon_layout.addLayout(save_layout)

    def _load_reference_data(self):
        ref_file = os.path.join(config.reference_spectra_path)
        if os.path.exists(ref_file):
            try: 
                self.reference_manager = ReferenceManager(ref_file)
                self.classifier = classifier.SpectralClassifier(self.reference_manager)
            except Exception as e:
                 QMessageBox.critical(self, "Reference File Error",
                                       f"Could not load or process reference file:\n{ref_file}\n{e}")
                 self.reference_manager = None
                 self.classifier = None
        else:
             QMessageBox.warning(self, "Reference File Missing",
                                  f"Reference file not found, classification disabled:\n{ref_file}")

    def _setup_metrics_tab(self):
        """Set up the metrics and performance analysis tab."""
        metrics_layout = QVBoxLayout(self.metrics_tab)

        # Add a title
        title_label = QLabel("Quantitative Reconstruction Metrics")
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        metrics_layout.addWidget(title_label)

        # Create two columns for metrics display
        metrics_section = QWidget()
        metrics_grid = QGridLayout(metrics_section)

        # Left column: Quality metrics
        quality_group = QGroupBox("Reconstruction Quality Metrics")
        quality_layout = QGridLayout(quality_group)

        # Create quality metrics labels
        self.metrics_labels = {}
        quality_metrics = [
            ("PSNR", "N/A dB", "Peak Signal-to-Noise Ratio - Higher is better"),
            ("SSIM", "N/A", "Structural Similarity Index - Higher is better"),
            ("RMSE", "N/A", "Root Mean Square Error - Lower is better"),
            ("MRAE", "N/A", "Mean Relative Average Error - Lower is better"),
            ("Spectral Fidelity", "N/A", "Similarity between spectral signatures - Higher is better"),
            ("SAM", "N/A", "Spectral Angle Mapper - Lower is better")
        ]

        # Add metrics to the layout with descriptions
        for i, (metric_name, default_value, tooltip) in enumerate(quality_metrics):
            # Label for metric name
            name_label = QLabel(f"{metric_name}:")
            name_label.setToolTip(tooltip)
            quality_layout.addWidget(name_label, i, 0)

            # Label for metric value
            self.metrics_labels[metric_name] = QLabel(default_value)
            self.metrics_labels[metric_name].setStyleSheet("font-weight: bold;")
            self.metrics_labels[metric_name].setToolTip(tooltip)
            quality_layout.addWidget(self.metrics_labels[metric_name], i, 1)

        metrics_grid.addWidget(quality_group, 0, 0)

        # Right column: Efficiency metrics
        efficiency_group = QGroupBox("Model Efficiency Metrics")
        efficiency_layout = QVBoxLayout(efficiency_group)

        # Create model efficiency table
        self.efficiency_table = QTableWidget(3, 2)
        self.efficiency_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.efficiency_table.verticalHeader().setVisible(False)
        self.efficiency_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.efficiency_table.setItem(0, 0, QTableWidgetItem("Parameters"))
        self.efficiency_table.setItem(1, 0, QTableWidgetItem("FLOPs"))
        self.efficiency_table.setItem(2, 0, QTableWidgetItem("Inference Time"))

        self.efficiency_table.setItem(0, 1, QTableWidgetItem("Calculating..."))
        self.efficiency_table.setItem(1, 1, QTableWidgetItem("Calculating..."))
        self.efficiency_table.setItem(2, 1, QTableWidgetItem("Calculating..."))

        efficiency_layout.addWidget(self.efficiency_table)

        # Add configuration info
        config_label = QLabel(f"Model Configuration: {config.num_filters} filters, {config.superpixel_height}x{config.superpixel_width} superpixels")
        config_label.setStyleSheet("font-style: italic;")
        efficiency_layout.addWidget(config_label)

        metrics_grid.addWidget(efficiency_group, 0, 1)
        metrics_layout.addWidget(metrics_section)

        # Metrics visualization section
        viz_label = QLabel("Metrics Visualization")
        viz_label.setStyleSheet("font-size: 12pt; font-weight: bold;")
        metrics_layout.addWidget(viz_label)

        self.metrics_figure = plt.figure(figsize=(12, 6))
        self.metrics_canvas = FigureCanvas(self.metrics_figure)
        metrics_layout.addWidget(self.metrics_canvas)

        # Batch evaluation section
        batch_group = QGroupBox("Batch Evaluation")
        batch_layout = QHBoxLayout(batch_group)

        self.batch_dir_btn = QPushButton("Select Directory")
        self.batch_dir_btn.clicked.connect(self.select_batch_directory)
        batch_layout.addWidget(self.batch_dir_btn)

        self.batch_evaluate_btn = QPushButton("Evaluate All Images")
        self.batch_evaluate_btn.clicked.connect(self.batch_evaluate)
        batch_layout.addWidget(self.batch_evaluate_btn)

        self.batch_progress = QProgressBar()
        self.batch_progress.setVisible(False)
        batch_layout.addWidget(self.batch_progress)

        metrics_layout.addWidget(batch_group)

    def _setup_analysis_tab(self):
        """Set up the analysis tools tab."""
        analysis_layout = QVBoxLayout(self.analysis_tab)

        # Add a title
        title_label = QLabel("Advanced Analysis Tools")
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        analysis_layout.addWidget(title_label)

        # Add analysis tools
        tools_section = QWidget()
        tools_layout = QHBoxLayout(tools_section)

        # Band comparison tool
        band_comparison = QGroupBox("Spectral Band Comparison")
        band_layout = QHBoxLayout(band_comparison)

        band_layout.addWidget(QLabel("Band 1:"))
        self.band1_combo = QComboBox()
        band_layout.addWidget(self.band1_combo)

        band_layout.addWidget(QLabel("Band 2:"))
        self.band2_combo = QComboBox()
        band_layout.addWidget(self.band2_combo)

        self.compare_bands_btn = QPushButton("Compare Bands")
        self.compare_bands_btn.clicked.connect(self.compare_spectral_bands)
        band_layout.addWidget(self.compare_bands_btn)

        tools_layout.addWidget(band_comparison)

        # Error analysis tools
        error_analysis = QGroupBox("Error Analysis")
        error_layout = QHBoxLayout(error_analysis)

        self.error_map_btn = QPushButton("RMSE Map")
        self.error_map_btn.clicked.connect(lambda: self.show_error_map("rmse"))
        error_layout.addWidget(self.error_map_btn)

        self.mae_map_btn = QPushButton("MAE Map")
        self.mae_map_btn.clicked.connect(lambda: self.show_error_map("mae"))
        error_layout.addWidget(self.mae_map_btn)

        self.sam_map_btn = QPushButton("Spectral Angle Map")
        self.sam_map_btn.clicked.connect(self.show_spectral_angle_map)
        error_layout.addWidget(self.sam_map_btn)

        # Add spectral smoothness analysis button
        self.smoothness_btn = QPushButton("Analyze Spectral Smoothness")
        self.smoothness_btn.clicked.connect(self.analyze_spectral_smoothness)
        error_layout.addWidget(self.smoothness_btn)

        tools_layout.addWidget(error_analysis)

        analysis_layout.addWidget(tools_section)

        # Analysis visualization
        self.analysis_figure = plt.figure(figsize=(12, 8))
        self.analysis_canvas = FigureCanvas(self.analysis_figure)
        analysis_layout.addWidget(self.analysis_canvas)

        # Results section
        results_section = QWidget()
        results_layout = QHBoxLayout(results_section)

        self.save_analysis_btn = QPushButton("Save Analysis Results")
        self.save_analysis_btn.clicked.connect(self.save_analysis_results)
        results_layout.addWidget(self.save_analysis_btn)

        self.export_analysis_btn = QPushButton("Export Analysis Data")
        self.export_analysis_btn.clicked.connect(self.export_analysis_data)
        results_layout.addWidget(self.export_analysis_btn)

        analysis_layout.addWidget(results_section)

        # --- Add Material Highlighting Tool ---
        highlight_group = QGroupBox("Material Identification")
        highlight_layout = QHBoxLayout(highlight_group)

        highlight_layout.addWidget(QLabel("Highlight Material:"))
        self.target_material_combo = QComboBox()
        # Items added in __init__ after classifier is loaded
        highlight_layout.addWidget(self.target_material_combo)

        self.highlight_btn = QPushButton("Highlight")
        self.highlight_btn.clicked.connect(self.highlight_material)
        highlight_layout.addWidget(self.highlight_btn)

        highlight_layout.addWidget(QLabel("SAM Threshold:"))
        self.sam_threshold_input = QLineEdit("0.025") # Default threshold
        self.sam_threshold_input.setFixedWidth(50)
        highlight_layout.addWidget(self.sam_threshold_input)

        tools_layout.addWidget(highlight_group) # Add this group to your existing tools layout

    def _get_available_files(self):
        """Get list of available HSI files (.npy or .tif) in the dataset directory."""
        try:
            files = os.listdir(self.data_dir)
            # include both .npy and .tif/.tiff
            hsi_files = sorted(
                f for f in files
                if f.lower().endswith(('.npy', '.tif', '.tiff'))
            )
            if not hsi_files:
                QMessageBox.warning(
                    self, "No Files Found",
                    f"No .npy or .tif files found in {self.data_dir}"
                )
            return hsi_files
        except Exception as e:
            QMessageBox.warning(
                self, "Warning",
                f"Could not access data directory:\n{self.data_dir}\n{str(e)}"
            )
            return []

    def _analyze_model_efficiency(self):
        """Analyze and display model efficiency metrics."""
        try:
            # Get sample input dimensions
            sample_input = torch.zeros((1, 1, 100, 100)).to(self.device)
            sample_filter = torch.zeros((1, len(config.wavelength_indices), 100, 100)).to(self.device)

            # Calculate efficiency metrics
            efficiency_metrics = HyperspectralMetrics.analyze_model_efficiency(
                self.model, sample_input.shape, sample_filter.shape, device=self.device
            )

            # Update efficiency table
            num_params = efficiency_metrics['num_params']
            param_str = f"{num_params:,}" if num_params < 1e6 else f"{num_params/1e6:.2f}M"
            self.efficiency_table.setItem(0, 1, QTableWidgetItem(param_str))

            flops = efficiency_metrics['flops']
            if flops > 1e9:
                flops_str = f"{flops/1e9:.2f} GFLOPs"
            else:
                flops_str = f"{flops/1e6:.2f} MFLOPs"
            self.efficiency_table.setItem(1, 1, QTableWidgetItem(flops_str))

            time_ms = efficiency_metrics['time_per_frame'] * 1000
            self.efficiency_table.setItem(2, 1, QTableWidgetItem(f"{time_ms:.2f} ms"))

            # Save to file for future reference
            os.makedirs('results', exist_ok=True)
            efficiency_data = {
                'parameters': num_params,
                'flops': flops,
                'time_per_frame_ms': time_ms,
                'model_config': {
                    'dimension': 64,
                    'stages': 3,
                    'blocks': [1, 2, 3],
                    'heads': [2, 4, 8]
                }
            }

            # Save to file (append if exists)
            try:
                if os.path.exists('results/efficiency_history.json'):
                    with open('results/efficiency_history.json', 'r') as f:
                        history = json.load(f)
                    history.append({
                        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                        'data': efficiency_data
                    })
                else:
                    history = [{
                        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                        'data': efficiency_data
                    }]

                with open('results/efficiency_history.json', 'w') as f:
                    json.dump(history, f, indent=2)
            except:
                print("Warning: Could not save efficiency metrics to file")
        except Exception as e:
            print(f"Error analyzing model efficiency: {str(e)}")
            self.efficiency_table.setItem(0, 1, QTableWidgetItem("Error"))
            self.efficiency_table.setItem(1, 1, QTableWidgetItem("Error"))
            self.efficiency_table.setItem(2, 1, QTableWidgetItem("Error"))

    def reconstruct_image(self):
        if not self.file_combo.currentText():
            QMessageBox.warning(self, "Warning", "No file selected.")
            return

        QApplication.setOverrideCursor(Qt.WaitCursor) # Show busy cursor
        start_time_total = time.time()
        selected_file = self.file_combo.currentText()

        try:
            fullpath = os.path.join(self.data_dir, selected_file)

            # 1. Load Image Data (NumPy)
            # This part is similar to your existing logic
            if selected_file.lower().endswith('.npy'):
                cube_np_bands_first = np.load(fullpath) # Assuming (bands, H, W)
                # Transpose to (H, W, bands) for consistency
                img_data_np_h_w_bands = np.transpose(cube_np_bands_first, (1, 2, 0))
            elif selected_file.lower().endswith(('.tif', '.tiff')):
                with rasterio.open(fullpath) as src:
                    img_data_np_bands_first = src.read() # (bands, H, W)
                img_data_np_h_w_bands = np.transpose(img_data_np_bands_first, (1, 2, 0)) # (H, W, bands)
            else:
                QMessageBox.critical(self, "Error", f"Unsupported file type: {selected_file}")
                QApplication.restoreOverrideCursor()
                return

            H_orig, W_orig, C_orig = img_data_np_h_w_bands.shape

            # 2. Padding (NumPy, then to Tensor)
            # Pad to be divisible by superpixel dimensions
            sp_h, sp_w = config.superpixel_height, config.superpixel_width
            pad_h = (sp_h - (H_orig % sp_h)) % sp_h
            pad_w = (sp_w - (W_orig % sp_w)) % sp_w

            # Using np.pad is often cleaner for this
            # Pad only height and width (axis 0 and 1), not channels (axis 2)
            # ((before_axis0, after_axis0), (before_axis1, after_axis1), ...)
            img_data_padded_np = np.pad(
                img_data_np_h_w_bands,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode='constant',
                constant_values=0
            )
            H_pad, W_pad, _ = img_data_padded_np.shape

            # 3. Convert to PyTorch Tensor and Select Wavelengths
            # Assuming img_data_padded_np has all C_orig bands and config.wavelength_indices
            # correctly selects from these C_orig bands.
            # If C_orig is already the 32 bands from 800-1700nm, and config.wavelength_indices = np.arange(32),
            # then this just takes all of them.
            # Ensure config.wavelength_indices are valid for C_orig dimension.
            if max(config.wavelength_indices) >= C_orig:
                 QMessageBox.critical(self, "Error", f"Wavelength index out of bounds. Max index: {max(config.wavelength_indices)}, C_orig: {C_orig}")
                 QApplication.restoreOverrideCursor()
                 return

            hypercube_slice_h_w_c = torch.from_numpy(
                img_data_padded_np[:, :, config.wavelength_indices]
            ).float() # Shape: [H_pad, W_pad, C_selected]

            # 4. Normalization (PyTorch on GPU)
            hypercube_slice_gpu = hypercube_slice_h_w_c.to(self.device)
            max_val = torch.max(hypercube_slice_gpu)
            if max_val > 0: # Avoid division by zero for blank images
                hypercube_slice_gpu = hypercube_slice_gpu / max_val

            # Store for visualization: permute to C, H, W and convert to NumPy
            # This is the ground truth for metrics
            self.original_image_chw_numpy = hypercube_slice_gpu.permute(2, 0, 1).cpu().numpy()

            # 5. Create Filter Pattern Tensor (Vectorized)
            # self.superpixel_filter_indices_template is [sp_h, sp_w]
            # Tile it to full padded image size
            num_h_tiles = H_pad // sp_h
            num_w_tiles = W_pad // sp_w

            # pixel_filter_indices_map is [H_pad, W_pad], values are 0 to num_filters-1
            pixel_filter_indices_map = self.superpixel_filter_indices_template.tile(num_h_tiles, num_w_tiles)

            # self.filter_matrix is [num_filters, C_selected] on device
            # Gather the full spectral curve for each pixel based on its assigned filter
            # filter_pattern_h_w_c will be [H_pad, W_pad, C_selected]
            filter_pattern_h_w_c = self.filter_matrix[pixel_filter_indices_map.to(self.device)] # Ensure map is on device

            # Permute to [C_selected, H_pad, W_pad] for model input
            filter_pattern_input_tensor = filter_pattern_h_w_c.permute(2, 0, 1)

            # 6. Create Filtered Measurements (Vectorized)
            # hypercube_slice_gpu is [H_pad, W_pad, C_selected]
            # filter_pattern_h_w_c is [H_pad, W_pad, C_selected] (from step 5)

            # Element-wise multiplication and sum over the channel dimension
            # This simulates applying each filter to its corresponding pixel's spectrum
            measurements_h_w = torch.sum(hypercube_slice_gpu * filter_pattern_h_w_c, dim=-1)

            # Add channel dimension: [1, H_pad, W_pad] for model input
            filtered_measurements_input_tensor = measurements_h_w.unsqueeze(0)

            # 7. Model Inference
            # Add batch dimension to inputs
            filtered_measurements_bchw = filtered_measurements_input_tensor.unsqueeze(0) # [1, 1, H_pad, W_pad]
            filter_pattern_bchw = filter_pattern_input_tensor.unsqueeze(0)       # [1, C_sel, H_pad, W_pad]

            inference_start_time = time.time()
            with torch.no_grad():
                reconstructed_bchw = self.model(filtered_measurements_bchw, filter_pattern_bchw)
            inference_time = time.time() - inference_start_time

            # Remove batch dim, result is [C_selected, H_pad, W_pad]
            self.full_reconstruction = reconstructed_bchw.cpu().squeeze(0) # Keep as tensor for metrics

            # --- Classification (after reconstruction, using self.full_reconstruction tensor) ---
            if self.classifier and self.full_reconstruction is not None:
                 print("Running classification...")
                 try:
                     sam_threshold = float(self.sam_threshold_input.text())
                 except ValueError:
                     QMessageBox.warning(self, "Invalid Input", "Invalid SAM threshold. Using 0.025.")
                     sam_threshold = 0.025
                     self.sam_threshold_input.setText("0.025")

                 # Classifier expects C, H, W tensor or numpy array
                 self.classification_map, self.sam_map = self.classifier.classify_image(
                     self.full_reconstruction, # Pass the PyTorch tensor directly
                     sam_threshold=sam_threshold
                 )
                 print("Classification complete.")
            else:
                 self.classification_map = None
                 self.sam_map = None
                 print("Classifier not available or reconstruction failed.")


            # 8. Calculate Metrics
            # Metrics functions expect (pred_tensor, target_tensor) or (pred_numpy, target_numpy)
            # self.original_image_chw_numpy is already NumPy C,H,W
            # self.full_reconstruction is PyTorch C,H,W
            metrics = HyperspectralMetrics.compute_all_metrics(
                self.full_reconstruction, # Pass tensor
                torch.from_numpy(self.original_image_chw_numpy) # Convert target to tensor for consistency
            )

            # Calculate model's loss components (for reporting, not training)
            with torch.no_grad():
                # Model's compute_loss expects batch dimension
                _, loss_components = self.model.compute_loss(
                    self.full_reconstruction.unsqueeze(0).to(self.device), # Add batch, move to device
                    torch.from_numpy(self.original_image_chw_numpy).unsqueeze(0).to(self.device), # Add batch, move to device
                    torch.nn.MSELoss()
                )
            # Convert loss components from tensors to floats if they are not already
            loss_components = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_components.items()}


            # Update UI
            total_elapsed_time = time.time() - start_time_total
            self.time_label.setText(f"Time: {total_elapsed_time:.2f}s (Infer: {inference_time*1000:.0f}ms)")

            self.metrics_labels["PSNR"].setText(f"{metrics['psnr']:.2f} dB")
            self.metrics_labels["SSIM"].setText(f"{metrics['ssim']:.4f}")
            self.metrics_labels["RMSE"].setText(f"{metrics['rmse']:.6f}")
            self.metrics_labels["MRAE"].setText(f"{metrics['mrae']:.4f}") # MRAE calculation was fine
            self.metrics_labels["Spectral Fidelity"].setText(f"{metrics['spectral_fidelity']:.4f}")

            sam_loss_val = loss_components.get('spectral_angle_loss', 0.0)
            self.metrics_labels["SAM"].setText(f"{sam_loss_val:.4f}")

            self._update_metrics_visualization(metrics, loss_components)
            self.update_wavelength_display() # This will use self.original_image_chw_numpy

            self.current_metrics = metrics
            self.current_loss_components = loss_components
            self.setWindowTitle(f'HSI Reconstruction - {selected_file} - PSNR: {metrics["psnr"]:.2f}dB')

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to reconstruct image: {str(e)}\nCheck console for traceback.")
            import traceback
            print(f"Error during reconstruction of {selected_file}:")
            traceback.print_exc()
        finally:
            QApplication.restoreOverrideCursor() # Ensure cursor is restored

    def update_wavelength_display(self):
        if self.wavelengths is None or self.original_image_chw_numpy is None or self.full_reconstruction is None:
            return

        idx = self.wavelength_slider.value()
        self.current_wavelength_idx = idx
        wavelength = self.wavelengths[idx] # self.wavelengths is already initialized
        self.wavelength_label.setText(f"Wavelength: {wavelength:.2f} nm")

        self.figure.clear()
        self.ax_orig = self.figure.add_subplot(self.gs[0, 0])
        self.ax_recon = self.figure.add_subplot(self.gs[0, 1])
        self.ax_spectrum = self.figure.add_subplot(self.gs[1, :])

        # Use the C,H,W numpy array for original
        orig_img = self.original_image_chw_numpy[idx]
        # Use the C,H,W tensor for reconstructed, convert to numpy for imshow
        recon_img = self.full_reconstruction[idx].numpy()

        # Use same color scale for fair comparison
        vmin = min(orig_img.min(), recon_img.min())
        vmax = max(orig_img.max(), recon_img.max())

        orig_plot = self.ax_orig.imshow(orig_img, cmap='viridis', vmin=vmin, vmax=vmax)
        recon_plot = self.ax_recon.imshow(recon_img, cmap='viridis', vmin=vmin, vmax=vmax)

        self.ax_orig.set_title(f'Original Image ({wavelength:.1f}nm)')
        self.ax_recon.set_title(f'Reconstructed Image ({wavelength:.1f}nm)')
        self.ax_orig.axis('off')
        self.ax_recon.axis('off')

        # Add colorbars
        self.figure.colorbar(orig_plot, ax=self.ax_orig, fraction=0.046, pad=0.04)
        self.figure.colorbar(recon_plot, ax=self.ax_recon, fraction=0.046, pad=0.04)

        # Setup the spectrum axis
        self.ax_spectrum.set_title('Click on image to see spectrum')
        self.ax_spectrum.set_xlabel('Wavelength (nm)')
        self.ax_spectrum.set_ylabel('Intensity')
        self.ax_spectrum.grid(True)

        self.figure.tight_layout()
        self.canvas.draw()

    def highlight_material(self):
        """Highlights pixels classified as the selected material."""
        if self.classifier is None:
            QMessageBox.warning(self, "Warning", "Classifier not initialized. Load reference file.")
            return
        if not hasattr(self, 'classification_map') or self.classification_map is None:
             QMessageBox.warning(self, "Warning", "Please reconstruct and classify an image first.")
             return
        if self.full_reconstruction is None:
             QMessageBox.warning(self, "Warning", "Please reconstruct an image first.")
             return

        selected_material = self.target_material_combo.currentText()
        if not selected_material:
            QMessageBox.warning(self, "Warning", "Please select a material to highlight.")
            return

        try:
            target_class_id = self.classifier.material_names.index(selected_material) + 1
        except ValueError:
             QMessageBox.warning(self, "Warning", f"Material '{selected_material}' not found in reference.")
             return

        # --- Visualization on the Analysis Tab Canvas ---
        self.analysis_figure.clear()
        ax1 = self.analysis_figure.add_subplot(121)
        ax2 = self.analysis_figure.add_subplot(122)

        # Display a base image (e.g., reconstructed image at a specific wavelength)
        display_band_idx = self.wavelength_slider.value() # Use current slider value
        base_image = self.full_reconstruction[display_band_idx].cpu().numpy()
        ax1.imshow(base_image, cmap='gray') # Display in grayscale for contrast
        ax1.set_title(f"Base Image ({self.wavelengths[display_band_idx]:.1f} nm)")
        ax1.axis('off')

        # Create highlight mask
        highlight_mask = (self.classification_map == target_class_id)

        # Create an RGBA overlay image
        # Make background transparent, highlight color for target class
        overlay = np.zeros((*highlight_mask.shape, 4)) # H, W, RGBA
        overlay[highlight_mask] = [1, 0, 0, 0.6] # Red with 60% opacity

        ax1.imshow(overlay) # Overlay the highlights
        ax1.set_title(f"Highlighted: {selected_material}")

        # Display the full classification map
        num_classes_total = len(self.classifier.material_names) + 1 # incl background
        cmap = plt.get_cmap('tab10', num_classes_total) # Discrete colormap
        im = ax2.imshow(self.classification_map, cmap=cmap, vmin=0, vmax=num_classes_total-1)
        ax2.set_title("Full Classification Map")
        ax2.axis('off')

        # Add a colorbar with labels
        formatter = plt.FuncFormatter(lambda val, loc: self.classifier.get_class_name(int(val)))
        ticks = np.arange(num_classes_total)
        self.analysis_figure.colorbar(im, ax=ax2, ticks=ticks, format=formatter)

        self.analysis_figure.tight_layout()
        self.analysis_canvas.draw()

    def plot_coordinates(self):
        """Plot spectrum for manually entered coordinates."""
        try:
            x = int(self.x_coord.text())
            y = int(self.y_coord.text())
            self.plot_spectrum(x, y)
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid integer coordinates.")
    def plot_spectrum(self, x, y):
        if self.original_image_chw_numpy is None or self.full_reconstruction is None:
            return

        self.ax_spectrum.clear()
        legend = self.ax_spectrum.get_legend()
        if legend is not None: legend.remove()

        # original_image_chw_numpy is [C, H, W]
        h, w = self.original_image_chw_numpy.shape[1:]
        if 0 <= y < h and 0 <= x < w:
            # Original spectrum from the NumPy array
            orig_spec_data = self.original_image_chw_numpy[:, y, x]
            # Reconstructed spectrum from the PyTorch tensor (convert to numpy for plotting)
            recon_spec_data = self.full_reconstruction[:, y, x].numpy()

            self.ax_spectrum.plot(self.wavelengths, orig_spec_data, 'b-', label='Original', linewidth=2)
            self.ax_spectrum.plot(self.wavelengths, recon_spec_data, 'r--', label='Reconstructed', linewidth=2)

            if self.current_wavelength_idx is not None:
                current_wl = self.wavelengths[self.current_wavelength_idx]
                orig_value = orig_spec_data[self.current_wavelength_idx]
                recon_value = recon_spec_data[self.current_wavelength_idx]
                self.ax_spectrum.axvline(current_wl, color='gray', linestyle=':', alpha=0.7)
                self.ax_spectrum.plot([current_wl], [orig_value], 'bo', markersize=8)
                self.ax_spectrum.plot([current_wl], [recon_value], 'ro', markersize=8)

            self.ax_spectrum.set_xlabel('Wavelength (nm)')
            self.ax_spectrum.set_ylabel('Intensity')
            self.ax_spectrum.set_title(f'Spectrum at ({x}, {y})')
            self.ax_spectrum.set_ylim(0, 1) # Keep fixed y-axis
            self.ax_spectrum.legend()
            self.ax_spectrum.grid(True)

            # Local metrics calculation (using NumPy arrays)
            rmse_local = np.sqrt(np.mean((orig_spec_data - recon_spec_data)**2))
            norm_orig = orig_spec_data / (np.linalg.norm(orig_spec_data) + 1e-8)
            norm_recon = recon_spec_data / (np.linalg.norm(recon_spec_data) + 1e-8)
            dot_product = np.dot(norm_orig, norm_recon)
            dot_product = np.clip(dot_product, -1.0, 1.0)
            spectral_angle_local = np.arccos(dot_product)
            corr_local = np.corrcoef(orig_spec_data, recon_spec_data)[0,1] if np.std(orig_spec_data) > 1e-6 and np.std(recon_spec_data) > 1e-6 else 0.0


            metrics_text = (
                f"Point Metrics at ({x},{y}):\n"
                f"RMSE: {rmse_local:.4f}\n"
                f"SAM: {spectral_angle_local:.4f} rad\n" # Changed from Spectral Angle
                f"Corr: {corr_local:.4f}"
            )
            self.ax_spectrum.text(
                0.98, 0.98, metrics_text,
                transform=self.ax_spectrum.transAxes,
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

            # Place the text in the upper right corner
            self.ax_spectrum.text(
                0.98, 0.98, metrics_text,
                transform=self.ax_spectrum.transAxes,
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

            self.canvas.draw()

    def on_click(self, event):
        """Handle click events to show spectrum at clicked point."""
        if event.inaxes in [self.ax_orig, self.ax_recon]:
            x, y = int(event.xdata), int(event.ydata)
            # Update coordinate input boxes
            self.x_coord.setText(str(x))
            self.y_coord.setText(str(y))
            self.plot_spectrum(x, y)

    def _update_metrics_visualization(self, metrics, loss_components):
        """Update the metrics visualization tab with current results."""
        self.metrics_figure.clear()

        # Create bar chart of key metrics
        ax1 = self.metrics_figure.add_subplot(121)
        metric_names = ['PSNR/50', 'SSIM', 'Spectral Fidelity']
        metric_values = [metrics['psnr']/50, metrics['ssim'], metrics['spectral_fidelity']]  # Normalize PSNR

        bars = ax1.bar(metric_names, metric_values, color=['green', 'blue', 'purple'])
        ax1.set_ylim(0, 1.0)
        ax1.set_title('Quality Metrics (Higher is Better)')

        # Add value labels
        for bar, name, value in zip(bars, metric_names, [metrics['psnr'], metrics['ssim'], metrics['spectral_fidelity']]):
            if name == 'PSNR/50':
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                         f"{value:.2f} dB", ha='center', va='bottom')
            else:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                         f"{value:.4f}", ha='center', va='bottom')

        # Add error metrics (lower is better)
        ax2 = self.metrics_figure.add_subplot(122)
        error_names = ['RMSE', 'MRAE', 'SAM']
        error_values = [
            metrics['rmse'],
            metrics['mrae'],
            loss_components.get('spectral_angle_loss', 0)
        ]

        bars = ax2.bar(error_names, error_values, color=['red', 'orange', 'brown'])
        ax2.set_title('Error Metrics (Lower is Better)')

        for bar, value in zip(bars, error_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{value:.4f}", ha='center', va='bottom')

        self.metrics_figure.tight_layout()
        self.metrics_canvas.draw()

    def compare_spectral_bands(self):
        """Compare reconstruction quality between two spectral bands."""
        if not hasattr(self, 'full_reconstruction') or self.full_reconstruction is None:
            QMessageBox.warning(self, "Warning", "Please reconstruct an image first.")
            return

        # Get selected band indices
        band1_idx = self.band1_combo.currentIndex()
        band2_idx = self.band2_combo.currentIndex()

        # Clear previous figure
        self.analysis_figure.clear()

        # Plot comparison
        ax1 = self.analysis_figure.add_subplot(221)
        ax2 = self.analysis_figure.add_subplot(222)
        ax3 = self.analysis_figure.add_subplot(223)
        ax4 = self.analysis_figure.add_subplot(224)

        # Original vs Reconstructed for band 1
        orig_band1 = self.original_image_chw_numpy[band1_idx]
        recon_band1 = self.full_reconstruction[band1_idx].numpy()

        ax1.imshow(orig_band1, cmap='viridis')
        ax1.set_title(f'Original Band {band1_idx} ({self.wavelengths[band1_idx]:.1f}nm)')
        ax1.axis('off')

        ax2.imshow(recon_band1, cmap='viridis')
        ax2.set_title(f'Reconstructed Band {band1_idx}')
        ax2.axis('off')

        # Original vs Reconstructed for band 2
        orig_band2 = self.original_image_chw_numpy[band2_idx]
        recon_band2 = self.full_reconstruction[band2_idx].numpy()

        ax3.imshow(orig_band2, cmap='viridis')
        ax3.set_title(f'Original Band {band2_idx} ({self.wavelengths[band2_idx]:.1f}nm)')
        ax3.axis('off')

        ax4.imshow(recon_band2, cmap='viridis')
        ax4.set_title(f'Reconstructed Band {band2_idx}')
        ax4.axis('off')

        # Calculate per-band metrics
        rmse1 = np.sqrt(np.mean((orig_band1 - recon_band1) ** 2))
        rmse2 = np.sqrt(np.mean((orig_band2 - recon_band2) ** 2))

        # Add metrics as text
        ax2.text(0.5, -0.1, f'RMSE: {rmse1:.4f}', ha='center', va='center',
                transform=ax2.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

        ax4.text(0.5, -0.1, f'RMSE: {rmse2:.4f}', ha='center', va='center',
                transform=ax4.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

        self.analysis_figure.tight_layout()
        self.analysis_canvas.draw()

    def show_error_map(self, error_type="rmse"):
        """Show spatial error distribution map."""
        if not hasattr(self, 'full_reconstruction') or self.full_reconstruction is None:
            QMessageBox.warning(self, "Warning", "Please reconstruct an image first.")
            return

        # Clear previous figure
        self.analysis_figure.clear()

        # Select current wavelength for analysis
        wavelength_idx = self.current_wavelength_idx if self.current_wavelength_idx is not None else 0
        wavelength = self.wavelengths[wavelength_idx]

        orig = self.original_image_chw_numpy[wavelength_idx]
        recon = self.full_reconstruction[wavelength_idx].numpy()

        # Create error maps
        if error_type == "rmse":
            error_map = np.abs(orig - recon)
            title = f'Absolute Error Map at {wavelength:.1f}nm'
            cmap = 'hot'
        elif error_type == "mae":
            error_map = np.abs(orig - recon)
            title = f'Mean Absolute Error Map at {wavelength:.1f}nm'
            cmap = 'hot'
        else:
            error_map = np.abs(orig - recon) / (np.abs(orig) + 1e-6)
            title = f'Relative Error Map at {wavelength:.1f}nm'
            cmap = 'hot'

        # Create subplots
        ax1 = self.analysis_figure.add_subplot(131)
        ax2 = self.analysis_figure.add_subplot(132)
        ax3 = self.analysis_figure.add_subplot(133)

        # Plot original and reconstructed images
        ax1.imshow(orig, cmap='viridis')
        ax1.set_title(f'Original at {wavelength:.1f}nm')
        ax1.axis('off')

        ax2.imshow(recon, cmap='viridis')
        ax2.set_title(f'Reconstructed at {wavelength:.1f}nm')
        ax2.axis('off')

        # Plot error map
        im = ax3.imshow(error_map, cmap=cmap)
        ax3.set_title(title)
        ax3.axis('off')
        self.analysis_figure.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

        # Add overall metrics
        if error_type == "rmse":
            global_error = np.sqrt(np.mean((orig - recon) ** 2))
            metric_name = "RMSE"
        elif error_type == "mae":
            global_error = np.mean(np.abs(orig - recon))
            metric_name = "MAE"
        else:
            global_error = np.mean(np.abs(orig - recon) / (np.abs(orig) + 1e-6))
            metric_name = "MRAE"

        ax3.text(0.5, -0.1, f'Global {metric_name}: {global_error:.4f}',
                ha='center', va='center', transform=ax3.transAxes,
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

        self.analysis_figure.tight_layout()
        self.analysis_canvas.draw()

    def show_spectral_angle_map(self):
        """Show spectral angle mapper (SAM) visualization."""
        if not hasattr(self, 'full_reconstruction') or self.full_reconstruction is None:
            QMessageBox.warning(self, "Warning", "Please reconstruct an image first.")
            return

        # Clear previous figure
        self.analysis_figure.clear()

        # Calculate Spectral Angle Map
        orig = self.original_image_chw_numpy  # Shape: [C, H, W]
        recon = self.full_reconstruction.numpy()  # Shape: [C, H, W]

        # Reshape for easier calculations
        c, h, w = orig.shape
        orig_reshaped = orig.reshape(c, -1)  # [C, H*W]
        recon_reshaped = recon.reshape(c, -1)  # [C, H*W]

        # Normalize each spectral vector
        orig_norm = np.linalg.norm(orig_reshaped, axis=0, keepdims=True)
        recon_norm = np.linalg.norm(recon_reshaped, axis=0, keepdims=True)

        # Avoid division by zero
        orig_normalized = orig_reshaped / (orig_norm + 1e-8)
        recon_normalized = recon_reshaped / (recon_norm + 1e-8)

        # Calculate dot product and clip to valid range
        dot_product = np.sum(orig_normalized * recon_normalized, axis=0)
        dot_product = np.clip(dot_product, -1.0, 1.0)

        # Calculate spectral angle
        sam_map = np.arccos(dot_product)
        sam_map = sam_map.reshape(h, w)

        # Plot results
        ax1 = self.analysis_figure.add_subplot(121)
        SAM_VMIN = 0.0
        SAM_VMAX = np.pi / 8
        im = ax1.imshow(sam_map, cmap='hot', vmin=SAM_VMIN, vmax=SAM_VMAX)
        ax1.set_title('Spectral Angle Map (SAM)')
        ax1.axis('off')
        self.analysis_figure.colorbar(im, ax=ax1, label='Angle (radians)')

        # Plot histogram of SAM values
        ax2 = self.analysis_figure.add_subplot(122)
        ax2.hist(sam_map.flatten(), bins=50, color='red', alpha=0.7)
        ax2.set_title('Distribution of Spectral Angles')
        ax2.set_xlabel('Spectral Angle (radians)')
        ax2.set_ylabel('Frequency')

        # Add mean SAM value
        mean_sam = np.mean(sam_map)
        ax2.axvline(mean_sam, color='black', linestyle='--',
                   label=f'Mean: {mean_sam:.4f} rad')
        ax2.legend()

        # Add global metric to figure
        self.analysis_figure.suptitle(
            f'Spectral Angle Analysis - Mean SAM: {mean_sam:.4f} rad',
            fontsize=14
        )

        self.analysis_figure.tight_layout(rect=[0, 0, 1, 0.95])
        self.analysis_canvas.draw()

    def analyze_spectral_smoothness(self):
        """Analyze and visualize the spectral smoothness improvement."""
        if not hasattr(self, 'full_reconstruction') or self.full_reconstruction is None:
            QMessageBox.warning(self, "Warning", "Please reconstruct an image first.")
            return

        # Clear previous figure
        self.analysis_figure.clear()

        # Pick a few random pixels to analyze
        h, w = self.original_image_chw_numpy.shape[1:]
        num_pixels = 4
        pixels = []

        # Try to find pixels with interesting spectra
        for _ in range(20):  # Try up to 20 random positions to find good examples
            if len(pixels) >= num_pixels:
                break

            x, y = np.random.randint(0, w), np.random.randint(0, h)
            orig_spectrum = self.original_image_chw_numpy[:, y, x]

            # Check if spectrum has enough variation (not just flat)
            if np.std(orig_spectrum) > 0.05:
                pixels.append((x, y))

        # If we couldn't find enough interesting pixels, just pick random ones
        while len(pixels) < num_pixels:
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            pixels.append((x, y))

        # Calculate spectral derivatives for each pixel
        for i, (x, y) in enumerate(pixels):
            # Get spectra
            orig_spectrum = self.original_image_chw_numpy[:, y, x]
            recon_spectrum = self.full_reconstruction[:, y, x].numpy()

            # Create subplot
            ax_spec = self.analysis_figure.add_subplot(2, 4, i*2 + 1)
            ax_deriv = self.analysis_figure.add_subplot(2, 4, i*2 + 2)

            # Plot spectra
            ax_spec.plot(self.wavelengths, orig_spectrum, 'b-', label='Original', linewidth=2)
            ax_spec.plot(self.wavelengths, recon_spectrum, 'r-', label='Reconstructed', linewidth=2)
            ax_spec.set_title(f'Spectrum at ({x}, {y})')
            if i == 0:
                ax_spec.legend()

            # Calculate and plot derivatives (for smoothness analysis)
            orig_deriv = np.diff(orig_spectrum)
            recon_deriv = np.diff(recon_spectrum)

            derivative_x = self.wavelengths[1:]  # x-axis for derivatives

            ax_deriv.plot(derivative_x, orig_deriv, 'b-', label='Original Derivative', alpha=0.7)
            ax_deriv.plot(derivative_x, recon_deriv, 'r-', label='Reconstructed Derivative', alpha=0.7)
            ax_deriv.set_title('Spectral Derivatives (Smoothness)')
            if i == 0:
                ax_deriv.legend()

            # Calculate smoothness metrics
            orig_roughness = np.sqrt(np.mean(orig_deriv**2))
            recon_roughness = np.sqrt(np.mean(recon_deriv**2))

            # Add roughness values to the plot
            ax_deriv.text(0.05, 0.95, f"Original roughness: {orig_roughness:.4f}",
                     transform=ax_deriv.transAxes, va='top', ha='left',
                     bbox=dict(facecolor='white', alpha=0.7))
            ax_deriv.text(0.05, 0.85, f"Reconstructed roughness: {recon_roughness:.4f}",
                     transform=ax_deriv.transAxes, va='top', ha='left',
                     bbox=dict(facecolor='white', alpha=0.7))

            # If the reconstructed is smoother, highlight it
            if recon_roughness < orig_roughness:
                ax_deriv.text(0.05, 0.75, "✓ Reconstruction is smoother",
                     transform=ax_deriv.transAxes, va='top', ha='left',
                     color='green', fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.7))

        # Add overall title
        self.analysis_figure.suptitle("Spectral Smoothness Analysis", fontsize=14)
        self.analysis_figure.tight_layout(rect=[0, 0, 1, 0.95])
        self.analysis_canvas.draw()

    def save_current_visualization(self):
        """Save the current visualization to a file."""
        if not hasattr(self, 'full_reconstruction') or self.full_reconstruction is None:
            QMessageBox.warning(self, "Warning", "No reconstruction to save.")
            return

        # Get file name
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Visualization",
            os.path.join(config.results_path, "visualization.png"),
            "PNG Files (*.png);;All Files (*)"
        )

        if not file_path:
            return

        # Save figure
        self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
        QMessageBox.information(self, "Success", f"Visualization saved to {file_path}")

    def export_current_metrics(self):
        """Export the current metrics to a CSV file."""
        if not hasattr(self, 'current_metrics') or self.current_metrics is None:
            QMessageBox.warning(self, "Warning", "No metrics to export.")
            return

        # Get file name
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Metrics",
            os.path.join(config.results_path, "metrics.csv"),
            "CSV Files (*.csv);;All Files (*)"
        )

        if not file_path:
            return

        # Combine all metrics
        all_metrics = {
            'filename': self.file_combo.currentText(),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **self.current_metrics,
            **{f'loss_{k}': v for k, v in self.current_loss_components.items()},
            'inference_time_ms': float(self.time_label.text().split(':')[1].strip('s')) * 1000
        }

        # Write to CSV
        df = pd.DataFrame([all_metrics])
        df.to_csv(file_path, index=False)
        QMessageBox.information(self, "Success", f"Metrics exported to {file_path}")

    def save_analysis_results(self):
        """Save the current analysis visualization to a file."""
        if len(self.analysis_figure.axes) == 0:
            QMessageBox.warning(self, "Warning", "No analysis to save.")
            return

        # Get file name
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Analysis",
            os.path.join(config.results_path, "analysis.png"),
            "PNG Files (*.png);;All Files (*)"
        )

        if not file_path:
            return

        # Save figure
        self.analysis_figure.savefig(file_path, dpi=300, bbox_inches='tight')
        QMessageBox.information(self, "Success", f"Analysis saved to {file_path}")

    def export_analysis_data(self):
        """Export the current analysis data to a file."""
        if not hasattr(self, 'full_reconstruction') or self.full_reconstruction is None:
            QMessageBox.warning(self, "Warning", "No data to export.")
            return

        # Get file name
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Analysis Data",
            os.path.join(config.results_path, "analysis_data.npz"),
            "NumPy Files (*.npz);;All Files (*)"
        )

        if not file_path:
            return

        # Export data
        np.savez(
            file_path,
            original=self.original_image_chw_numpy,
            reconstructed=self.full_reconstruction.numpy(),
            wavelengths=self.wavelengths,
            metrics=self.current_metrics,
            filename=self.file_combo.currentText()
        )

        QMessageBox.information(self, "Success", f"Analysis data exported to {file_path}")

    def select_batch_directory(self):
        """Select directory for batch processing."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Directory for Batch Processing",
            self.data_dir
        )

        if dir_path:
            self.batch_dir_path = dir_path
            # Count supported files in the directory
            tif_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.tif', '.tiff'))]
            npy_files = [f for f in os.listdir(dir_path) if f.lower().endswith('.npy')]
            QMessageBox.information(
                self, "Selected Directory",
                f"Selected directory: {dir_path}\nFound {len(tif_files)} TIF files and {len(npy_files)} NPY files"
            )

    def batch_evaluate(self):
        """Evaluate all images in the selected directory."""
        if not hasattr(self, 'batch_dir_path'):
            QMessageBox.warning(self, "Warning", "Please select a directory first.")
            return

        # Find all image files (both TIF and NPY)
        image_files = [f for f in os.listdir(self.batch_dir_path)
                     if f.lower().endswith(('.tif', '.tiff', '.npy'))]

        if not image_files:
            QMessageBox.warning(self, "Warning", "No TIF or NPY files found in the selected directory.")
            return

        # Setup progress bar
        self.batch_progress.setVisible(True)
        self.batch_progress.setMaximum(len(image_files))
        self.batch_progress.setValue(0)

        # Create results dataframe
        results = []

        # Process each file
        for i, file_name in enumerate(image_files):
            try:
                # Update progress
                self.batch_progress.setValue(i)
                QApplication.processEvents()  # Keep UI responsive

                # Load hyperspectral data based on file type
                file_path = os.path.join(self.batch_dir_path, file_name)

                if file_name.lower().endswith('.npy'):
                    # Load NPY file
                    cube = np.load(file_path)
                    # Handle both (bands, H, W) and (H, W, bands) formats
                    if cube.shape[0] <= cube.shape[1] and cube.shape[0] <= cube.shape[2]:
                        # Format is likely (bands, H, W)
                        img_data = np.transpose(cube, (1, 2, 0))
                    else:
                        # Format is likely (H, W, bands)
                        img_data = cube
                    # Add batch dimension
                    img_data = img_data.reshape((1, *img_data.shape))
                else:
                    # Load TIF file
                    with rasterio.open(file_path) as src:
                        data = src.read()
                        img_data = np.transpose(data, (1, 2, 0))
                        img_data = img_data.reshape((1, *img_data.shape))

                # Create dataset
                dataset = FullImageHyperspectralDataset(img_data)
                filtered_measurements, filter_pattern, original_spectrum = dataset[0]

                # Measure inference time
                start_time = time.time()

                # Perform reconstruction
                with torch.no_grad():
                    filtered_measurements = filtered_measurements.to(self.device).unsqueeze(0)
                    filter_pattern = filter_pattern.to(self.device).unsqueeze(0)
                    reconstructed = self.model(filtered_measurements, filter_pattern)
                    reconstructed = reconstructed.cpu().squeeze(0)

                # Calculate inference time
                inference_time = time.time() - start_time

                # Calculate metrics
                metrics = HyperspectralMetrics.compute_all_metrics(
                    reconstructed, original_spectrum
                )

                # Calculate loss components
                with torch.no_grad():
                    _, loss_components = self.model.compute_loss(
                        reconstructed.unsqueeze(0),
                        original_spectrum.unsqueeze(0),
                        torch.nn.MSELoss()
                    )

                # Store results
                result = {
                    'filename': file_name,
                    'inference_time_ms': inference_time * 1000,
                    **metrics,
                    **{f'loss_{k}': v for k, v in loss_components.items()}
                }

                results.append(result)

            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
                continue

        # Hide progress bar
        self.batch_progress.setVisible(False)

        if not results:
            QMessageBox.warning(self, "Warning", "No results were generated.")
            return

        # Create output directory
        output_dir = os.path.join(config.results_path, f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(output_dir, exist_ok=True)

        # Save results to CSV
        results_df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "batch_metrics.csv")
        results_df.to_csv(csv_path, index=False)

        # Generate summary plots
        self._generate_batch_summary_plots(results_df, output_dir)

        QMessageBox.information(
            self, "Batch Processing Complete",
            f"Processed {len(results)} files.\nResults saved to {output_dir}"
        )

    def _generate_batch_summary_plots(self, results_df, output_dir):
        """Generate summary plots for batch processing results."""
        # Create figure for key metrics
        plt.figure(figsize=(15, 10))

        # Plot PSNR
        plt.subplot(221)
        plt.hist(results_df['psnr'], bins=20, color='green', alpha=0.7)
        plt.axvline(results_df['psnr'].mean(), color='r', linestyle='--',
                   label=f'Mean: {results_df["psnr"].mean():.2f} dB')
        plt.title('PSNR Distribution')
        plt.xlabel('PSNR (dB)')
        plt.ylabel('Frequency')
        plt.legend()

        # Plot SSIM
        plt.subplot(222)
        plt.hist(results_df['ssim'], bins=20, color='blue', alpha=0.7)
        plt.axvline(results_df['ssim'].mean(), color='r', linestyle='--',
                   label=f'Mean: {results_df["ssim"].mean():.4f}')
        plt.title('SSIM Distribution')
        plt.xlabel('SSIM')
        plt.ylabel('Frequency')
        plt.legend()

        # Plot RMSE
        plt.subplot(223)
        plt.hist(results_df['rmse'], bins=20, color='red', alpha=0.7)
        plt.axvline(results_df['rmse'].mean(), color='k', linestyle='--',
                   label=f'Mean: {results_df["rmse"].mean():.4f}')
        plt.title('RMSE Distribution')
        plt.xlabel('RMSE')
        plt.ylabel('Frequency')
        plt.legend()

        # Plot inference time
        plt.subplot(224)
        plt.hist(results_df['inference_time_ms'], bins=20, color='orange', alpha=0.7)
        plt.axvline(results_df['inference_time_ms'].mean(), color='k', linestyle='--',
                   label=f'Mean: {results_df["inference_time_ms"].mean():.2f} ms')
        plt.title('Inference Time Distribution')
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "batch_metrics_summary.png"), dpi=300)
        plt.close()

        # Create additional plot for spectral smoothness metrics if available
        if 'loss_spectral_smoothness_loss' in results_df.columns:
            plt.figure(figsize=(10, 8))
            plt.hist(results_df['loss_spectral_smoothness_loss'], bins=20, color='purple', alpha=0.7)
            plt.axvline(results_df['loss_spectral_smoothness_loss'].mean(), color='k', linestyle='--',
                       label=f'Mean: {results_df["loss_spectral_smoothness_loss"].mean():.6f}')
            plt.title('Spectral Smoothness Loss Distribution')
            plt.xlabel('Smoothness Loss')
            plt.ylabel('Frequency')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "spectral_smoothness_summary.png"), dpi=300)
            plt.close()

        # Create summary table
        summary = {
            'Metric': ['PSNR', 'SSIM', 'RMSE', 'MRAE', 'Spectral Fidelity', 'Inference Time'],
            'Mean': [
                results_df['psnr'].mean(),
                results_df['ssim'].mean(),
                results_df['rmse'].mean(),
                results_df['mrae'].mean(),
                results_df['spectral_fidelity'].mean(),
                results_df['inference_time_ms'].mean()
            ],
            'Std': [
                results_df['psnr'].std(),
                results_df['ssim'].std(),
                results_df['rmse'].std(),
                results_df['mrae'].std(),
                results_df['spectral_fidelity'].std(),
                results_df['inference_time_ms'].std()
            ],
            'Min': [
                results_df['psnr'].min(),
                results_df['ssim'].min(),
                results_df['rmse'].min(),
                results_df['mrae'].min(),
                results_df['spectral_fidelity'].min(),
                results_df['inference_time_ms'].min()
            ],
            'Max': [
                results_df['psnr'].max(),
                results_df['ssim'].max(),
                results_df['rmse'].max(),
                results_df['mrae'].max(),
                results_df['spectral_fidelity'].max(),
                results_df['inference_time_ms'].max()
            ]
        }

        # Add smoothness metrics if available
        if 'loss_spectral_smoothness_loss' in results_df.columns:
            summary['Metric'].append('Spectral Smoothness Loss')
            summary['Mean'].append(results_df['loss_spectral_smoothness_loss'].mean())
            summary['Std'].append(results_df['loss_spectral_smoothness_loss'].std())
            summary['Min'].append(results_df['loss_spectral_smoothness_loss'].min())
            summary['Max'].append(results_df['loss_spectral_smoothness_loss'].max())

        if 'loss_spectral_tv_loss' in results_df.columns:
            summary['Metric'].append('Spectral TV Loss')
            summary['Mean'].append(results_df['loss_spectral_tv_loss'].mean())
            summary['Std'].append(results_df['loss_spectral_tv_loss'].std())
            summary['Min'].append(results_df['loss_spectral_tv_loss'].min())
            summary['Max'].append(results_df['loss_spectral_tv_loss'].max())

        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(output_dir, "batch_summary.csv"), index=False)


def main():
    app = QApplication(sys.argv)
    viewer = ReconstructionViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
