# modern_ui.py
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QSlider, 
                            QFileDialog, QComboBox, QGroupBox, QTabWidget,
                            QSpinBox, QDoubleSpinBox, QCheckBox, QSplitter,
                            QScrollArea, QFrame, QProgressBar, QMessageBox,
                            QToolBar, QAction, QStatusBar, QToolButton, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont, QPalette, QColor
import pyqtgraph as pg
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import numpy as np

class ModernImageEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.original_image = None
        self.image_path = None
        self.setup_ui()
        self.apply_styles()
        
    def setup_ui(self):
        self.setWindowTitle("AI Image Editor Pro")
        self.setGeometry(100, 50, 1600, 1000)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Image display
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Image display with border
        image_frame = QFrame()
        image_frame.setFrameStyle(QFrame.Box)
        image_layout = QVBoxLayout(image_frame)
        
        self.image_display = QLabel()
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setMinimumSize(800, 600)
        self.image_display.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                border: 2px solid #555;
                border-radius: 8px;
            }
        """)
        image_layout.addWidget(self.image_display)
        left_layout.addWidget(image_frame)
        
        # Image info bar
        info_frame = QFrame()
        info_layout = QHBoxLayout(info_frame)
        self.info_label = QLabel("No image loaded")
        self.info_label.setStyleSheet("color: #888; padding: 5px;")
        info_layout.addWidget(self.info_label)
        left_layout.addWidget(info_frame)
        
        # Right panel - Controls
        right_panel = QScrollArea()
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # File operations
        file_group = self.create_styled_group("File Operations")
        file_layout = QVBoxLayout(file_group)
        
        btn_open = self.create_styled_button("📁 Open Image", self.open_image)
        btn_save = self.create_styled_button("💾 Save Image", self.save_image)
        btn_reset = self.create_styled_button("🔄 Reset", self.reset_image)
        
        file_layout.addWidget(btn_open)
        file_layout.addWidget(btn_save)
        file_layout.addWidget(btn_reset)
        right_layout.addWidget(file_group)
        
        # Create tabs for different feature categories
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #555; background-color: #2b2b2b; }
            QTabBar::tab { background-color: #333; color: white; padding: 8px 16px; }
            QTabBar::tab:selected { background-color: #0078d7; }
        """)
        
        # Basic adjustments tab
        self.basic_tab = self.create_basic_tab()
        self.tabs.addTab(self.basic_tab, "🎛️ Basic")
        
        # Filters tab
        self.filters_tab = self.create_filters_tab()
        self.tabs.addTab(self.filters_tab, "🎨 Filters")
        
        # Advanced tab
        self.advanced_tab = self.create_advanced_tab()
        self.tabs.addTab(self.advanced_tab, "⚡ Advanced")
        
        # AI Tools tab
        self.ai_tab = self.create_ai_tab()
        self.tabs.addTab(self.ai_tab, "🤖 AI Tools")
        
        right_layout.addWidget(self.tabs)
        
        right_panel.setWidget(right_widget)
        right_panel.setWidgetResizable(True)
        right_panel.setMaximumWidth(450)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel, 70)
        main_layout.addWidget(right_panel, 30)
        
        # Status bar
        self.statusBar().showMessage("Ready to edit images!")
        
    def create_styled_group(self, title):
        group = QGroupBox(title)
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: white;
                border: 2px solid #555;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
                color: #0078d7;
            }
        """)
        return group
        
    def create_styled_button(self, text, callback):
        btn = QPushButton(text)
        btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d7;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-weight: bold;
                margin: 2px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
        """)
        btn.clicked.connect(callback)
        return btn
        
    def create_slider_control(self, label, min_val, max_val, default, callback):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        label_widget = QLabel(label)
        label_widget.setStyleSheet("color: white; font-weight: bold;")
        layout.addWidget(label_widget)
        
        slider_layout = QHBoxLayout()
        
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        slider.valueChanged.connect(callback)
        slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #555;
                height: 8px;
                background: #333;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0078d7;
                border: 1px solid #555;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)
        
        value_label = QLabel(str(default))
        value_label.setStyleSheet("color: #888; min-width: 30px;")
        slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
        
        slider_layout.addWidget(slider)
        slider_layout.addWidget(value_label)
        layout.addLayout(slider_layout)
        
        return widget, slider
        
    def create_basic_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Brightness
        brightness_widget, self.brightness_slider = self.create_slider_control(
            "Brightness", -100, 100, 0, self.apply_basic_adjustments)
        layout.addWidget(brightness_widget)
        
        # Contrast
        contrast_widget, self.contrast_slider = self.create_slider_control(
            "Contrast", -100, 100, 0, self.apply_basic_adjustments)
        layout.addWidget(contrast_widget)
        
        # Saturation
        saturation_widget, self.saturation_slider = self.create_slider_control(
            "Saturation", -100, 100, 0, self.apply_basic_adjustments)
        layout.addWidget(saturation_widget)
        
        # Sharpness
        sharpness_widget, self.sharpness_slider = self.create_slider_control(
            "Sharpness", -100, 100, 0, self.apply_basic_adjustments)
        layout.addWidget(sharpness_widget)
        
        layout.addStretch()
        return widget
        
    def create_filters_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Filter selection
        filter_group = self.create_styled_group("Image Filters")
        filter_layout = QVBoxLayout(filter_group)
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItems([
            "None", "Blur", "Gaussian Blur", "Sharpen", "Edge Enhance",
            "Emboss", "Find Edges", "Smooth", "Detail", "Contour"
        ])
        self.filter_combo.setStyleSheet("""
            QComboBox {
                background-color: #333;
                color: white;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
            }
        """)
        filter_layout.addWidget(self.filter_combo)
        
        btn_apply_filter = self.create_styled_button("Apply Filter", self.apply_filter)
        filter_layout.addWidget(btn_apply_filter)
        layout.addWidget(filter_group)
        
        # Color effects
        color_group = self.create_styled_group("Color Effects")
        color_layout = QVBoxLayout(color_group)
        
        self.effect_combo = QComboBox()
        self.effect_combo.addItems([
            "None", "Grayscale", "Sepia", "Invert", "Solarize",
            "Posterize", "Equalize", "Auto Contrast"
        ])
        self.effect_combo.setStyleSheet("""
            QComboBox {
                background-color: #333;
                color: white;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
            }
        """)
        color_layout.addWidget(self.effect_combo)
        
        btn_apply_effect = self.create_styled_button("Apply Effect", self.apply_color_effect)
        color_layout.addWidget(btn_apply_effect)
        layout.addWidget(color_group)
        
        layout.addStretch()
        return widget
        
    def create_advanced_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Edge detection
        edge_group = self.create_styled_group("Edge Detection")
        edge_layout = QVBoxLayout(edge_group)
        
        self.edge_combo = QComboBox()
        self.edge_combo.addItems(["Sobel", "Canny", "Laplacian", "Prewitt", "Roberts"])
        self.edge_combo.setStyleSheet("""
            QComboBox {
                background-color: #333;
                color: white;
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
            }
        """)
        edge_layout.addWidget(self.edge_combo)
        
        btn_apply_edge = self.create_styled_button("Detect Edges", self.apply_edge_detection)
        edge_layout.addWidget(btn_apply_edge)
        layout.addWidget(edge_group)
        
        # Denoising
        denoise_group = self.create_styled_group("Noise Reduction")
        denoise_layout = QVBoxLayout(denoise_group)
        
        self.denoise_combo = QComboBox()
        self.denoise_combo.addItems(["Gaussian", "Median", "Bilateral", "Wavelet"])
        denoise_layout.addWidget(self.denoise_combo)
        
        btn_apply_denoise = self.create_styled_button("Remove Noise", self.apply_denoising)
        denoise_layout.addWidget(btn_apply_denoise)
        layout.addWidget(denoise_group)
        
        layout.addStretch()
        return widget
        
    def create_ai_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # AI Enhance
        ai_group = self.create_styled_group("AI Enhancements")
        ai_layout = QVBoxLayout(ai_group)
        
        btn_enhance = self.create_styled_button("🚀 Auto Enhance", self.auto_enhance)
        btn_upscale = self.create_styled_button("🔍 Super Resolution", self.super_resolution)
        btn_style = self.create_styled_button("🎨 Style Transfer", self.style_transfer)
        
        ai_layout.addWidget(btn_enhance)
        ai_layout.addWidget(btn_upscale)
        ai_layout.addWidget(btn_style)
        layout.addWidget(ai_group)
        
        layout.addStretch()
        return widget
        
    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
                color: white;
            }
            QWidget {
                background-color: #1e1e1e;
                color: white;
            }
        """)
        
    # Add your existing image processing methods here (open_image, save_image, apply_basic_adjustments, etc.)
    # Copy all the image processing methods from your previous code

def main():
    app = QApplication(sys.argv)
    
    # Set dark theme
    app.setStyle('Fusion')
    
    editor = ModernImageEditor()
    editor.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()