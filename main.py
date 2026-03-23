# main.py
import sys
import os
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QSlider, 
                            QFileDialog, QComboBox, QGroupBox, QTabWidget,
                            QSpinBox, QDoubleSpinBox, QCheckBox, QSplitter)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
import pyqtgraph as pg
from scipy import ndimage
from skimage import filters, exposure, segmentation, restoration

class AdvancedImageEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.original_image = None
        self.image_path = None
        self.setup_ui()
        
    def setup_ui(self):
        self.setWindowTitle("Advanced Image Editor")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel - Image display
        self.image_display = QLabel()
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setStyleSheet("border: 1px solid gray;")
        self.image_display.setMinimumSize(600, 600)
        
        # Right panel - Controls
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        
        # Create tabs for different feature categories
        self.tabs = QTabWidget()
        
        # Basic adjustments tab
        self.basic_tab = self.create_basic_tab()
        self.tabs.addTab(self.basic_tab, "Basic Adjustments")
        
        # Filters tab
        self.filters_tab = self.create_filters_tab()
        self.tabs.addTab(self.filters_tab, "Filters & Effects")
        
        # Advanced tab
        self.advanced_tab = self.create_advanced_tab()
        self.tabs.addTab(self.advanced_tab, "Advanced Features")
        
        # File operations
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout(file_group)
        
        btn_open = QPushButton("Open Image")
        btn_open.clicked.connect(self.open_image)
        file_layout.addWidget(btn_open)
        
        btn_save = QPushButton("Save Image")
        btn_save.clicked.connect(self.save_image)
        file_layout.addWidget(btn_save)
        
        btn_reset = QPushButton("Reset to Original")
        btn_reset.clicked.connect(self.reset_image)
        file_layout.addWidget(btn_reset)
        
        controls_layout.addWidget(file_group)
        controls_layout.addWidget(self.tabs)
        
        splitter.addWidget(self.image_display)
        splitter.addWidget(controls_widget)
        splitter.setSizes([800, 400])
        
        main_layout.addWidget(splitter)
        
    def create_basic_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Brightness
        brightness_group = QGroupBox("Brightness")
        brightness_layout = QVBoxLayout(brightness_group)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self.apply_basic_adjustments)
        brightness_layout.addWidget(self.brightness_slider)
        layout.addWidget(brightness_group)
        
        # Contrast
        contrast_group = QGroupBox("Contrast")
        contrast_layout = QVBoxLayout(contrast_group)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(-100, 100)
        self.contrast_slider.setValue(0)
        self.contrast_slider.valueChanged.connect(self.apply_basic_adjustments)
        contrast_layout.addWidget(self.contrast_slider)
        layout.addWidget(contrast_group)
        
        # Saturation
        saturation_group = QGroupBox("Saturation")
        saturation_layout = QVBoxLayout(saturation_group)
        self.saturation_slider = QSlider(Qt.Horizontal)
        self.saturation_slider.setRange(-100, 100)
        self.saturation_slider.setValue(0)
        self.saturation_slider.valueChanged.connect(self.apply_basic_adjustments)
        saturation_layout.addWidget(self.saturation_slider)
        layout.addWidget(saturation_group)
        
        # Sharpness
        sharpness_group = QGroupBox("Sharpness")
        sharpness_layout = QVBoxLayout(sharpness_group)
        self.sharpness_slider = QSlider(Qt.Horizontal)
        self.sharpness_slider.setRange(-100, 100)
        self.sharpness_slider.setValue(0)
        self.sharpness_slider.valueChanged.connect(self.apply_basic_adjustments)
        sharpness_layout.addWidget(self.sharpness_slider)
        layout.addWidget(sharpness_group)
        
        return widget
    
    def create_filters_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Filter selection
        filter_group = QGroupBox("Image Filters")
        filter_layout = QVBoxLayout(filter_group)
        
        self.filter_combo = QComboBox()
        self.filter_combo.addItems([
            "None", "Blur", "Gaussian Blur", "Sharpen", "Edge Enhance",
            "Emboss", "Find Edges", "Smooth", "Detail", "Contour"
        ])
        filter_layout.addWidget(self.filter_combo)
        
        btn_apply_filter = QPushButton("Apply Filter")
        btn_apply_filter.clicked.connect(self.apply_filter)
        filter_layout.addWidget(btn_apply_filter)
        
        layout.addWidget(filter_group)
        
        # Color effects
        color_group = QGroupBox("Color Effects")
        color_layout = QVBoxLayout(color_group)
        
        self.effect_combo = QComboBox()
        self.effect_combo.addItems([
            "None", "Grayscale", "Sepia", "Invert", "Solarize",
            "Posterize", "Equalize", "Auto Contrast"
        ])
        color_layout.addWidget(self.effect_combo)
        
        btn_apply_effect = QPushButton("Apply Effect")
        btn_apply_effect.clicked.connect(self.apply_color_effect)
        color_layout.addWidget(btn_apply_effect)
        
        layout.addWidget(color_group)
        
        return widget
    
    def create_advanced_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Edge detection
        edge_group = QGroupBox("Edge Detection")
        edge_layout = QVBoxLayout(edge_group)
        
        self.edge_combo = QComboBox()
        self.edge_combo.addItems(["Sobel", "Canny", "Laplacian", "Prewitt", "Roberts"])
        edge_layout.addWidget(self.edge_combo)
        
        btn_apply_edge = QPushButton("Apply Edge Detection")
        btn_apply_edge.clicked.connect(self.apply_edge_detection)
        edge_layout.addWidget(btn_apply_edge)
        
        layout.addWidget(edge_group)
        
        # Denoising
        denoise_group = QGroupBox("Noise Reduction")
        denoise_layout = QVBoxLayout(denoise_group)
        
        self.denoise_combo = QComboBox()
        self.denoise_combo.addItems(["Gaussian", "Median", "Bilateral", "Wavelet"])
        denoise_layout.addWidget(self.denoise_combo)
        
        btn_apply_denoise = QPushButton("Apply Denoising")
        btn_apply_denoise.clicked.connect(self.apply_denoising)
        denoise_layout.addWidget(btn_apply_denoise)
        
        layout.addWidget(denoise_group)
        
        # Morphological operations
        morph_group = QGroupBox("Morphological Operations")
        morph_layout = QVBoxLayout(morph_group)
        
        self.morph_combo = QComboBox()
        self.morph_combo.addItems(["Erosion", "Dilation", "Opening", "Closing"])
        morph_layout.addWidget(self.morph_combo)
        
        btn_apply_morph = QPushButton("Apply Operation")
        btn_apply_morph.clicked.connect(self.apply_morphological)
        morph_layout.addWidget(btn_apply_morph)
        
        layout.addWidget(morph_group)
        
        return widget
    
    def open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", 
            "Image Files (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        
        if file_path:
            self.image_path = file_path
            self.current_image = Image.open(file_path)
            self.original_image = self.current_image.copy()
            self.display_image()
    
    def save_image(self):
        if self.current_image:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Image", "", 
                "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;TIFF (*.tiff)"
            )
            if file_path:
                self.current_image.save(file_path)
    
    def reset_image(self):
        if self.original_image:
            self.current_image = self.original_image.copy()
            self.reset_sliders()
            self.display_image()
    
    def reset_sliders(self):
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(0)
        self.saturation_slider.setValue(0)
        self.sharpness_slider.setValue(0)
    
    def display_image(self):
        if self.current_image:
            # Convert PIL Image to QPixmap
            img = self.current_image.convert("RGB")
            data = img.tobytes("raw", "RGB")
            q_image = QImage(data, img.size[0], img.size[1], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            
            # Scale pixmap to fit display while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(
                self.image_display.width(), 
                self.image_display.height(),
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            self.image_display.setPixmap(scaled_pixmap)
    
    def apply_basic_adjustments(self):
        if not self.original_image:
            return
            
        self.current_image = self.original_image.copy()
        
        # Apply brightness
        brightness_factor = 1.0 + self.brightness_slider.value() / 100.0
        enhancer = ImageEnhance.Brightness(self.current_image)
        self.current_image = enhancer.enhance(brightness_factor)
        
        # Apply contrast
        contrast_factor = 1.0 + self.contrast_slider.value() / 100.0
        enhancer = ImageEnhance.Contrast(self.current_image)
        self.current_image = enhancer.enhance(contrast_factor)
        
        # Apply saturation
        saturation_factor = 1.0 + self.saturation_slider.value() / 100.0
        enhancer = ImageEnhance.Color(self.current_image)
        self.current_image = enhancer.enhance(saturation_factor)
        
        # Apply sharpness
        sharpness_factor = 1.0 + self.sharpness_slider.value() / 100.0
        enhancer = ImageEnhance.Sharpness(self.current_image)
        self.current_image = enhancer.enhance(sharpness_factor)
        
        self.display_image()
    
    def apply_filter(self):
        if not self.current_image:
            return
            
        filter_name = self.filter_combo.currentText()
        
        if filter_name == "Blur":
            self.current_image = self.current_image.filter(ImageFilter.BLUR)
        elif filter_name == "Gaussian Blur":
            self.current_image = self.current_image.filter(ImageFilter.GaussianBlur(2))
        elif filter_name == "Sharpen":
            self.current_image = self.current_image.filter(ImageFilter.SHARPEN)
        elif filter_name == "Edge Enhance":
            self.current_image = self.current_image.filter(ImageFilter.EDGE_ENHANCE)
        elif filter_name == "Emboss":
            self.current_image = self.current_image.filter(ImageFilter.EMBOSS)
        elif filter_name == "Find Edges":
            self.current_image = self.current_image.filter(ImageFilter.FIND_EDGES)
        elif filter_name == "Smooth":
            self.current_image = self.current_image.filter(ImageFilter.SMOOTH)
        elif filter_name == "Detail":
            self.current_image = self.current_image.filter(ImageFilter.DETAIL)
        elif filter_name == "Contour":
            self.current_image = self.current_image.filter(ImageFilter.CONTOUR)
        
        self.display_image()
    
    def apply_color_effect(self):
        if not self.current_image:
            return
            
        effect_name = self.effect_combo.currentText()
        
        if effect_name == "Grayscale":
            self.current_image = ImageOps.grayscale(self.current_image)
        elif effect_name == "Sepia":
            self.current_image = self.apply_sepia()
        elif effect_name == "Invert":
            self.current_image = ImageOps.invert(self.current_image)
        elif effect_name == "Solarize":
            self.current_image = ImageOps.solarize(self.current_image, 128)
        elif effect_name == "Posterize":
            self.current_image = ImageOps.posterize(self.current_image, 4)
        elif effect_name == "Equalize":
            self.current_image = ImageOps.equalize(self.current_image)
        elif effect_name == "Auto Contrast":
            self.current_image = ImageOps.autocontrast(self.current_image)
        
        self.display_image()
    
    def apply_sepia(self):
        # Convert to sepia tone
        width, height = self.current_image.size
        pixels = self.current_image.load()
        
        for py in range(height):
            for px in range(width):
                r, g, b = self.current_image.getpixel((px, py))
                
                tr = int(0.393 * r + 0.769 * g + 0.189 * b)
                tg = int(0.349 * r + 0.686 * g + 0.168 * b)
                tb = int(0.272 * r + 0.534 * g + 0.131 * b)
                
                pixels[px, py] = (
                    min(255, tr),
                    min(255, tg),
                    min(255, tb)
                )
        
        return self.current_image
    
    def apply_edge_detection(self):
        if not self.current_image:
            return
            
        method = self.edge_combo.currentText()
        
        # Convert PIL to numpy array for OpenCV processing
        img_array = np.array(self.current_image.convert('L'))  # Convert to grayscale
        
        if method == "Sobel":
            sobelx = cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(img_array, cv2.CV_64F, 0, 1, ksize=5)
            edges = np.sqrt(sobelx**2 + sobely**2)
        elif method == "Canny":
            edges = cv2.Canny(img_array, 100, 200)
        elif method == "Laplacian":
            edges = cv2.Laplacian(img_array, cv2.CV_64F)
        elif method == "Prewitt":
            kernelx = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
            kernely = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            prewittx = ndimage.convolve(img_array, kernelx)
            prewitty = ndimage.convolve(img_array, kernely)
            edges = np.sqrt(prewittx**2 + prewitty**2)
        elif method == "Roberts":
            edges = filters.roberts(img_array)
        
        # Normalize and convert back to PIL
        edges = (edges / edges.max() * 255).astype(np.uint8)
        self.current_image = Image.fromarray(edges)
        self.display_image()
    
    def apply_denoising(self):
        if not self.current_image:
            return
            
        method = self.denoise_combo.currentText()
        img_array = np.array(self.current_image)
        
        if method == "Gaussian":
            denoised = cv2.GaussianBlur(img_array, (5, 5), 0)
        elif method == "Median":
            denoised = cv2.medianBlur(img_array, 5)
        elif method == "Bilateral":
            denoised = cv2.bilateralFilter(img_array, 9, 75, 75)
        elif method == "Wavelet":
            denoised = restoration.denoise_wavelet(img_array, channel_axis=-1)
            denoised = (denoised * 255).astype(np.uint8)
        
        self.current_image = Image.fromarray(denoised)
        self.display_image()
    
    def apply_morphological(self):
        if not self.current_image:
            return
            
        operation = self.morph_combo.currentText()
        img_array = np.array(self.current_image.convert('L'))  # Convert to grayscale
        
        kernel = np.ones((5, 5), np.uint8)
        
        if operation == "Erosion":
            result = cv2.erode(img_array, kernel, iterations=1)
        elif operation == "Dilation":
            result = cv2.dilate(img_array, kernel, iterations=1)
        elif operation == "Opening":
            result = cv2.morphologyEx(img_array, cv2.MORPH_OPEN, kernel)
        elif operation == "Closing":
            result = cv2.morphologyEx(img_array, cv2.MORPH_CLOSE, kernel)
        
        self.current_image = Image.fromarray(result)
        self.display_image()

def main():
    app = QApplication(sys.argv)
    editor = AdvancedImageEditor()
    editor.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()