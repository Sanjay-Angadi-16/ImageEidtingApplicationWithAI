# modern_editor_ai_fixed.py
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
from scipy import ndimage
from skimage import filters, exposure, segmentation

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
        
        # Toolbar
        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        
        btn_open = self.create_styled_button("📁 Open Image", self.open_image)
        btn_save = self.create_styled_button("💾 Save Image", self.save_image)
        btn_reset = self.create_styled_button("🔄 Reset", self.reset_image)
        
        toolbar_layout.addWidget(btn_open)
        toolbar_layout.addWidget(btn_save)
        toolbar_layout.addWidget(btn_reset)
        toolbar_layout.addStretch()
        
        left_layout.addWidget(toolbar)
        
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
                color: #888;
                font-size: 16px;
            }
        """)
        self.image_display.setText("No image loaded\n\nDrag and drop an image or click 'Open Image'")
        image_layout.addWidget(self.image_display)
        left_layout.addWidget(image_frame)
        
        # Image info bar
        info_frame = QFrame()
        info_layout = QHBoxLayout(info_frame)
        self.info_label = QLabel("No image loaded")
        self.info_label.setStyleSheet("color: #888; padding: 5px; font-size: 12px;")
        info_layout.addWidget(self.info_label)
        left_layout.addWidget(info_frame)
        
        # Right panel - Controls
        right_panel = QScrollArea()
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Create tabs for different feature categories
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { 
                border: 1px solid #555; 
                background-color: #2b2b2b; 
                border-radius: 8px;
            }
            QTabBar::tab { 
                background-color: #333; 
                color: white; 
                padding: 12px 20px; 
                border-top-left-radius: 8px;
                border-top-right-radius: 8px;
                margin-right: 2px;
                font-weight: bold;
            }
            QTabBar::tab:selected { 
                background-color: #0078d7; 
            }
            QTabBar::tab:hover {
                background-color: #555;
            }
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
        right_panel.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #1e1e1e;
            }
        """)
        
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
                padding-top: 15px;
                background-color: #2b2b2b;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
                color: #0078d7;
                font-size: 14px;
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
                padding: 12px 16px;
                border-radius: 6px;
                font-weight: bold;
                margin: 2px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #555;
                color: #888;
            }
        """)
        btn.clicked.connect(callback)
        return btn
        
    def create_slider_control(self, label, min_val, max_val, default, callback):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(5)
        
        label_widget = QLabel(label)
        label_widget.setStyleSheet("color: white; font-weight: bold; font-size: 12px;")
        layout.addWidget(label_widget)
        
        slider_layout = QHBoxLayout()
        
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)
        slider.valueChanged.connect(callback)
        slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #555;
                height: 6px;
                background: #333;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #0078d7;
                border: 1px solid #555;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #106ebe;
            }
        """)
        
        value_label = QLabel(str(default))
        value_label.setStyleSheet("color: #888; min-width: 40px; font-size: 12px;")
        slider.valueChanged.connect(lambda v: value_label.setText(str(v)))
        
        slider_layout.addWidget(slider)
        slider_layout.addWidget(value_label)
        layout.addLayout(slider_layout)
        
        return widget, slider
        
    def create_basic_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
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
        
        # Rotation
        rotation_widget, self.rotation_slider = self.create_slider_control(
            "Rotation", -180, 180, 0, self.apply_rotation)
        layout.addWidget(rotation_widget)
        
        layout.addStretch()
        return widget
        
    def create_filters_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
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
                padding: 8px;
                font-size: 12px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #333;
                color: white;
                selection-background-color: #0078d7;
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
                padding: 8px;
                font-size: 12px;
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
        layout.setSpacing(15)
        
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
                padding: 8px;
                font-size: 12px;
            }
        """)
        edge_layout.addWidget(self.edge_combo)
        
        btn_apply_edge = self.create_styled_button("Detect Edges", self.apply_edge_detection)
        edge_layout.addWidget(btn_apply_edge)
        layout.addWidget(edge_group)
        
        # Denoising (without wavelet option)
        denoise_group = self.create_styled_group("Noise Reduction")
        denoise_layout = QVBoxLayout(denoise_group)
        
        self.denoise_combo = QComboBox()
        self.denoise_combo.addItems(["Gaussian", "Median", "Bilateral", "NL-Means"])
        denoise_layout.addWidget(self.denoise_combo)
        
        btn_apply_denoise = self.create_styled_button("Remove Noise", self.apply_denoising)
        denoise_layout.addWidget(btn_apply_denoise)
        layout.addWidget(denoise_group)
        
        # Morphological operations
        morph_group = self.create_styled_group("Morphological Operations")
        morph_layout = QVBoxLayout(morph_group)
        
        self.morph_combo = QComboBox()
        self.morph_combo.addItems(["Erosion", "Dilation", "Opening", "Closing"])
        morph_layout.addWidget(self.morph_combo)
        
        btn_apply_morph = self.create_styled_button("Apply Operation", self.apply_morphological)
        morph_layout.addWidget(btn_apply_morph)
        layout.addWidget(morph_group)
        
        layout.addStretch()
        return widget
        
    def create_ai_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(15)
        
        # AI Enhance
        ai_group = self.create_styled_group("AI Enhancements")
        ai_layout = QVBoxLayout(ai_group)
        
        btn_enhance = self.create_styled_button("🚀 Auto Enhance", self.auto_enhance)
        btn_upscale = self.create_styled_button("🔍 Super Resolution", self.super_resolution)
        btn_style = self.create_styled_button("🎨 Style Transfer", self.style_transfer)
        btn_face_blur = self.create_styled_button("😊 Blur Faces", self.blur_faces)
        btn_cartoon = self.create_styled_button("✏️ Cartoon Effect", self.cartoon_effect)
        btn_background_remove = self.create_styled_button("🧹 Remove Background", self.remove_background)
        
        ai_layout.addWidget(btn_enhance)
        ai_layout.addWidget(btn_upscale)
        ai_layout.addWidget(btn_style)
        ai_layout.addWidget(btn_face_blur)
        ai_layout.addWidget(btn_cartoon)
        ai_layout.addWidget(btn_background_remove)
        
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
            QStatusBar {
                background-color: #2b2b2b;
                color: #888;
            }
        """)
    
    # Image Processing Methods (keep all the existing methods from previous code)
    # ... [Include all the existing image processing methods here] ...
    
    # WORKING AI METHODS:
    
    def auto_enhance(self):
        """Advanced auto-enhancement using multiple techniques"""
        if not self.current_image:
            QMessageBox.warning(self, "Warning", "Please load an image first!")
            return
        
        try:
            img_array = np.array(self.current_image)
            
            # Multiple enhancement techniques
            if len(img_array.shape) == 3:  # Color image
                # Convert to LAB color space for better enhancement
                lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                # CLAHE (Contrast Limited Adaptive Histogram Equalization)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                
                # Merge back and convert to RGB
                lab = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
                
                # Add slight sharpening
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                enhanced = cv2.filter2D(enhanced, -1, kernel)
            else:  # Grayscale
                enhanced = cv2.equalizeHist(img_array)
                enhanced = cv2.GaussianBlur(enhanced, (0,0), 1.0)
                enhanced = cv2.addWeighted(img_array, 1.5, enhanced, -0.5, 0)
            
            self.current_image = Image.fromarray(enhanced)
            self.display_image()
            self.statusBar().showMessage("Applied advanced auto-enhancement")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply auto-enhance: {str(e)}")
    
    def super_resolution(self):
        """Simple super-resolution using interpolation"""
        if not self.current_image:
            QMessageBox.warning(self, "Warning", "Please load an image first!")
            return
        
        try:
            # Get user choice for scale factor
            scale, ok = QInputDialog.getInt(self, "Super Resolution", 
                                          "Enter scale factor (2-4):", 2, 2, 4, 1)
            if not ok:
                return
            
            img_array = np.array(self.current_image)
            
            # Use different interpolation methods for better results
            if len(img_array.shape) == 3:
                # For color images, use Lanczos interpolation
                new_width = int(img_array.shape[1] * scale)
                new_height = int(img_array.shape[0] * scale)
                upscaled = cv2.resize(img_array, (new_width, new_height), 
                                    interpolation=cv2.INTER_LANCZOS4)
            else:
                # For grayscale
                new_width = int(img_array.shape[1] * scale)
                new_height = int(img_array.shape[0] * scale)
                upscaled = cv2.resize(img_array, (new_width, new_height), 
                                    interpolation=cv2.INTER_CUBIC)
            
            # Apply slight sharpening to enhance details
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            upscaled = cv2.filter2D(upscaled, -1, kernel)
            
            self.current_image = Image.fromarray(upscaled)
            self.display_image()
            self.statusBar().showMessage(f"Upscaled image by {scale}x")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply super resolution: {str(e)}")
    
    def style_transfer(self):
        """Simple style transfer using color manipulation"""
        if not self.current_image:
            QMessageBox.warning(self, "Warning", "Please load an image first!")
            return
        
        try:
            styles = ["Warm", "Cool", "Vintage", "Cinematic", "Pop Art"]
            style, ok = QInputDialog.getItem(self, "Style Transfer", 
                                           "Choose a style:", styles, 0, False)
            if not ok:
                return
            
            img_array = np.array(self.current_image)
            
            if style == "Warm":
                # Add warm tones (more red/yellow)
                img_array = img_array.astype(np.float32)
                img_array[:,:,0] = np.clip(img_array[:,:,0] * 1.2, 0, 255)  # Red
                img_array[:,:,1] = np.clip(img_array[:,:,1] * 1.1, 0, 255)  # Green
            elif style == "Cool":
                # Add cool tones (more blue)
                img_array = img_array.astype(np.float32)
                img_array[:,:,2] = np.clip(img_array[:,:,2] * 1.2, 0, 255)  # Blue
            elif style == "Vintage":
                # Sepia-like vintage effect
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                img_array = cv2.applyColorMap(img_array, cv2.COLORMAP_SEPIA)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            elif style == "Cinematic":
                # Increase contrast and add blue tint
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(img_array)
                clahe = cv2.createCLAHE(clipLimit=3.0)
                l = clahe.apply(l)
                b = np.clip(b + 10, 0, 255)  # Add blue to LAB
                img_array = cv2.merge([l, a, b])
                img_array = cv2.cvtColor(img_array, cv2.COLOR_LAB2RGB)
            elif style == "Pop Art":
                # Posterize and enhance colors
                img_pil = Image.fromarray(img_array)
                img_pil = ImageOps.posterize(img_pil, 4)
                enhancer = ImageEnhance.Color(img_pil)
                img_pil = enhancer.enhance(1.5)
                img_array = np.array(img_pil)
            
            self.current_image = Image.fromarray(img_array.astype(np.uint8))
            self.display_image()
            self.statusBar().showMessage(f"Applied {style} style")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply style transfer: {str(e)}")
    
    def blur_faces(self):
        """Advanced face detection and blurring"""
        if not self.current_image:
            QMessageBox.warning(self, "Warning", "Please load an image first!")
            return
        
        try:
            img_array = np.array(self.current_image)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Load multiple face detectors for better accuracy
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            
            faces1 = face_cascade.detectMultiScale(gray, 1.1, 4)
            faces2 = profile_cascade.detectMultiScale(gray, 1.1, 4)
            
            all_faces = list(faces1) + list(faces2)
            
            if len(all_faces) == 0:
                QMessageBox.information(self, "Info", "No faces detected in the image")
                return
            
            for (x, y, w, h) in all_faces:
                # Expand the region slightly for better blurring
                x = max(0, x - 10)
                y = max(0, y - 10)
                w = min(w + 20, img_array.shape[1] - x)
                h = min(h + 20, img_array.shape[0] - y)
                
                face_roi = img_array[y:y+h, x:x+w]
                # Use different blur strength based on face size
                blur_strength = max(25, min(99, w // 4))
                if blur_strength % 2 == 0:  # Ensure odd number for Gaussian blur
                    blur_strength += 1
                blurred_face = cv2.GaussianBlur(face_roi, (blur_strength, blur_strength), 0)
                img_array[y:y+h, x:x+w] = blurred_face
            
            self.current_image = Image.fromarray(img_array)
            self.display_image()
            self.statusBar().showMessage(f"Blurred {len(all_faces)} faces")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to blur faces: {str(e)}")
    
    def cartoon_effect(self):
        """Create cartoon-like effect"""
        if not self.current_image:
            QMessageBox.warning(self, "Warning", "Please load an image first!")
            return
        
        try:
            img_array = np.array(self.current_image)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            # Apply bilateral filter to reduce noise while keeping edges sharp
            filtered = cv2.bilateralFilter(img_array, 9, 75, 75)
            
            # Detect edges
            edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                        cv2.THRESH_BINARY, 9, 2)
            
            # Convert edges to 3-channel
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            
            # Combine filtered image with edges
            cartoon = cv2.bitwise_and(filtered, edges)
            
            self.current_image = Image.fromarray(cartoon)
            self.display_image()
            self.statusBar().showMessage("Applied cartoon effect")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply cartoon effect: {str(e)}")
    
    def remove_background(self):
        """Simple background removal using color-based segmentation"""
        if not self.current_image:
            QMessageBox.warning(self, "Warning", "Please load an image first!")
            return
        
        try:
            img_array = np.array(self.current_image)
            
            # Convert to HSV for better color segmentation
            hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
            
            # Create a mask based on color ranges (this is a simple approach)
            # You might need to adjust these values based on your image
            lower_bound = np.array([0, 0, 0])
            upper_bound = np.array([180, 255, 200])
            
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            mask = cv2.bitwise_not(mask)
            
            # Apply morphological operations to clean up the mask
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Create transparent background
            rgba = cv2.cvtColor(img_array, cv2.COLOR_RGB2RGBA)
            rgba[:, :, 3] = mask
            
            self.current_image = Image.fromarray(rgba)
            self.display_image()
            self.statusBar().showMessage("Applied background removal (simple)")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to remove background: {str(e)}")

# Add the missing import
from PyQt5.QtWidgets import QInputDialog

def main():
    app = QApplication(sys.argv)
    
    # Set dark theme
    app.setStyle('Fusion')
    
    editor = ModernImageEditor()
    editor.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()