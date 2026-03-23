# advanced_features.py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from scipy import ndimage

class AdvancedImageProcessor:
    def __init__(self):
        pass
    
    @staticmethod
    def apply_histogram_equalization(image):
        """Apply histogram equalization for contrast enhancement"""
        img_array = np.array(image)
        
        if len(img_array.shape) == 3:  # Color image
            # Convert to YUV and equalize Y channel
            img_yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        else:  # Grayscale
            equalized = cv2.equalizeHist(img_array)
        
        return Image.fromarray(equalized)
    
    @staticmethod
    def apply_unsharp_mask(image, strength=1.0, radius=2.0):
        """Apply unsharp mask for sharpening"""
        img_array = np.array(image)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(img_array, (0, 0), radius)
        
        # Calculate sharpened image
        sharpened = cv2.addWeighted(img_array, 1.0 + strength, blurred, -strength, 0)
        
        return Image.fromarray(np.clip(sharpened, 0, 255).astype(np.uint8))
    
    @staticmethod
    def apply_perspective_transform(image, points):
        """Apply perspective transformation"""
        width, height = image.size
        
        # Source points (corners of original image)
        src_points = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        # Destination points
        dst_points = np.array(points, dtype=np.float32)
        
        # Calculate perspective transform matrix
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply perspective transform
        img_array = np.array(image)
        transformed = cv2.warpPerspective(img_array, matrix, (width, height))
        
        return Image.fromarray(transformed)
    
    @staticmethod
    def detect_and_blur_faces(image, blur_strength=15):
        """Detect faces and apply blur"""
        # Load face detection classifier
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Apply blur to detected faces
        for (x, y, w, h) in faces:
            face_roi = img_array[y:y+h, x:x+w]
            blurred_face = cv2.GaussianBlur(face_roi, (blur_strength, blur_strength), 0)
            img_array[y:y+h, x:x+w] = blurred_face
        
        return Image.fromarray(img_array)
    
    @staticmethod
    def create_watermark(image, text, position=(10, 10), opacity=0.5):
        """Add text watermark to image"""
        watermarked = image.copy()
        draw = ImageDraw.Draw(watermarked)
        
        try:
            font = ImageFont.truetype("arial.ttf", 36)
        except:
            font = ImageFont.load_default()
        
        # Create semi-transparent text
        from PIL import ImageFont
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Create temporary image for text with alpha channel
        text_img = Image.new('RGBA', (text_width, text_height), (0, 0, 0, 0))
        text_draw = ImageDraw.Draw(text_img)
        text_draw.text((0, 0), text, font=font, fill=(255, 255, 255, int(255 * opacity)))
        
        # Paste text onto main image
        watermarked.paste(text_img, position, text_img)
        
        return watermarked