"""
Textile Quality Assessment Application
======================================
Automatic quality assessment of textile (knitted) products based on 
image and real-time camera analysis.

Technologies: Python, OpenCV, Tkinter, NumPy, Pillow
Author: Diploma Project
"""

import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
import os

from PIL import Image

# PNG dan ICO ga konvertatsiya
img = Image.open('icon.png')
img.save('icon.ico', format='ICO', sizes=[(256,256), (128,128), (64,64), (32,32), (16,16)])


class TextileQualityAssessment:
    """
    Main application class for textile quality assessment.
    Provides GUI interface for image upload and real-time webcam analysis.
    """
    
    def __init__(self, root):
        """
        Initialize the application GUI and variables.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title("To'qimachilik Mahsulotlari Sifatini Baholash Tizimi")
        self.root.geometry("1200x800")
        
        # Soft UI Color Palette
        self.COLOR_BG_MAIN = '#f5f7fa'          # Soft light gray background
        self.COLOR_PANEL = '#ffffff'            # White panel background
        self.COLOR_BUTTON_BLUE = '#6c9bcf'      # Soft blue
        self.COLOR_BUTTON_GREEN = '#7fb069'     # Soft green
        self.COLOR_BUTTON_RED = '#d67b7b'       # Soft red
        self.COLOR_TEXT_PRIMARY = '#4a5568'     # Soft dark gray
        self.COLOR_TEXT_SECONDARY = '#718096'   # Medium gray
        self.COLOR_BORDER = '#e2e8f0'           # Soft border
        self.COLOR_STATUS = '#4a5568'           # Status bar color
        
        self.root.configure(bg=self.COLOR_BG_MAIN)
        
        # Variables for image processing
        self.current_image = None
        self.processed_image = None
        self.cap = None  # Video capture object
        self.is_camera_running = False
        self.camera_thread = None
        self.is_analyzing = False  # Track analysis state for UX improvements
        
        # Quality thresholds (adjustable based on testing)
        # Yumshatilgan threshold qiymatlar - internetdan olingan rasmlar uchun
        self.COLOR_VARIANCE_THRESHOLD_GOOD = 1500     # Low variance = uniform color = good (3x yumshatildi)
        self.COLOR_VARIANCE_THRESHOLD_MEDIUM = 3000   # Medium variance (2x yumshatildi)
        self.DEFECT_AREA_THRESHOLD_SMALL = 0.03       # 3% of image area (3x yumshatildi)
        self.DEFECT_AREA_THRESHOLD_MEDIUM = 0.10      # 10% of image area (2x yumshatildi)
        
        # Store button references for state management
        self.upload_btn = None
        self.camera_btn = None
        self.stop_camera_btn = None
        
        self.setup_gui()
    
    def setup_gui(self):
        """
        Create and arrange all GUI components with Soft UI styling.
        """
        # Title with Soft UI styling
        title_label = tk.Label(
            self.root, 
            text="To'qimachilik Mahsulotlari Sifatini Baholash Tizimi",
            font=("Arial", 18, "bold"),
            bg=self.COLOR_BG_MAIN,
            fg=self.COLOR_TEXT_PRIMARY
        )
        title_label.pack(pady=15)
        
        # Subtitle
        subtitle_label = tk.Label(
            self.root,
            text="Rasm va Real-vaqtli kamera tahlili asosida avtomatik sifat baholash",
            font=("Arial", 11),
            bg=self.COLOR_BG_MAIN,
            fg=self.COLOR_TEXT_SECONDARY
        )
        subtitle_label.pack(pady=5)
        
        # Main container frame with Soft UI padding
        main_frame = tk.Frame(self.root, bg=self.COLOR_BG_MAIN)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=25, pady=15)
        
        # Left panel - Controls with Soft UI styling (elevated panel effect)
        # Outer frame for shadow effect
        control_outer = tk.Frame(main_frame, bg=self.COLOR_BG_MAIN)
        control_outer.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 15))
        control_outer.config(width=320)
        
        control_frame = tk.Frame(
            control_outer,
            bg=self.COLOR_PANEL,
            relief=tk.FLAT,
            bd=0
        )
        control_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Add inner padding for Soft UI effect
        control_inner = tk.Frame(control_frame, bg=self.COLOR_PANEL)
        control_inner.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Control panel title
        tk.Label(
            control_inner,
            text="Boshqaruv Paneli",
            font=("Arial", 14, "bold"),
            bg=self.COLOR_PANEL,
            fg=self.COLOR_TEXT_PRIMARY
        ).pack(pady=(0, 20))
        
        # Upload image button with Soft UI styling
        self.upload_btn = tk.Button(
            control_inner,
            text="📁 Rasm Yuklash",
            font=("Arial", 12),
            bg=self.COLOR_BUTTON_BLUE,
            fg='white',
            padx=25,
            pady=12,
            command=self.upload_image,
            cursor='hand2',
            relief=tk.FLAT,
            bd=0,
            activebackground='#5a8bb8',
            activeforeground='white'
        )
        self.upload_btn.pack(pady=8, padx=15, fill=tk.X)
        
        # Start camera button
        self.camera_btn = tk.Button(
            control_inner,
            text="📷 Kamerani Ishga Tushirish",
            font=("Arial", 12),
            bg=self.COLOR_BUTTON_GREEN,
            fg='white',
            padx=25,
            pady=12,
            command=self.toggle_camera,
            cursor='hand2',
            relief=tk.FLAT,
            bd=0,
            activebackground='#6a9a5a',
            activeforeground='white'
        )
        self.camera_btn.pack(pady=8, padx=15, fill=tk.X)
        
        # Stop camera button (initially disabled)
        self.stop_camera_btn = tk.Button(
            control_inner,
            text="⏹ Kamerani To'xtatish",
            font=("Arial", 12),
            bg=self.COLOR_BUTTON_RED,
            fg='white',
            padx=25,
            pady=12,
            command=self.toggle_camera,
            state=tk.DISABLED,
            cursor='hand2',
            relief=tk.FLAT,
            bd=0,
            activebackground='#c06a6a',
            activeforeground='white',
            disabledforeground='#cccccc'
            # Removed disabledbackground - not supported on Windows Tkinter
        )
        self.stop_camera_btn.pack(pady=8, padx=15, fill=tk.X)
        
        # Separator with Soft UI styling
        separator = tk.Frame(control_inner, height=1, bg=self.COLOR_BORDER)
        separator.pack(fill=tk.X, padx=15, pady=20)
        
        # Results section
        tk.Label(
            control_inner,
            text="Sifat Baholash Natijalari",
            font=("Arial", 12, "bold"),
            bg=self.COLOR_PANEL,
            fg=self.COLOR_TEXT_PRIMARY
        ).pack(pady=(0, 15))
        
        # Quality result label
        self.result_label = tk.Label(
            control_inner,
            text="Hali tahlil qilinmadi",
            font=("Arial", 16, "bold"),
            bg=self.COLOR_PANEL,
            fg=self.COLOR_TEXT_SECONDARY,
            wraplength=250
        )
        self.result_label.pack(pady=10, padx=10)
        
        # Explanation text area
        tk.Label(
            control_inner,
            text="Tushuntirish:",
            font=("Arial", 10, "bold"),
            bg=self.COLOR_PANEL,
            fg=self.COLOR_TEXT_PRIMARY
        ).pack(pady=(20, 8), anchor=tk.W, padx=10)
        
        self.explanation_text = tk.Text(
            control_inner,
            height=8,
            width=30,
            font=("Arial", 9),
            wrap=tk.WORD,
            bg='#fafafa',
            relief=tk.FLAT,
            bd=1,
            fg=self.COLOR_TEXT_PRIMARY,
            highlightthickness=1,
            highlightbackground=self.COLOR_BORDER,
            highlightcolor=self.COLOR_BUTTON_BLUE
        )
        self.explanation_text.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        
        # Right panel - Image display with Soft UI styling
        display_outer = tk.Frame(main_frame, bg=self.COLOR_BG_MAIN)
        display_outer.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        display_frame = tk.Frame(
            display_outer,
            bg=self.COLOR_PANEL,
            relief=tk.FLAT,
            bd=0
        )
        display_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        display_inner = tk.Frame(display_frame, bg=self.COLOR_PANEL)
        display_inner.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        tk.Label(
            display_inner,
            text="Rasm Ko'rsatish",
            font=("Arial", 14, "bold"),
            bg=self.COLOR_PANEL,
            fg=self.COLOR_TEXT_PRIMARY
        ).pack(pady=(0, 15))
        
        # Canvas for image display with Soft UI border
        self.canvas = tk.Canvas(
            display_inner,
            bg='#fafafa',
            width=800,
            height=600,
            highlightthickness=1,
            highlightbackground=self.COLOR_BORDER,
            relief=tk.FLAT
        )
        self.canvas.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        
        # Status bar with Soft UI styling
        self.status_label = tk.Label(
            self.root,
            text="Tayyor - Tahlilni boshlash uchun rasm yuklang yoki kamerani ishga tushiring",
            font=("Arial", 9),
            bg=self.COLOR_STATUS,
            fg='white',
            anchor=tk.W,
            padx=15,
            pady=8
        )
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def upload_image(self):
        """
        Open file dialog to select and load an image file.
        Then process and analyze the image.
        """
        # Prevent multiple clicks during analysis
        if self.is_analyzing:
            return
        
        # Stop camera if running
        if self.is_camera_running:
            self.toggle_camera()
        
        # Open file dialog with Uzbek title
        file_path = filedialog.askopenfilename(
            title="Rasm Faylini Tanlash",
            filetypes=[
                ("Rasm fayllari", "*.jpg *.jpeg *.png *.bmp"),
                ("JPEG fayllari", "*.jpg *.jpeg"),
                ("PNG fayllari", "*.png"),
                ("Barcha fayllar", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Disable buttons during loading
                self.set_buttons_state(tk.DISABLED)
                self.status_label.config(text="Rasm yuklanmoqda...")
                self.root.update()
                
                # Read image using OpenCV
                self.current_image = cv2.imread(file_path)
                
                if self.current_image is None:
                    messagebox.showerror(
                        "Xatolik", 
                        "Rasm yuklanmadi. Iltimos, to'g'ri rasm faylini tanlang."
                    )
                    self.set_buttons_state(tk.NORMAL)
                    return
                
                # Update status
                self.status_label.config(text=f"Yuklandi: {os.path.basename(file_path)}")
                
                # Process and analyze the image
                self.analyze_image(self.current_image)
                
            except Exception as e:
                messagebox.showerror("Xatolik", f"Rasm yuklashda xatolik: {str(e)}")
                self.set_buttons_state(tk.NORMAL)
                self.status_label.config(text="Xatolik yuz berdi")
    
    def toggle_camera(self):
        """
        Start or stop the webcam capture and real-time analysis.
        """
        if not self.is_camera_running:
            # Start camera
            self.status_label.config(text="Kamera ochilmoqda...")
            self.root.update()
            
            self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                messagebox.showerror(
                    "Xatolik", 
                    "Kamera ochilmadi. Iltimos, kameraning ulanganligini tekshiring."
                )
                self.status_label.config(text="Kamera ochilmadi")
                return
            
            self.is_camera_running = True
            self.camera_btn.config(state=tk.DISABLED)
            self.stop_camera_btn.config(state=tk.NORMAL)
            self.upload_btn.config(state=tk.DISABLED)
            self.status_label.config(text="Kamera ishga tushdi - Real-vaqtda tahlil qilinmoqda...")
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
            self.camera_thread.start()
            
        else:
            # Stop camera
            self.is_camera_running = False
            if self.cap:
                self.cap.release()
            self.cap = None
            self.camera_btn.config(state=tk.NORMAL)
            self.stop_camera_btn.config(state=tk.DISABLED)
            self.upload_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Kamera to'xtatildi")
    
    def set_buttons_state(self, state):
        """
        Set the state of all control buttons.
        
        Args:
            state: tk.NORMAL or tk.DISABLED
        """
        if self.upload_btn:
            self.upload_btn.config(state=state)
        if self.camera_btn:
            self.camera_btn.config(state=state)
        # Don't change stop_camera_btn state here as it's controlled by camera state
    
    
    def camera_loop(self):
        """
        Main loop for capturing and processing camera frames in real-time.
        Runs in a separate thread to avoid blocking the GUI.
        """
        frame_count = 0
        quality_history = []
        
        while self.is_camera_running and self.cap:
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Analyze every frame (or every nth frame for performance)
            if frame_count % 1 == 0:  # Analyze every frame
                # Process and analyze the frame
                self.analyze_image(frame)
                
                # Store quality for averaging (optional)
                # This can be used to calculate average quality over time
            
            # Small delay to control frame rate
            cv2.waitKey(1)
    
    def preprocess_image(self, image):
        """
        Preprocess the input image for quality analysis.
        
        Steps:
        1. Resize to standard size for consistent analysis
        2. Apply Gaussian blur to reduce noise
        3. Convert to grayscale for intensity analysis
        4. Convert to HSV for color analysis
        
        Args:
            image: Input BGR image (OpenCV format)
            
        Returns:
            dict: Dictionary containing processed images
        """
        # Step 1: Resize image to standard size (800x600) for consistent analysis
        # This ensures that analysis is not affected by image resolution
        height, width = image.shape[:2]
        max_dimension = 800
        if width > max_dimension or height > max_dimension:
            scale = max_dimension / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            resized = image.copy()
        
        # Step 2: Apply Gaussian blur to reduce noise and smooth the image
        # Kernel size (5,5) and sigma=0 means automatic calculation
        # This helps in removing small artifacts and noise
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        
        # Step 3: Convert to grayscale for intensity-based analysis
        # Grayscale helps in detecting brightness variations and defects
        grayscale = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        
        # Step 4: Convert to HSV color space for color analysis
        # HSV separates color information (Hue, Saturation, Value)
        # This is better for color uniformity analysis than RGB
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        return {
            'original': resized,
            'blurred': blurred,
            'grayscale': grayscale,
            'hsv': hsv
        }
    
    def preprocess_image_enhanced(self, image):
        """
        Yaxshilangan preprocessing - illumination correction, CLAHE, bilateral filter.
        
        Steps:
        1. Resize to standard size
        2. Illumination correction (LAB color space)
        3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        4. Bilateral filter (noise reduction, edges saqlaydi)
        5. Gaussian blur
        6. Convert to grayscale and HSV
        
        Args:
            image: Input BGR image (OpenCV format)
            
        Returns:
            dict: Dictionary containing processed images
        """
        # Step 1: Resize image
        height, width = image.shape[:2]
        max_dimension = 800
        if width > max_dimension or height > max_dimension:
            scale = max_dimension / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            resized = image.copy()
        
        # Step 2: Illumination correction using LAB color space
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Bu yorug'likni tekislaydi va kontrastni yaxshilaydi
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_channel = clahe.apply(l_channel)
        
        # LAB dan BGR ga qaytarish
        corrected = cv2.merge([l_channel, a, b])
        corrected = cv2.cvtColor(corrected, cv2.COLOR_LAB2BGR)
        
        # Step 3: Bilateral filter - noise reduction, lekin edges saqlaydi
        # Bu Gaussian blur dan yaxshiroq, chunki edges ni saqlaydi
        denoised = cv2.bilateralFilter(corrected, 9, 75, 75)
        
        # Step 4: Gaussian blur (qo'shimcha smoothing)
        blurred = cv2.GaussianBlur(denoised, (5, 5), 0)
        
        # Step 5: Convert to grayscale and HSV
        grayscale = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        return {
            'original': resized,
            'corrected': corrected,
            'denoised': denoised,
            'blurred': blurred,
            'grayscale': grayscale,
            'hsv': hsv
        }
    
    def analyze_color_uniformity(self, hsv_image, grayscale_image):
        """
        Analyze color uniformity of the textile using variance and standard deviation.
        
        Method:
        - Calculate variance of pixel intensities in grayscale and HSV channels
        - Low variance indicates uniform color (good quality)
        - High variance indicates color variations (potential defects)
        
        Args:
            hsv_image: HSV color space image
            grayscale_image: Grayscale image
            
        Returns:
            dict: Color uniformity metrics
        """
        # Extract Value channel from HSV (brightness/intensity)
        value_channel = hsv_image[:, :, 2]
        
        # Calculate variance and standard deviation for grayscale
        # Variance measures how spread out pixel intensities are
        gray_variance = np.var(grayscale_image)
        gray_std = np.std(grayscale_image)
        
        # Calculate variance and standard deviation for HSV Value channel
        value_variance = np.var(value_channel)
        value_std = np.std(value_channel)
        
        # Combined variance (average of both)
        combined_variance = (gray_variance + value_variance) / 2
        
        return {
            'gray_variance': gray_variance,
            'gray_std': gray_std,
            'value_variance': value_variance,
            'value_std': value_std,
            'combined_variance': combined_variance
        }
    
    def analyze_color_uniformity_enhanced(self, hsv_image, grayscale_image):
        """
        Yaxshilangan rang bir xilligi tahlili.
        
        Qo'shimcha metodlar:
        - Histogram analysis (std va entropy)
        - Local patch-based uniformity
        - Color gradient analysis
        - Hue va saturation consistency
        
        Args:
            hsv_image: HSV color space image
            grayscale_image: Grayscale image
            
        Returns:
            dict: Enhanced color uniformity metrics
        """
        # Asosiy metrikalar (mavjud)
        value_channel = hsv_image[:, :, 2]
        gray_variance = np.var(grayscale_image)
        gray_std = np.std(grayscale_image)
        value_variance = np.var(value_channel)
        value_std = np.std(value_channel)
        
        # YANGI: Histogram analysis
        gray_hist = cv2.calcHist([grayscale_image], [0], None, [256], [0, 256])
        hist_std = np.std(gray_hist)  # Histogram tarqalishi
        hist_normalized = gray_hist / np.sum(gray_hist)
        hist_entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
        
        # YANGI: Local color uniformity (patch-based)
        h, w = grayscale_image.shape
        patch_size = 64
        local_variances = []
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                patch = grayscale_image[i:i+patch_size, j:j+patch_size]
                local_variances.append(np.var(patch))
        local_uniformity = np.std(local_variances) if local_variances else 0
        # Past local_uniformity = bir xil rang taqsimoti = yaxshi
        
        # YANGI: Color gradient analysis
        grad_x = cv2.Sobel(grayscale_image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(grayscale_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_variance = np.var(gradient_magnitude)
        
        # YANGI: Hue va Saturation consistency
        hue_channel = hsv_image[:, :, 0]
        saturation_channel = hsv_image[:, :, 1]
        hue_variance = np.var(hue_channel)
        sat_variance = np.var(saturation_channel)
        
        # Combined enhanced variance (weighted)
        enhanced_variance = (
            gray_variance * 0.25 +
            value_variance * 0.20 +
            hist_std * 0.10 +
            local_uniformity * 0.15 +
            gradient_variance * 0.15 +
            hue_variance * 0.10 +
            sat_variance * 0.05
        )
        
        return {
            'gray_variance': gray_variance,
            'gray_std': gray_std,
            'value_variance': value_variance,
            'value_std': value_std,
            'combined_variance': enhanced_variance,
            'hist_std': hist_std,
            'hist_entropy': hist_entropy,
            'local_uniformity': local_uniformity,
            'gradient_variance': gradient_variance,
            'hue_variance': hue_variance,
            'sat_variance': sat_variance
        }
    
    def calculate_color_consistency_score(self, hsv_image):
        """
        Rang izchilligi ballini hisoblash (0-100, yuqori = yaxshi).
        
        Args:
            hsv_image: HSV color space image
            
        Returns:
            float: Consistency score (0-100)
        """
        hue_channel = hsv_image[:, :, 0]
        saturation_channel = hsv_image[:, :, 1]
        
        # Hue consistency (past = yaxshi)
        hue_variance = np.var(hue_channel)
        hue_std = np.std(hue_channel)
        
        # Saturation consistency
        sat_variance = np.var(saturation_channel)
        
        # Color consistency score (0-100, yuqori = yaxshi)
        # Normalize: variance 0-1000 range ni 0-100 ga map qilish
        consistency_score = 100 - min(100, (hue_variance + sat_variance) / 10)
        
        return max(0, consistency_score)
    
    def detect_defects(self, grayscale_image):
        """
        Detect visible defects (stains, dark spots, bright areas) using thresholding and contours.
        
        Method:
        1. Apply adaptive thresholding to separate defects from normal areas
        2. Use morphological operations to clean up the thresholded image
        3. Find contours to identify defect regions
        4. Calculate total defect area
        
        Args:
            grayscale_image: Preprocessed grayscale image
            
        Returns:
            dict: Defect detection results
        """
        # Step 1: Apply adaptive thresholding
        # Adaptive thresholding is better than global thresholding because
        # it adjusts to local image conditions
        # This helps detect both dark defects (stains) and bright defects
        threshold1 = cv2.adaptiveThreshold(
            grayscale_image, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 
            11,  # Block size for local threshold calculation
            2    # Constant subtracted from mean
        )
        
        # Also detect bright areas (defects that are brighter than normal)
        threshold2 = cv2.adaptiveThreshold(
            grayscale_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )
        
        # Combine both thresholds to detect all types of defects
        combined_threshold = cv2.bitwise_or(threshold1, threshold2)
        
        # Step 2: Apply morphological operations to clean up noise
        # Opening: erosion followed by dilation - removes small noise
        # Closing: dilation followed by erosion - fills small holes
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(combined_threshold, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Step 3: Find contours (boundaries of defect regions)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Step 4: Calculate total defect area
        total_defect_area = 0
        significant_defects = []
        
        image_area = grayscale_image.shape[0] * grayscale_image.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter out very small contours (likely noise)
            if area > 200:  # Yumshatildi: faqat katta nuqsonlarni aniqlash (50 o'rniga 200)
                total_defect_area += area
                significant_defects.append(contour)
        
        # Calculate defect area as percentage of total image
        defect_percentage = (total_defect_area / image_area) * 100 if image_area > 0 else 0
        
        # Create visualization of defects
        defect_visualization = grayscale_image.copy()
        defect_visualization = cv2.cvtColor(defect_visualization, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(defect_visualization, significant_defects, -1, (0, 0, 255), 2)
        
        return {
            'defect_count': len(significant_defects),
            'total_defect_area': total_defect_area,
            'defect_percentage': defect_percentage,
            'contours': significant_defects,
            'visualization': defect_visualization
        }
    
    def analyze_texture_features(self, grayscale_image):
        """
        Texture xususiyatlarini chuqur tahlil qilish.
        
        Args:
            grayscale_image: Preprocessed grayscale image
            
        Returns:
            dict: Texture features
        """
        # 1. Local Binary Pattern (LBP) - oddiy variant
        # Patch-based texture analysis
        h, w = grayscale_image.shape
        patch_size = 32
        texture_scores = []
        
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                patch = grayscale_image[i:i+patch_size, j:j+patch_size]
                texture_scores.append(np.std(patch))
        
        texture_variance = np.var(texture_scores) if texture_scores else 0
        texture_mean = np.mean(texture_scores) if texture_scores else 0
        
        # 2. Edge density
        edges = cv2.Canny(grayscale_image, 50, 150)
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.shape[0] * edges.shape[1]
        edge_density = (edge_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        
        # 3. Texture uniformity (past = yaxshi)
        texture_uniformity = texture_variance
        
        return {
            'texture_variance': texture_variance,
            'texture_mean': texture_mean,
            'edge_density': edge_density,
            'texture_uniformity': texture_uniformity
        }
    
    def calculate_defect_severity(self, defect_percentage, defect_count, defect_shapes=None):
        """
        Nuqson og'irligi ballini hisoblash (0-100, past = yaxshi).
        
        Args:
            defect_percentage: Nuqson maydoni foizi
            defect_count: Nuqsonlar soni
            defect_shapes: Nuqson shakllari ro'yxati (optional)
            
        Returns:
            float: Severity score (0-100)
        """
        # Area-based severity
        area_score = min(100, defect_percentage * 10)
        
        # Count-based severity
        count_score = min(100, defect_count * 5)
        
        # Shape-based severity (agar mavjud bo'lsa)
        shape_score = 0
        if defect_shapes and len(defect_shapes) > 0:
            # Circularity: 1 = aylana, 0 = chiziq
            # Past circularity = murakkab shakl = yomon
            avg_circularity = np.mean([s.get('circularity', 0.5) for s in defect_shapes])
            shape_score = (1 - avg_circularity) * 50
        
        # Combined severity (weighted)
        severity = (area_score * 0.5 + count_score * 0.3 + shape_score * 0.2)
        
        return min(100, max(0, severity))
    
    def detect_defects_enhanced(self, grayscale_image):
        """
        Yaxshilangan nuqson aniqlash - multi-scale, edge-based, texture-based.
        
        Qo'shimcha metodlar:
        - Multi-scale analysis
        - Edge-based detection
        - Texture-based detection
        - Blob detection
        - Contour shape analysis
        
        Args:
            grayscale_image: Preprocessed grayscale image
            
        Returns:
            dict: Enhanced defect detection results
        """
        # 1. Mavjud adaptive thresholding
        threshold1 = cv2.adaptiveThreshold(
            grayscale_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        threshold2 = cv2.adaptiveThreshold(
            grayscale_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        combined_threshold = cv2.bitwise_or(threshold1, threshold2)
        
        # 2. YANGI: Multi-scale analysis
        scales = [0.5, 1.0, 2.0]  # Kichik, original, katta
        for scale in scales:
            if scale != 1.0:
                scaled = cv2.resize(grayscale_image, None, fx=scale, fy=scale)
                scaled_thresh = cv2.adaptiveThreshold(
                    scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 11, 2
                )
                scaled_thresh = cv2.resize(scaled_thresh, 
                                          (grayscale_image.shape[1], 
                                           grayscale_image.shape[0]))
                combined_threshold = cv2.bitwise_or(combined_threshold, scaled_thresh)
        
        # 3. YANGI: Edge-based defect detection (yumshatildi)
        edges = cv2.Canny(grayscale_image, 50, 150)
        edge_density_map = cv2.GaussianBlur(edges.astype(np.float32), (15, 15), 0)
        edge_threshold = edge_density_map > (np.mean(edge_density_map) * 2.5)  # 1.5 o'rniga 2.5 - kamroq sezuvchan
        combined_threshold = cv2.bitwise_or(
            combined_threshold, 
            (edge_threshold * 255).astype(np.uint8)
        )
        
        # 4. Morphological operations (yaxshilangan)
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_medium = np.ones((5, 5), np.uint8)
        
        # Opening - kichik shovqinni olib tashlash
        cleaned = cv2.morphologyEx(combined_threshold, cv2.MORPH_OPEN, kernel_small)
        # Closing - teshiklarni to'ldirish
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium)
        
        # 5. YANGI: Connected components filtering
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned)
        min_area = 200  # Yumshatildi: faqat katta nuqsonlarni aniqlash (50 o'rniga 200)
        cleaned_filtered = np.zeros_like(cleaned)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned_filtered[labels == i] = 255
        
        # 6. Contour analysis (yaxshilangan)
        contours, _ = cv2.findContours(cleaned_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        total_defect_area = 0
        significant_defects = []
        defect_shapes = []  # YANGI: Nuqson shakllari
        
        image_area = grayscale_image.shape[0] * grayscale_image.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:  # Yumshatildi: faqat katta nuqsonlarni aniqlash (50 o'rniga 200)
                total_defect_area += area
                
                # YANGI: Contour shape analysis
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    defect_shapes.append({
                        'area': area,
                        'circularity': circularity,
                        'perimeter': perimeter
                    })
                
                significant_defects.append(contour)
        
        defect_percentage = (total_defect_area / image_area) * 100 if image_area > 0 else 0
        
        # YANGI: Defect severity scoring
        severity_score = self.calculate_defect_severity(
            defect_percentage,
            len(significant_defects),
            defect_shapes
        )
        
        # Visualization
        defect_visualization = grayscale_image.copy()
        defect_visualization = cv2.cvtColor(defect_visualization, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(defect_visualization, significant_defects, -1, (0, 0, 255), 2)
        
        return {
            'defect_count': len(significant_defects),
            'total_defect_area': total_defect_area,
            'defect_percentage': defect_percentage,
            'defect_shapes': defect_shapes,
            'severity_score': severity_score,
            'contours': significant_defects,
            'visualization': defect_visualization
        }
    
    def classify_quality(self, color_metrics, defect_metrics):
        """
        Classify textile quality based on rule-based logic.
        
        Classification Rules:
        - Yaxshi (Good): Low color variance AND small defect area
        - O'rtacha (Medium): Medium color variance OR moderate defect area
        - Yaroqsiz (Bad): High color variance OR large defect area
        
        Args:
            color_metrics: Dictionary from analyze_color_uniformity()
            defect_metrics: Dictionary from detect_defects()
            
        Returns:
            tuple: (quality_label, explanation)
        """
        combined_variance = color_metrics['combined_variance']
        defect_percentage = defect_metrics['defect_percentage']
        defect_count = defect_metrics['defect_count']
        
        # Rule 1: Check color uniformity
        color_good = combined_variance < self.COLOR_VARIANCE_THRESHOLD_GOOD
        color_medium = (combined_variance >= self.COLOR_VARIANCE_THRESHOLD_GOOD and 
                       combined_variance < self.COLOR_VARIANCE_THRESHOLD_MEDIUM)
        color_bad = combined_variance >= self.COLOR_VARIANCE_THRESHOLD_MEDIUM
        
        # Rule 2: Check defect area
        defect_good = defect_percentage < (self.DEFECT_AREA_THRESHOLD_SMALL * 100)
        defect_medium = (defect_percentage >= (self.DEFECT_AREA_THRESHOLD_SMALL * 100) and 
                        defect_percentage < (self.DEFECT_AREA_THRESHOLD_MEDIUM * 100))
        defect_bad = defect_percentage >= (self.DEFECT_AREA_THRESHOLD_MEDIUM * 100)
        
        # Rule 3: Check defect count (yumshatildi)
        defect_count_good = defect_count < 10  # 5 o'rniga 10
        defect_count_medium = defect_count >= 10 and defect_count < 25  # 5-15 o'rniga 10-25
        defect_count_bad = defect_count >= 25  # 15 o'rniga 25
        
        # Combined classification logic
        # Priority: If any metric is bad, overall quality is at least medium
        # If any metric is very bad, overall quality is bad
        
        if (color_bad or defect_bad or defect_count_bad):
            quality = "Yaroqsiz"  # Bad
            explanation = (
                f"Sifat: Yaroqsiz\n\n"
                f"Tahlil Natijalari:\n"
                f"• Rang Bir xilligi: {combined_variance:.2f} (Yuqori - rang bir xil emas)\n"
                f"• Nuqson Maydoni: {defect_percentage:.2f}% rasmdan\n"
                f"• Nuqsonlar Soni: {defect_count} ta nuqson aniqlandi\n\n"
                f"Sabab: Sezilarli rang o'zgarishlari yoki nuqsonlar aniqlandi. "
                f"To'qimachilik mahsuloti sifat standartlariga javob bermaydi."
            )
            
        elif (color_medium or defect_medium or defect_count_medium):
            quality = "O'rtacha"  # Medium
            explanation = (
                f"Sifat: O'rtacha\n\n"
                f"Tahlil Natijalari:\n"
                f"• Rang Bir xilligi: {combined_variance:.2f} (O'rtacha)\n"
                f"• Nuqson Maydoni: {defect_percentage:.2f}% rasmdan\n"
                f"• Nuqsonlar Soni: {defect_count} ta nuqson aniqlandi\n\n"
                f"Sabab: Ba'zi rang o'zgarishlari yoki kichik nuqsonlar mavjud. "
                f"To'qimachilik mahsuloti sifat jihatidan qabul qilinadi, lekin yaxshilanishi mumkin."
            )
            
        else:  # color_good and defect_good and defect_count_good
            quality = "Yaxshi"  # Good
            explanation = (
                f"Sifat: Yaxshi\n\n"
                f"Tahlil Natijalari:\n"
                f"• Rang Bir xilligi: {combined_variance:.2f} (Past - bir xil rang)\n"
                f"• Nuqson Maydoni: {defect_percentage:.2f}% rasmdan\n"
                f"• Nuqsonlar Soni: {defect_count} ta nuqson aniqlandi\n\n"
                f"Sabab: Bir xil rang taqsimoti va minimal nuqsonlar. "
                f"To'qimachilik mahsuloti yuqori sifat standartlariga javob beradi."
            )
        
        return quality, explanation
    
    def classify_quality_enhanced(self, color_metrics, defect_metrics, texture_metrics):
        """
        Yaxshilangan sifat tasnifi - weighted scoring system.
        
        Args:
            color_metrics: Enhanced color analysis results
            defect_metrics: Enhanced defect detection results
            texture_metrics: Texture analysis results
            
        Returns:
            tuple: (quality, explanation, score)
        """
        # Weighted scoring
        scores = {
            'color': 0,
            'defect': 0,
            'texture': 0
        }
        
        # Color score (0-100, yuqori = yaxshi)
        color_variance = color_metrics['combined_variance']
        # Color consistency score (agar hsv_image mavjud bo'lsa)
        hsv_for_consistency = color_metrics.get('hsv_image', None)
        if hsv_for_consistency is not None:
            color_consistency = self.calculate_color_consistency_score(hsv_for_consistency)
        else:
            # Default consistency score based on variance
            color_consistency = max(0, 100 - (color_variance / 20))
        
        # Yumshatilgan threshold qiymatlar
        if color_variance < 1500:  # 500 o'rniga 1500
            scores['color'] = 90
        elif color_variance < 2500:  # 1000 o'rniga 2500
            scores['color'] = 70
        elif color_variance < 3000:  # 1500 o'rniga 3000
            scores['color'] = 50
        else:
            scores['color'] = 30
        
        # Color consistency ni qo'shish
        scores['color'] = (scores['color'] * 0.7 + color_consistency * 0.3)
        
        # Defect score (0-100, yuqori = yaxshi)
        defect_percentage = defect_metrics['defect_percentage']
        defect_count = defect_metrics['defect_count']
        severity = defect_metrics.get('severity_score', 0)
        
        defect_score = 100 - min(100, severity)
        if defect_count > 30:  # 20 o'rniga 30 - yumshatildi
            defect_score -= 20
        if defect_percentage > 10:  # 5 o'rniga 10 - yumshatildi
            defect_score -= 15
        
        scores['defect'] = max(0, defect_score)
        
        # Texture score (0-100, yuqori = yaxshi)
        texture_variance = texture_metrics.get('texture_variance', 0)
        texture_uniformity = texture_metrics.get('texture_uniformity', 0)
        
        if texture_variance < 20:
            scores['texture'] = 90
        elif texture_variance < 40:
            scores['texture'] = 70
        else:
            scores['texture'] = 50
        
        # Texture uniformity ni hisobga olish
        if texture_uniformity < 10:
            scores['texture'] += 10
        
        scores['texture'] = min(100, scores['texture'])
        
        # Weighted final score
        weights = {'color': 0.4, 'defect': 0.5, 'texture': 0.1}
        final_score = (
            scores['color'] * weights['color'] +
            scores['defect'] * weights['defect'] +
            scores['texture'] * weights['texture']
        )
        
        # Classification with confidence
        if final_score >= 80:
            quality = "Yaxshi"
            confidence = (final_score - 80) / 20  # 0-1 scale
        elif final_score >= 60:
            quality = "O'rtacha"
            confidence = (final_score - 60) / 20
        else:
            quality = "Yaroqsiz"
            confidence = (60 - final_score) / 60
        
        # Detailed explanation
        explanation = (
            f"Sifat: {quality}\n\n"
            f"Yaxshilangan Tahlil Natijalari:\n"
            f"• Umumiy Ball: {final_score:.1f}/100\n"
            f"• Rang Balli: {scores['color']:.1f}/100\n"
            f"• Nuqson Balli: {scores['defect']:.1f}/100\n"
            f"• Texture Balli: {scores['texture']:.1f}/100\n\n"
            f"Tafsilotlar:\n"
            f"• Rang Bir xilligi: {color_variance:.2f}\n"
            f"• Nuqson Maydoni: {defect_percentage:.2f}% rasmdan\n"
            f"• Nuqsonlar Soni: {defect_count} ta\n"
            f"• Nuqson Og'irligi: {severity:.1f}/100\n\n"
            f"Tahlil: Yaxshilangan OpenCV metodlari asosida "
            f"mahsulot sifatini '{quality}' deb baholadi "
            f"(Ball: {final_score:.1f}/100)."
        )
        
        return quality, explanation, final_score
    
    def analyze_with_cv(self, processed_images):
        """
        Klassik CV tahlil (backup yoki Hybrid uchun).
        
        Args:
            processed_images: Preprocessed images dict
            
        Returns:
            dict: CV analysis results
        """
        color_metrics = self.analyze_color_uniformity(
            processed_images['hsv'],
            processed_images['grayscale']
        )
        defect_metrics = self.detect_defects(processed_images['grayscale'])
        
        quality, explanation = self.classify_quality(color_metrics, defect_metrics)
        
        return {
            'quality': quality,
            'explanation': explanation,
            'visualization': defect_metrics['visualization'],
            'color_metrics': color_metrics,
            'defect_metrics': defect_metrics
        }
    
    def analyze_with_cv_enhanced(self, processed_images):
        """
        Yaxshilangan CV tahlil - enhanced metodlar bilan.
        
        Args:
            processed_images: Preprocessed images dict (enhanced preprocessing)
            
        Returns:
            dict: Enhanced CV analysis results
        """
        # Enhanced color analysis
        color_metrics = self.analyze_color_uniformity_enhanced(
            processed_images['hsv'],
            processed_images['grayscale']
        )
        color_metrics['hsv_image'] = processed_images['hsv']  # For consistency score
        
        # Enhanced defect detection
        defect_metrics = self.detect_defects_enhanced(processed_images['grayscale'])
        
        # Texture analysis
        texture_metrics = self.analyze_texture_features(processed_images['grayscale'])
        
        # Enhanced classification
        quality, explanation, score = self.classify_quality_enhanced(
            color_metrics,
            defect_metrics,
            texture_metrics
        )
        
        return {
            'quality': quality,
            'explanation': explanation,
            'visualization': defect_metrics['visualization'],
            'score': score,
            'color_metrics': color_metrics,
            'defect_metrics': defect_metrics,
            'texture_metrics': texture_metrics
        }
    
    
    def analyze_image(self, image):
        """
        Main analysis function that orchestrates all processing steps.
        
        Steps:
        1. Preprocess image
        2. Analyze color uniformity
        3. Detect defects
        4. Classify quality
        5. Display results
        
        Args:
            image: Input BGR image (OpenCV format)
        """
        try:
            # Set analyzing state
            self.is_analyzing = True
            
            # Update status with loading indicator
            if not self.is_camera_running:
                self.status_label.config(text="Tahlil qilinmoqda...")
                self.root.update()
            
            # Step 1: Preprocess the image (enhanced)
            processed = self.preprocess_image_enhanced(image)
            
            # Step 2: Enhanced CV analysis
            cv_results = self.analyze_with_cv_enhanced(processed)
            quality = cv_results['quality']
            explanation = cv_results['explanation']
            visualization = cv_results['visualization']
            score = cv_results.get('score', 0.0)
            
            # Step 3: Display results
            self.display_results(
                processed['original'],
                visualization,
                quality,
                explanation,
                score
            )
            
            # Reset analyzing state
            self.is_analyzing = False
            
            # Re-enable buttons if not in camera mode
            if not self.is_camera_running:
                self.set_buttons_state(tk.NORMAL)
            
        except Exception as e:
            self.is_analyzing = False
            messagebox.showerror("Tahlil Xatoligi", f"Tahlil qilishda xatolik: {str(e)}")
            self.status_label.config(text=f"Xatolik: {str(e)}")
            if not self.is_camera_running:
                self.set_buttons_state(tk.NORMAL)
    
    def display_results(self, original_image, processed_image, quality, explanation, score=0.0):
        """
        Display the analysis results in the GUI.
        
        Args:
            original_image: Original input image
            processed_image: Image with defect visualization
            quality: Quality classification string
            explanation: Detailed explanation text
            score: CV analysis score (0-100)
        """
        # Update quality result label with color coding (Soft UI colors)
        self.result_label.config(text=quality)
        
        if quality == "Yaxshi":
            self.result_label.config(fg=self.COLOR_BUTTON_GREEN)  # Soft green
        elif quality == "O'rtacha":
            self.result_label.config(fg='#d4a574')  # Soft orange/tan
        else:  # Yaroqsiz
            self.result_label.config(fg=self.COLOR_BUTTON_RED)  # Soft red
        
        # Update explanation text
        self.explanation_text.delete(1.0, tk.END)
        self.explanation_text.insert(1.0, explanation)
        
        # Display image on canvas
        # Combine original and processed images side by side for comparison
        height, width = original_image.shape[:2]
        
        # Resize images to fit canvas
        canvas_width = 800
        canvas_height = 600
        
        # Calculate display size maintaining aspect ratio
        scale_w = canvas_width / (width * 2)  # *2 because we show 2 images side by side
        scale_h = canvas_height / height
        scale = min(scale_w, scale_h, 1.0)  # Don't upscale
        
        display_width = int(width * scale)
        display_height = int(height * scale)
        
        # Resize images
        original_resized = cv2.resize(original_image, (display_width, display_height))
        processed_resized = cv2.resize(processed_image, (display_width, display_height))
        
        # Combine images side by side
        combined = np.hstack([original_resized, processed_resized])
        
        # Convert BGR to RGB for Tkinter
        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image and then to PhotoImage
        pil_image = Image.fromarray(combined_rgb)
        photo = ImageTk.PhotoImage(image=pil_image)
        
        # Update canvas
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            image=photo,
            anchor=tk.CENTER
        )
        self.canvas.image = photo  # Keep a reference
        
        # Add labels on canvas with background for better visibility
        label_bg_offset = 5
        # Background rectangles for labels
        self.canvas.create_rectangle(
            display_width // 2 - 80,
            5,
            display_width // 2 + 80,
            35,
            fill='black',
            outline='',
            stipple='gray50'
        )
        self.canvas.create_rectangle(
            display_width + display_width // 2 - 90,
            5,
            display_width + display_width // 2 + 90,
            35,
            fill='black',
            outline='',
            stipple='gray50'
        )
        
        # Labels in Uzbek
        self.canvas.create_text(
            display_width // 2,
            20,
            text="Asl Rasm",
            fill="white",
            font=("Arial", 12, "bold")
        )
        self.canvas.create_text(
            display_width + display_width // 2,
            20,
            text="Nuqson Aniqlash",
            fill="white",
            font=("Arial", 12, "bold")
        )
        
        # Update status with Uzbek text
        if self.is_camera_running:
            self.status_label.config(text=f"Real-vaqtda tahlil - Sifat: {quality}")
        else:
            self.status_label.config(text=f"Tahlil yakunlandi - Sifat: {quality}")


def main():
    """
    Main entry point of the application.
    Creates and runs the Tkinter GUI application.
    """
    root = tk.Tk()
    app = TextileQualityAssessment(root)
    root.mainloop()


if __name__ == "__main__":
    main()

