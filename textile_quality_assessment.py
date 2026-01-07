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
        self.COLOR_VARIANCE_THRESHOLD_GOOD = 500      # Low variance = uniform color = good
        self.COLOR_VARIANCE_THRESHOLD_MEDIUM = 1500   # Medium variance
        self.DEFECT_AREA_THRESHOLD_SMALL = 0.01       # 1% of image area
        self.DEFECT_AREA_THRESHOLD_MEDIUM = 0.05      # 5% of image area
        
        # ML Models and Analysis Mode
        self.ml_quality_model = None
        self.analysis_mode = "CV"  # "CV", "ML", or "Hybrid"
        self.ml_available = False
        
        # Store button references for state management
        self.upload_btn = None
        self.camera_btn = None
        self.stop_camera_btn = None
        self.mode_var = None
        self.confidence_label = None
        
        # Load ML models
        self.load_ml_models()
        
        self.setup_gui()
    
    def load_ml_models(self):
        """
        Load pre-trained ML models if available.
        Falls back to CV mode if models are not found.
        """
        try:
            import joblib
            model_path = os.path.join('models', 'quality_classifier.pkl')
            
            if os.path.exists(model_path):
                self.ml_quality_model = joblib.load(model_path)
                self.ml_available = True
                self.analysis_mode = "ML"  # Default to ML if available
                print(f"ML model yuklandi: {model_path}")
            else:
                self.ml_available = False
                self.analysis_mode = "CV"
                print(f"ML model topilmadi. CV rejimi ishlatilmoqda.")
                
        except ImportError:
            print("joblib o'rnatilmagan. ML rejimi ishlatilmaydi.")
            self.ml_available = False
            self.analysis_mode = "CV"
        except Exception as e:
            print(f"ML model yuklashda xatolik: {e}")
            self.ml_available = False
            self.analysis_mode = "CV"
    
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
            text="Rasm va Real-Vaqtli Kamera Tahlili Asosida Avtomatik Sifat Baholash",
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
        
        # Analysis Mode Selector
        mode_frame = tk.Frame(control_inner, bg=self.COLOR_PANEL)
        mode_frame.pack(pady=10, padx=15, fill=tk.X)
        
        tk.Label(
            mode_frame,
            text="Tahlil Usuli:",
            font=("Arial", 10, "bold"),
            bg=self.COLOR_PANEL,
            fg=self.COLOR_TEXT_PRIMARY
        ).pack(anchor=tk.W, pady=(0, 5))
        
        self.mode_var = tk.StringVar(value=self.analysis_mode)
        
        mode_options = []
        if self.ml_available:
            mode_options = [("ML", "ML"), ("Hybrid", "Hybrid"), ("CV", "CV")]
        else:
            mode_options = [("CV", "CV")]
            self.mode_var.set("CV")
        
        for text, value in mode_options:
            rb = tk.Radiobutton(
                mode_frame,
                text=text,
                variable=self.mode_var,
                value=value,
                command=self.change_analysis_mode,
                bg=self.COLOR_PANEL,
                fg=self.COLOR_TEXT_PRIMARY,
                selectcolor=self.COLOR_PANEL,
                activebackground=self.COLOR_PANEL,
                font=("Arial", 9)
            )
            rb.pack(anchor=tk.W, pady=2)
        
        # Confidence display label
        self.confidence_label = tk.Label(
            control_inner,
            text="Ishoning: -",
            font=("Arial", 9),
            bg=self.COLOR_PANEL,
            fg=self.COLOR_TEXT_SECONDARY
        )
        self.confidence_label.pack(pady=5, padx=15)
        
        # Separator with Soft UI styling
        separator2 = tk.Frame(control_inner, height=1, bg=self.COLOR_BORDER)
        separator2.pack(fill=tk.X, padx=15, pady=20)
        
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
    
    def change_analysis_mode(self):
        """
        Change analysis mode when user selects different mode.
        """
        self.analysis_mode = self.mode_var.get()
        mode_text = {
            "ML": "ML Model",
            "Hybrid": "Hybrid (ML + CV)",
            "CV": "Klassik CV"
        }
        self.status_label.config(text=f"Tahlil usuli: {mode_text.get(self.analysis_mode, self.analysis_mode)}")
    
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
            if area > 50:  # Minimum area threshold
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
        
        # Rule 3: Check defect count
        defect_count_good = defect_count < 5
        defect_count_medium = defect_count >= 5 and defect_count < 15
        defect_count_bad = defect_count >= 15
        
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
    
    def extract_ml_features(self, processed_images):
        """
        ML model uchun xususiyatlarni ajratish.
        
        Args:
            processed_images: Preprocessed images dict
            
        Returns:
            np.array: Feature vector for ML model
        """
        # CV dan olingan features
        color_metrics = self.analyze_color_uniformity(
            processed_images['hsv'],
            processed_images['grayscale']
        )
        defect_metrics = self.detect_defects(processed_images['grayscale'])
        
        # Texture features
        texture_score = self.extract_texture_features(processed_images['grayscale'])
        
        # Edge density
        edge_density = self.calculate_edge_density(processed_images['grayscale'])
        
        # Feature array (ML model uchun)
        feature_array = np.array([
            color_metrics['combined_variance'],
            color_metrics['gray_std'],
            color_metrics['value_variance'],
            defect_metrics['defect_percentage'],
            defect_metrics['defect_count'],
            texture_score,
            edge_density
        ])
        
        return feature_array
    
    def extract_texture_features(self, grayscale_image):
        """
        Texture xususiyatlarini ajratish.
        Oddiy variant: Standard deviation of local patches.
        """
        # Image ni kichik qismlarga bo'lish va har bir qismning std ni hisoblash
        h, w = grayscale_image.shape
        patch_size = 32
        texture_scores = []
        
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                patch = grayscale_image[i:i+patch_size, j:j+patch_size]
                texture_scores.append(np.std(patch))
        
        return np.mean(texture_scores) if texture_scores else 0
    
    def calculate_edge_density(self, grayscale_image):
        """
        Edge density hisoblash.
        """
        edges = cv2.Canny(grayscale_image, 50, 150)
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.shape[0] * edges.shape[1]
        return (edge_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    
    def analyze_with_ml(self, processed_images):
        """
        To'liq ML model orqali tahlil qilish.
        
        Args:
            processed_images: Preprocessed images dict
            
        Returns:
            dict: ML analysis results
        """
        results = {}
        
        # Feature extraction
        ml_features = self.extract_ml_features(processed_images)
        
        # Quality Classification with ML
        if self.ml_quality_model:
            try:
                # Reshape for single sample
                features_reshaped = ml_features.reshape(1, -1)
                
                # Predict
                quality_prediction = self.ml_quality_model.predict(features_reshaped)[0]
                quality_proba = self.ml_quality_model.predict_proba(features_reshaped)[0]
                confidence = float(np.max(quality_proba))
                
                # Map to Uzbek labels
                quality_map = {
                    0: 'Yaxshi',
                    1: "O'rtacha",
                    2: 'Yaroqsiz'
                }
                
                # Handle string labels too
                if isinstance(quality_prediction, str):
                    quality_uz = quality_prediction
                else:
                    quality_uz = quality_map.get(quality_prediction, f"Class_{quality_prediction}")
                
                results['quality'] = quality_uz
                results['confidence'] = confidence
                results['probabilities'] = {
                    'Yaxshi': float(quality_proba[0]) if len(quality_proba) > 0 else 0.0,
                    "O'rtacha": float(quality_proba[1]) if len(quality_proba) > 1 else 0.0,
                    'Yaroqsiz': float(quality_proba[2]) if len(quality_proba) > 2 else 0.0
                }
                
                # Visualization (use CV defect visualization)
                defect_metrics = self.detect_defects(processed_images['grayscale'])
                results['visualization'] = defect_metrics['visualization']
                
            except Exception as e:
                print(f"ML prediction xatoligi: {e}")
                # Fallback to CV
                cv_results = self.analyze_with_cv(processed_images)
                results['quality'] = cv_results['quality']
                results['confidence'] = 0.0
                results['probabilities'] = {'Yaxshi': 0.0, "O'rtacha": 0.0, 'Yaroqsiz': 0.0}
                results['visualization'] = cv_results['visualization']
        else:
            # No model available, use CV
            cv_results = self.analyze_with_cv(processed_images)
            results = cv_results
            results['confidence'] = 0.0
        
        return results
    
    def format_ml_results(self, ml_results):
        """
        ML natijalarini formatlash.
        
        Args:
            ml_results: ML analysis results dict
            
        Returns:
            tuple: (quality, explanation)
        """
        quality = ml_results['quality']
        confidence = ml_results.get('confidence', 0.0)
        probabilities = ml_results.get('probabilities', {})
        
        ortacha_prob = probabilities.get("O'rtacha", 0) * 100
        yaxshi_prob = probabilities.get('Yaxshi', 0) * 100
        yaroqsiz_prob = probabilities.get('Yaroqsiz', 0) * 100
        
        explanation = (
            f"Sifat: {quality}\n\n"
            f"ML Model Tahlili:\n"
            f"• Ishoning Darajasi: {confidence*100:.1f}%\n"
            f"• Yaxshi Ehtimoli: {yaxshi_prob:.1f}%\n"
            f"• O'rtacha Ehtimoli: {ortacha_prob:.1f}%\n"
            f"• Yaroqsiz Ehtimoli: {yaroqsiz_prob:.1f}%\n\n"
            f"Tahlil: ML model {confidence*100:.1f}% ishonch bilan "
            f"mahsulot sifatini '{quality}' deb baholadi."
        )
        
        return quality, explanation
    
    def combine_ml_cv_results(self, ml_results, cv_results):
        """
        ML va CV natijalarini birlashtirish (Hybrid mode).
        
        Args:
            ml_results: ML analysis results
            cv_results: CV analysis results
            
        Returns:
            tuple: (quality, explanation)
        """
        ml_quality = ml_results['quality']
        ml_confidence = ml_results.get('confidence', 0.0)
        cv_quality = cv_results['quality']
        
        # Decision logic
        if ml_confidence > 0.85:
            final_quality = ml_quality
            method = "ML Model (Yuqori ishonch)"
        elif ml_confidence > 0.60:
            if ml_quality == cv_quality:
                final_quality = ml_quality
                method = "Hybrid (ML + CV - Bir xil natija)"
            else:
                final_quality = ml_quality
                method = "Hybrid (ML asosiy, CV tekshiruv)"
        else:
            final_quality = cv_quality
            method = "CV (ML ishonchsiz)"
        
        # Format explanation
        ml_explanation = self.format_ml_results(ml_results)[1]
        
        explanation = (
            f"{ml_explanation}\n\n"
            f"---\n"
            f"Tahlil Usuli: {method}\n"
            f"CV Natijasi: {cv_quality}\n"
            f"ML Natijasi: {ml_quality}"
        )
        
        return final_quality, explanation
    
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
            mode_text = {
                "ML": "ML model tahlil qilmoqda...",
                "Hybrid": "Hybrid tahlil qilmoqda...",
                "CV": "Tahlil qilinmoqda..."
            }
            if not self.is_camera_running:
                self.status_label.config(text=mode_text.get(self.analysis_mode, "Tahlil qilinmoqda..."))
                self.root.update()
            
            # Step 1: Preprocess the image
            processed = self.preprocess_image(image)
            
            # Step 2: Analyze based on selected mode
            confidence = 0.0
            if self.analysis_mode == "ML" and self.ml_available:
                # Pure ML analysis
                ml_results = self.analyze_with_ml(processed)
                quality, explanation = self.format_ml_results(ml_results)
                visualization = ml_results.get('visualization', processed['grayscale'])
                confidence = ml_results.get('confidence', 0.0)
                
            elif self.analysis_mode == "Hybrid" and self.ml_available:
                # Hybrid: ML + CV
                ml_results = self.analyze_with_ml(processed)
                cv_results = self.analyze_with_cv(processed)
                quality, explanation = self.combine_ml_cv_results(ml_results, cv_results)
                visualization = ml_results.get('visualization', cv_results['visualization'])
                confidence = ml_results.get('confidence', 0.0)
                
            else:
                # CV analysis (fallback or selected)
                cv_results = self.analyze_with_cv(processed)
                quality = cv_results['quality']
                explanation = cv_results['explanation']
                visualization = cv_results['visualization']
                confidence = 0.0  # CV da confidence yo'q
            
            # Step 3: Display results
            self.display_results(
                processed['original'],
                visualization,
                quality,
                explanation,
                confidence
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
    
    def display_results(self, original_image, processed_image, quality, explanation, confidence=0.0):
        """
        Display the analysis results in the GUI.
        
        Args:
            original_image: Original input image
            processed_image: Image with defect visualization
            quality: Quality classification string
            explanation: Detailed explanation text
            confidence: ML model confidence score (0.0-1.0)
        """
        # Update quality result label with color coding (Soft UI colors)
        self.result_label.config(text=quality)
        
        if quality == "Yaxshi":
            self.result_label.config(fg=self.COLOR_BUTTON_GREEN)  # Soft green
        elif quality == "O'rtacha":
            self.result_label.config(fg='#d4a574')  # Soft orange/tan
        else:  # Yaroqsiz
            self.result_label.config(fg=self.COLOR_BUTTON_RED)  # Soft red
        
        # Update confidence label
        if confidence > 0:
            self.confidence_label.config(
                text=f"Ishoning: {confidence*100:.1f}%",
                fg=self.COLOR_TEXT_PRIMARY
            )
        else:
            self.confidence_label.config(
                text="Ishoning: - (CV rejimi)",
                fg=self.COLOR_TEXT_SECONDARY
            )
        
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

