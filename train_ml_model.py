"""
ML Model Training Script
========================
Bu script ML modelni o'qitish uchun ishlatiladi.
To'qimachilik mahsulotlari sifatini baholash uchun Random Forest model o'qitiladi.

Ishlatish:
1. Ma'lumotlarni to'plang (rasmlar va ularning label'lari)
2. Feature extraction qiling
3. Model o'qiting
4. Modelni saqlang
"""

import cv2
import numpy as np
import os
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
from pathlib import Path


class TextileFeatureExtractor:
    """
    To'qimachilik rasmlaridan xususiyatlarni ajratish.
    Bu asosiy aplikatsiyadagi feature extraction bilan bir xil.
    """
    
    def __init__(self):
        self.COLOR_VARIANCE_THRESHOLD_GOOD = 500
        self.COLOR_VARIANCE_THRESHOLD_MEDIUM = 1500
        self.DEFECT_AREA_THRESHOLD_SMALL = 0.01
        self.DEFECT_AREA_THRESHOLD_MEDIUM = 0.05
    
    def preprocess_image(self, image):
        """Rasmni tayyorlash"""
        height, width = image.shape[:2]
        max_dimension = 800
        if width > max_dimension or height > max_dimension:
            scale = max_dimension / max(width, height)
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        else:
            resized = image.copy()
        
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        grayscale = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        return {
            'original': resized,
            'blurred': blurred,
            'grayscale': grayscale,
            'hsv': hsv
        }
    
    def analyze_color_uniformity(self, hsv_image, grayscale_image):
        """Rang bir xilligini tahlil qilish"""
        value_channel = hsv_image[:, :, 2]
        gray_variance = np.var(grayscale_image)
        gray_std = np.std(grayscale_image)
        value_variance = np.var(value_channel)
        value_std = np.std(value_channel)
        combined_variance = (gray_variance + value_variance) / 2
        
        return {
            'gray_variance': gray_variance,
            'gray_std': gray_std,
            'value_variance': value_variance,
            'value_std': value_std,
            'combined_variance': combined_variance
        }
    
    def detect_defects(self, grayscale_image):
        """Nuqsonlarni aniqlash"""
        threshold1 = cv2.adaptiveThreshold(
            grayscale_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        threshold2 = cv2.adaptiveThreshold(
            grayscale_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        combined_threshold = cv2.bitwise_or(threshold1, threshold2)
        
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(combined_threshold, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        total_defect_area = 0
        significant_defects = []
        image_area = grayscale_image.shape[0] * grayscale_image.shape[1]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:
                total_defect_area += area
                significant_defects.append(contour)
        
        defect_percentage = (total_defect_area / image_area) * 100 if image_area > 0 else 0
        
        return {
            'defect_count': len(significant_defects),
            'total_defect_area': total_defect_area,
            'defect_percentage': defect_percentage
        }
    
    def extract_texture_features(self, grayscale_image):
        """Texture xususiyatlarini ajratish"""
        h, w = grayscale_image.shape
        patch_size = 32
        texture_scores = []
        
        for i in range(0, h - patch_size, patch_size):
            for j in range(0, w - patch_size, patch_size):
                patch = grayscale_image[i:i+patch_size, j:j+patch_size]
                texture_scores.append(np.std(patch))
        
        return np.mean(texture_scores) if texture_scores else 0
    
    def calculate_edge_density(self, grayscale_image):
        """Edge density hisoblash"""
        edges = cv2.Canny(grayscale_image, 50, 150)
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.shape[0] * edges.shape[1]
        return (edge_pixels / total_pixels) * 100 if total_pixels > 0 else 0
    
    def extract_features(self, image_path):
        """
        Rasmdan barcha xususiyatlarni ajratish.
        
        Args:
            image_path: Rasm fayl yo'li
            
        Returns:
            np.array: Feature vector
        """
        # Rasmni yuklash
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Rasm yuklanmadi: {image_path}")
        
        # Preprocessing
        processed = self.preprocess_image(image)
        
        # Color analysis
        color_metrics = self.analyze_color_uniformity(
            processed['hsv'],
            processed['grayscale']
        )
        
        # Defect detection
        defect_metrics = self.detect_defects(processed['grayscale'])
        
        # Texture and edge features
        texture_score = self.extract_texture_features(processed['grayscale'])
        edge_density = self.calculate_edge_density(processed['grayscale'])
        
        # Feature array
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


def load_dataset(data_dir):
    """
    Dataset ni yuklash.
    
    Kutilgan struktura:
    data_dir/
        yaxshi/
            image1.jpg
            image2.jpg
            ...
        ortacha/
            image1.jpg
            ...
        yaroqsiz/
            image1.jpg
            ...
    
    Args:
        data_dir: Dataset papkasi yo'li
        
    Returns:
        tuple: (features, labels)
    """
    extractor = TextileFeatureExtractor()
    
    features = []
    labels = []
    
    # Label mapping
    label_map = {
        'yaxshi': 0,
        'ortacha': 1,
        'yaroqsiz': 2
    }
    
    # Har bir klass uchun
    for label_name, label_id in label_map.items():
        label_dir = os.path.join(data_dir, label_name)
        
        if not os.path.exists(label_dir):
            print(f"Ogohlantirish: {label_dir} topilmadi, o'tkazib yuborildi.")
            continue
        
        image_files = [f for f in os.listdir(label_dir) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        print(f"\n{label_name.upper()} klassidan {len(image_files)} ta rasm topildi...")
        
        for img_file in image_files:
            img_path = os.path.join(label_dir, img_file)
            try:
                feature_vector = extractor.extract_features(img_path)
                features.append(feature_vector)
                labels.append(label_id)
                print(f"  ✓ {img_file}")
            except Exception as e:
                print(f"  ✗ {img_file} - Xatolik: {e}")
    
    return np.array(features), np.array(labels)


def train_model(X, y, test_size=0.2, random_state=42):
    """
    ML modelni o'qitish.
    
    Args:
        X: Feature matrix
        y: Labels
        test_size: Test set o'lchami
        random_state: Random seed
        
    Returns:
        tuple: (trained_model, test_accuracy, classification_report)
    """
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\n{'='*60}")
    print("MODEL O'QITISH")
    print(f"{'='*60}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train.shape[1]}")
    
    # Model yaratish
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    
    # O'qitish
    print("\nModel o'qitilmoqda...")
    model.fit(X_train, y_train)
    
    # Baholash
    print("\nModel baholanmoqda...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    # Classification report
    label_names = ['Yaxshi', "O'rtacha", 'Yaroqsiz']
    report = classification_report(y_test, y_pred, target_names=label_names)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print("NATIJALAR")
    print(f"{'='*60}")
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    print(f"Cross-Validation Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*2*100:.2f}%)")
    print(f"\nClassification Report:\n{report}")
    print(f"\nConfusion Matrix:\n{cm}")
    
    return model, accuracy, report


def save_model(model, output_dir='models'):
    """
    Modelni saqlash.
    
    Args:
        model: Trained model
        output_dir: Saqlash papkasi
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'quality_classifier.pkl')
    joblib.dump(model, model_path)
    print(f"\n✓ Model saqlandi: {model_path}")


def main():
    """
    Asosiy funksiya.
    """
    print("="*60)
    print("TO'QIMACHILIK MAHSULOTLARI SIFATINI BAHOLASH")
    print("ML MODEL O'QITISH")
    print("="*60)
    
    # Dataset yo'li
    data_dir = input("\nDataset papkasi yo'lini kiriting (masalan: data/train): ").strip()
    
    if not data_dir or not os.path.exists(data_dir):
        print("Xatolik: Dataset papkasi topilmadi!")
        return
    
    # Dataset yuklash
    print(f"\nDataset yuklanmoqda: {data_dir}")
    try:
        X, y = load_dataset(data_dir)
        
        if len(X) == 0:
            print("Xatolik: Hech qanday rasm topilmadi!")
            return
        
        print(f"\n✓ Jami {len(X)} ta rasm yuklandi")
        print(f"  - Yaxshi: {np.sum(y == 0)}")
        print(f"  - O'rtacha: {np.sum(y == 1)}")
        print(f"  - Yaroqsiz: {np.sum(y == 2)}")
        
    except Exception as e:
        print(f"Xatolik: Dataset yuklashda muammo - {e}")
        return
    
    # Model o'qitish
    try:
        model, accuracy, report = train_model(X, y)
        
        # Modelni saqlash
        save_choice = input("\nModelni saqlashni xohlaysizmi? (ha/yoq): ").strip().lower()
        if save_choice in ['ha', 'yes', 'y', '']:
            save_model(model)
            print("\n✓ Model muvaffaqiyatli o'qitildi va saqlandi!")
        else:
            print("\nModel saqlanmadi.")
            
    except Exception as e:
        print(f"Xatolik: Model o'qitishda muammo - {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

