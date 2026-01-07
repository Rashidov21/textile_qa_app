# ML Model O'qitish Bo'yicha Guideline

## 📋 Kirish

Bu guideline sizga to'qimachilik mahsulotlari sifatini baholash uchun ML modelni qanday o'qitishni tushuntirib beradi.

## 🎯 Maqsad

ML model o'qitish orqali:
- Aniqroq sifat baholash
- Confidence score ko'rsatish
- Murakkab holatlarni yaxshiroq hal qilish

## 📁 Loyiha Strukturasi

```
textile_qa_app/
├── textile_quality_assessment.py  # Asosiy aplikatsiya
├── train_ml_model.py              # Model o'qitish scripti
├── models/                        # O'qitilgan modellar
│   └── quality_classifier.pkl
├── data/                          # Training ma'lumotlari
│   └── train/
│       ├── yaxshi/
│       │   ├── image1.jpg
│       │   ├── image2.jpg
│       │   └── ...
│       ├── ortacha/
│       │   ├── image1.jpg
│       │   └── ...
│       └── yaroqsiz/
│           ├── image1.jpg
│           └── ...
└── ML_TRAINING_GUIDELINE.md       # Bu fayl
```

## 📦 Kerakli Paketlar

Avval quyidagi paketlarni o'rnating:

```bash
pip install scikit-learn joblib
```

Yoki `requirements.txt` ga qo'shing:

```txt
scikit-learn>=1.3.0
joblib>=1.3.0
```

## 📊 Dataset Tayyorlash

### 1. Ma'lumotlar To'plash

Sizga kerak:
- **Yaxshi** sifatli to'qimachilik rasmlari (kamida 50-100 ta)
- **O'rtacha** sifatli to'qimachilik rasmlari (kamida 50-100 ta)
- **Yaroqsiz** sifatli to'qimachilik rasmlari (kamida 50-100 ta)

**Maslahat:**
- Har bir klass uchun kamida 50 ta rasm bo'lishi kerak
- Rasm sifati yaxshi bo'lishi kerak (aniq, yaxshi yoritilgan)
- Turli xil to'qimachilik turlarini qo'shing
- Turli xil nuqsonlarni qo'shing

### 2. Papka Strukturasi

Dataset ni quyidagicha tuzing:

```
data/
└── train/
    ├── yaxshi/
    │   ├── textile_001.jpg
    │   ├── textile_002.jpg
    │   └── ...
    ├── ortacha/
    │   ├── textile_101.jpg
    │   ├── textile_102.jpg
    │   └── ...
    └── yaroqsiz/
        ├── textile_201.jpg
        ├── textile_202.jpg
        └── ...
```

**Muhim:**
- Papka nomlari kichik harflarda bo'lishi kerak: `yaxshi`, `ortacha`, `yaroqsiz`
- Rasm formatlari: `.jpg`, `.jpeg`, `.png`, `.bmp`

### 3. Rasm Talablari

- **O'lcham:** Ixtiyoriy (script avtomatik resize qiladi)
- **Format:** JPG, PNG, BMP
- **Sifat:** Aniq, yaxshi yoritilgan
- **Fon:** Bir xil, yorug' fon (masalan, oq yoki kulrang)

## 🚀 Model O'qitish

### 1. Scriptni Ishga Tushirish

```bash
python train_ml_model.py
```

### 2. Dataset Yo'lini Kiriting

Script sizdan dataset papkasi yo'lini so'raydi:

```
Dataset papkasi yo'lini kiriting (masalan: data/train): data/train
```

### 3. Jarayonni Kuzatish

Script quyidagilarni ko'rsatadi:

```
TO'QIMACHILIK MAHSULOTLARI SIFATINI BAHOLASH
ML MODEL O'QITISH
============================================================

Dataset yuklanmoqda: data/train

YAXSHI klassidan 75 ta rasm topildi...
  ✓ textile_001.jpg
  ✓ textile_002.jpg
  ...

ORTACHA klassidan 68 ta rasm topildi...
  ✓ textile_101.jpg
  ...

YAROQSIZ klassidan 82 ta rasm topildi...
  ✓ textile_201.jpg
  ...

✓ Jami 225 ta rasm yuklandi
  - Yaxshi: 75
  - O'rtacha: 68
  - Yaroqsiz: 82
```

### 4. Model O'qitish Natijalari

```
============================================================
MODEL O'QITISH
============================================================
Training samples: 180
Test samples: 45
Features: 7

Model o'qitilmoqda...

Model baholanmoqda...

============================================================
NATIJALAR
============================================================
Test Accuracy: 87.50%
Cross-Validation Accuracy: 85.33% (+/- 4.12%)

Classification Report:
              precision    recall  f1-score   support

      Yaxshi       0.90      0.89      0.90        19
    O'rtacha       0.85      0.88      0.86        16
    Yaroqsiz       0.88      0.87      0.87        10

    accuracy                           0.88        45
   macro avg       0.88      0.88      0.88        45
weighted avg       0.88      0.88      0.88        45

Confusion Matrix:
[[17  2  0]
 [ 2 14  0]
 [ 1  0  9]]
```

### 5. Modelni Saqlash

```
Modelni saqlashni xohlaysizmi? (ha/yoq): ha

✓ Model saqlandi: models/quality_classifier.pkl
✓ Model muvaffaqiyatli o'qitildi va saqlandi!
```

## 🔍 Feature Extraction

Model quyidagi 7 ta xususiyatdan foydalanadi:

1. **Color Variance (Rang Dispersiyasi)** - Rang bir xilligini ko'rsatadi
2. **Gray Standard Deviation** - Grayscale o'zgarishlari
3. **Value Variance** - HSV Value kanali dispersiyasi
4. **Defect Percentage** - Nuqson maydoni foizi
5. **Defect Count** - Nuqsonlar soni
6. **Texture Score** - Texture xususiyati
7. **Edge Density** - Edge zichligi

## 🎛️ Model Parametrlari

Hozirgi model: **Random Forest Classifier**

Parametrlar:
- `n_estimators=100` - Daraxtlar soni
- `max_depth=10` - Maksimal chuqurlik
- `min_samples_split=5` - Split uchun minimal namunalar
- `min_samples_leaf=2` - Leaf uchun minimal namunalar

Parametrlarni o'zgartirish uchun `train_ml_model.py` faylida `train_model()` funksiyasini tahrirlang.

## 📈 Model Yaxshilash

### 1. Ko'proq Ma'lumotlar

- Har bir klass uchun kamida 100+ rasm
- Turli xil sharoitlarda olingan rasmlar
- Turli xil to'qimachilik turlari

### 2. Feature Engineering

Yangi xususiyatlar qo'shish:
- Color histogram features
- Local Binary Pattern (LBP)
- Gabor filters
- Haralick texture features

### 3. Model Turlari

Boshqa modellarni sinab ko'ring:
- **SVM (Support Vector Machine)**
- **XGBoost**
- **Neural Network**

### 4. Hyperparameter Tuning

Grid Search yoki Random Search orqali eng yaxshi parametrlarni toping:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

## 🐛 Muammolarni Hal Qilish

### Muammo 1: "Dataset papkasi topilmadi"

**Yechim:** To'g'ri yo'lni kiriting. Masalan: `data/train` yoki `./data/train`

### Muammo 2: "Hech qanday rasm topilmadi"

**Yechim:** 
- Papka nomlarini tekshiring: `yaxshi`, `ortacha`, `yaroqsiz`
- Rasm formatlarini tekshiring: `.jpg`, `.jpeg`, `.png`, `.bmp`

### Muammo 3: "Model aniqligi past (<70%)"

**Yechim:**
- Ko'proq ma'lumotlar to'plang
- Ma'lumotlar sifatini yaxshilang
- Feature engineering qiling
- Model parametrlarini sozlang

### Muammo 4: "Class imbalance" (klasslar teng emas)

**Yechim:**
- Har bir klass uchun teng miqdordagi rasmlar to'plang
- SMOTE yoki undersampling ishlating
- `class_weight='balanced'` parametrini qo'shing

## ✅ Modelni Test Qilish

Model o'qitilgandan keyin:

1. **Aplikatsiyani ishga tushiring:**
   ```bash
   python textile_quality_assessment.py
   ```

2. **ML rejimini tanlang:**
   - GUI da "ML" yoki "Hybrid" rejimini tanlang

3. **Test rasmlarini yuklang:**
   - Test rasmlarini yuklab, natijalarni tekshiring

4. **Natijalarni baholang:**
   - Confidence score ni ko'ring
   - Natijalar to'g'ri yoki yo'qligini tekshiring

## 📝 Eslatmalar

1. **Model fayl yo'li:** Model `models/quality_classifier.pkl` da saqlanadi
2. **Model o'lchami:** Odatda 1-5 MB
3. **Ishlash tezligi:** Har bir rasm uchun ~0.1-0.5 soniya
4. **Memory:** ~50-100 MB RAM

## 🔄 Model Yangilash

Agar yangi ma'lumotlar to'plangan bo'lsa:

1. Eski modelni backup qiling
2. Yangi ma'lumotlarni qo'shing
3. Modelni qayta o'qiting
4. Yangi modelni test qiling
5. Agar yaxshiroq bo'lsa, eski modelni almashtiring

## 📚 Qo'shimcha Resurslar

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Random Forest Guide](https://scikit-learn.org/stable/modules/ensemble.html#forest)
- [Feature Engineering](https://en.wikipedia.org/wiki/Feature_engineering)

## 💡 Maslahatlar

1. **Ma'lumotlar sifatiga e'tibor bering** - Yaxshi ma'lumotlar = yaxshi model
2. **Muntazam test qiling** - Modelni doimiy test qilib turing
3. **Version control** - Har bir model versiyasini saqlang
4. **Documentation** - Qanday o'qitilganini yozib qo'ying

## 🎓 Diplom Loyihasi uchun

Diplom loyihasida quyidagilarni ko'rsating:

1. **Dataset tuzilishi** - Qancha rasm, qanday taqsimlangan
2. **Feature extraction** - Qanday xususiyatlar ishlatilgan
3. **Model tanlash** - Nima uchun Random Forest tanlangan
4. **Natijalar** - Accuracy, precision, recall
5. **Taqqoslash** - CV vs ML vs Hybrid natijalari
6. **Xulosa** - ML qo'shilgandan keyin qanday yaxshilanish

---

**Muvaffaqiyatlar!** 🚀

