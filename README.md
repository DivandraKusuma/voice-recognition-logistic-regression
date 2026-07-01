# 🎙️ Gender Recognition by Voice

Proyek ini bertujuan untuk **mengidentifikasi gender (Male/Female) berdasarkan fitur suara** menggunakan pendekatan Machine Learning Tradisional dan Deep Learning.

---

## 📋 Daftar Isi

- [Tentang Proyek](#-tentang-proyek)
- [Struktur Proyek](#-struktur-proyek)
- [Prasyarat](#-prasyarat)
- [Langkah 1: Install Dependencies](#langkah-1-install-dependencies)
- [Langkah 2: Pastikan Dataset Tersedia](#langkah-2-pastikan-dataset-tersedia)
- [Langkah 3: Jalankan Notebook](#langkah-3-jalankan-notebook)
- [Langkah 4: Memahami Alur Notebook](#langkah-4-memahami-alur-notebook)
- [Estimasi Waktu](#-estimasi-waktu)
- [Penjelasan Model](#-penjelasan-model)
- [Troubleshooting](#-troubleshooting)

---

## 📖 Tentang Proyek

### Permasalahan
Perbedaan gender sering muncul pada:
- **Pitch** — frekuensi dasar suara
- **Resonansi suara**
- **Distribusi frekuensi**

### Representasi Fitur
| Domain | Fitur |
|--------|-------|
| Domain Waktu | RMS Energy, Zero Crossing Rate |
| Domain Frekuensi | Spectral Centroid, Spectral Bandwidth |
| Representasi Tambahan | Pitch (Fundamental Frequency) |
| Representasi Utama | MFCC (13 koefisien) |

### Model yang Digunakan
| Kategori | Model |
|----------|-------|
| ML Tradisional | Logistic Regression, SVM, Random Forest |
| ML Modern | CNN pada Mel Spectrogram |

### Analisis Perbandingan
Notebook membandingkan 3 variasi fitur:
1. **MFCC saja** (52 fitur)
2. **Pitch saja** (4 fitur)
3. **MFCC + Pitch** (56 fitur)

> **Catatan:** Pitch merupakan fitur yang sangat penting karena secara fisiologis terdapat perbedaan rentang frekuensi dasar antara suara laki-laki (~85–180 Hz) dan perempuan (~165–255 Hz).

---

## 📁 Struktur Proyek

```
voice-recognition-logistic-regression-main/
│
├── dataset/                          # Folder dataset audio
│   ├── male/                         # File .wav/.mp3 suara laki-laki
│   └── female/                       # File .wav/.mp3 suara perempuan
│
├── models/                           # Modul model ML & DL
│   ├── __init__.py
│   ├── logistic_model.py             # Logistic Regression + StandardScaler
│   ├── svm_model.py                  # SVM (RBF kernel) + StandardScaler
│   ├── rf_model.py                   # Random Forest + StandardScaler
│   └── CNN_model.py                  # CNN 2D (TensorFlow/Keras)
│
├── features/                         # Modul ekstraksi fitur
│   ├── feature_extractor_ml.py       # Ekstraksi MFCC, Pitch, dll.
│   └── feature_extractor_dl.py       # Ekstraksi Mel Spectrogram
│
├── prepocessing/                     # Modul preprocessing audio
│   ├── __init__.py
│   ├── voice_preprocessing_ml.py     # Preprocessing untuk ML
│   └── voice_preprocessing_dl.py     # Preprocessing untuk DL (fixed duration)
│
├── dataset_fitur_audio.csv           # Dataset fitur yang sudah diekstrak
├── gender.ipynb                      # Notebook utama (analisis & eksperimen)
├── requirements.txt                  # Daftar dependencies Python
└── README.md                         # File ini
```

---

## ⚙️ Prasyarat

- **Python 3.10** atau lebih baru
- **pip** (package manager Python)
- **Jupyter Notebook** atau **VS Code** dengan ekstensi Python/Jupyter
- *(Opsional)* GPU dengan CUDA untuk mempercepat training CNN

Cek versi Python:
```bash
python --version
```

---

## Langkah 1: Install Dependencies

### Opsi A — Install Semua (ML + Deep Learning)

```bash
pip install -r requirements.txt
```

> ⚠️ **Catatan:** TensorFlow cukup besar (~500MB+). Pastikan koneksi internet stabil.

### Opsi B — Install ML Tradisional Saja (Tanpa TensorFlow)

Jika hanya ingin menjalankan bagian ML tradisional terlebih dahulu:

```bash
pip install numpy pandas librosa soundfile scikit-learn matplotlib seaborn tqdm joblib
```

> Bagian CNN di notebook bisa dijalankan nanti setelah TensorFlow terinstall.

### Verifikasi Instalasi

Jalankan perintah berikut untuk memastikan semua library terinstall:

```bash
python -c "import numpy, pandas, librosa, sklearn, matplotlib, seaborn; print('Semua library ML berhasil diimport!')"
```

Untuk verifikasi TensorFlow:

```bash
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} berhasil diimport!')"
```

---

## Langkah 2: Pastikan Dataset Tersedia

### Untuk ML Tradisional

Pastikan file `dataset_fitur_audio.csv` ada di root folder proyek. File ini berisi fitur-fitur audio yang sudah diekstrak (MFCC, Pitch, RMS, ZCR, dll.) dan sudah tersedia di repository.

```bash
# Cek apakah file ada
dir dataset_fitur_audio.csv
```

### Untuk CNN (Deep Learning)

Pastikan file audio asli tersedia di dalam folder `dataset/`:

```
dataset/
├── male/        ← berisi file .wav atau .mp3 suara laki-laki
└── female/      ← berisi file .wav atau .mp3 suara perempuan
```

Jika folder `dataset/male/` dan `dataset/female/` kosong, download dataset dari:
> 🔗 https://www.kaggle.com/datasets/murtadhanajim/gender-recognition-by-voiceoriginal

---

## Langkah 3: Jalankan Notebook

### Cara A: Menggunakan Jupyter Notebook (Rekomendasi)

1. Install Jupyter (jika belum):
   ```bash
   pip install jupyter
   ```

2. Jalankan Jupyter:
   ```bash
   jupyter notebook gender.ipynb
   ```

3. Browser akan terbuka otomatis. Jalankan cell:
   - **Satu per satu:** tekan `Shift + Enter` pada setiap cell
   - **Semua sekaligus:** klik menu **Cell → Run All**

### Cara B: Menggunakan VS Code

1. Buka file `gender.ipynb` di VS Code
2. VS Code akan otomatis mendeteksi file sebagai notebook
3. Pilih **Python kernel/interpreter** di pojok kanan atas
4. Klik tombol **Run All** (▶▶) di toolbar atas, atau jalankan per cell dengan tombol ▶️

### Cara C: Menggunakan Google Colab

1. Upload semua file proyek ke Google Drive
2. Buka `gender.ipynb` dengan Google Colab
3. Sesuaikan path dataset jika diperlukan
4. Klik **Runtime → Run All**

---

## Langkah 4: Memahami Alur Notebook

Notebook `gender.ipynb` terbagi dalam beberapa bagian utama:

### Bagian A — Setup & Data Loading (Cell 1–3)

| Cell | Isi | Output |
|------|-----|--------|
| 1 | Judul & deskripsi proyek | — |
| 2 | Import semua library & model | `"All imports successful!"` |
| 3 | Load `dataset_fitur_audio.csv` | Tabel preview & shape data |

### Bagian B — Exploratory Data Analysis (Cell 4–5)

| Cell | Isi | Output |
|------|-----|--------|
| 4 | Visualisasi distribusi gender & histogram pitch | 2 grafik (bar chart & histogram) |
| 5 | Definisi kolom fitur (MFCC, Pitch, MFCC+Pitch) | Jumlah fitur per variasi |

### Bagian C — Eksperimen ML Tradisional (Cell 6–11)

| Cell | Isi | Output |
|------|-----|--------|
| 6 | Train/Test split (80:20) | Jumlah data training & testing |
| 7 | **9 eksperimen** (3 model × 3 variasi fitur) | Akurasi setiap kombinasi |
| 8 | Tabel komparasi hasil | Tabel styled: Accuracy, Precision, Recall, F1 |
| 9 | Bar chart perbandingan akurasi | Grafik batang 3 model × 3 fitur |
| 10 | Bar chart perbandingan F1-Score | Grafik batang F1-Score |
| 11 | **Analisis kontribusi Pitch** | Heatmap + bukti empiris |

**9 Kombinasi Eksperimen:**

| No | Model | Fitur |
|----|-------|-------|
| 1 | Logistic Regression | MFCC |
| 2 | Logistic Regression | Pitch |
| 3 | Logistic Regression | MFCC + Pitch |
| 4 | SVM | MFCC |
| 5 | SVM | Pitch |
| 6 | SVM | MFCC + Pitch |
| 7 | Random Forest | MFCC |
| 8 | Random Forest | Pitch |
| 9 | Random Forest | MFCC + Pitch |

### Bagian D — CNN Deep Learning (Cell 12–17)

> ⚠️ Bagian ini membutuhkan **TensorFlow** dan **file audio asli** di folder `dataset/`.

| Cell | Isi | Output |
|------|-----|--------|
| 12 | Setup & scan file audio | Jumlah file audio ditemukan |
| 13 | Ekstraksi Mel Spectrogram | Shape array & distribusi label |
| 14 | Train/Val/Test split untuk CNN | Jumlah data per split |
| 15 | Build model CNN & summary | Arsitektur model |
| 16 | Training CNN | Progress per epoch |
| 17 | Plot training history | Grafik loss & accuracy curve |

### Bagian E — Hasil Akhir (Cell 18–20)

| Cell | Isi | Output |
|------|-----|--------|
| 18 | Evaluasi CNN pada test set | Accuracy, Precision, Recall, F1 |
| 19 | Tabel ranking semua model | Tabel final ML + CNN |
| 20 | Grafik ranking horizontal | Bar chart perbandingan semua model |

---

## ⏱️ Estimasi Waktu

| Bagian | Estimasi Waktu |
|--------|----------------|
| Cell 1–5 (Load & EDA) | ~10 detik |
| Cell 6–7 (9 eksperimen ML) | ~1–3 menit |
| Cell 8–11 (Visualisasi & analisis) | ~5 detik |
| Cell 12–13 (Ekstraksi Mel Spectrogram) | ~5–15 menit* |
| Cell 14–17 (Training CNN) | ~5–20 menit* |
| Cell 18–20 (Evaluasi & ranking) | ~10 detik |
| **Total** | **~15–40 menit** |

*\*Tergantung jumlah file audio, spesifikasi hardware, dan apakah menggunakan GPU.*

> **💡 Tips:** Jika hanya ingin melihat hasil ML tradisional, cukup jalankan **Cell 1 sampai 11** saja. Ini hanya memakan waktu ~3–5 menit.

---

## 🧠 Penjelasan Model

### Logistic Regression (`models/logistic_model.py`)
- Classifier linear untuk klasifikasi biner
- Menggunakan **StandardScaler** dalam pipeline
- Cocok sebagai baseline model

### SVM (`models/svm_model.py`)
- Support Vector Machine dengan kernel **RBF**
- Menggunakan **StandardScaler** dalam pipeline
- Efektif untuk data berdimensi tinggi

### Random Forest (`models/rf_model.py`)
- Ensemble dari 200 decision trees
- Menggunakan **StandardScaler** dalam pipeline
- Robust terhadap overfitting

### CNN (`models/CNN_model.py`)
- 3 blok konvolusi: Conv2D → BatchNorm → MaxPool → Dropout
- Filter: 32 → 64 → 128
- Output: Dense(1, sigmoid) untuk klasifikasi biner
- Callbacks: EarlyStopping & ReduceLROnPlateau


## 📄 Lisensi

Proyek ini dibuat untuk keperluan edukasi dan penelitian.

Dataset: [Gender Recognition by Voice (Kaggle)](https://www.kaggle.com/datasets/murtadhanajim/gender-recognition-by-voiceoriginal)
