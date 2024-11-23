# 🌟 Proyek SEHATI 

Welcome to our GitHub repository! 🎉 We are a team working on Capstone Project at Bangkit and we build an AI-powered health monitoring application designed to support users.  
---

## 📝 Tentang Proyek

- **Project Title:** SEHATI  
- **Short Description:** ~

---

## 🛠️ Teknologi yang Digunakan

- **Machine Learning:** ~
- **Cloud Computing:** ~
- **Mobile Development:** ~
- **Other:** ~

---

## 🤝 Our Team

Kami terdiri dari anggota dengan keahlian di bidang yang beragam untuk menyukseskan proyek ini:

- **Machine Learning Engineers:**
  - [Nama anggota 1]
  - [Nama anggota 2]
  - [Nama anggota 3]
- **Cloud Computing Engineers:**
  - [Nama anggota 4]
  - [Nama anggota 5]
- **Mobile Developer:**
  - [Nama anggota 6]

---

# Dokumentasi Penggunaan Model TFLite untuk Prediksi Penyakit

---

Dokumentasi ini menjelaskan bagaimana memuat model TFLite, memproses input pengguna, dan mendapatkan hasil prediksi.

---

## **1. Daftar Fitur (Gejala)**

Dataset memiliki 131 fitur (gejala) yang digunakan untuk melatih model. Pastikan daftar fitur ini sesuai dengan urutan dan jumlah fitur yang digunakan saat melatih model.

```python
# Daftar fitur lengkap dari dataset (131 fitur)
features = ['gatal', 'ruam kulit', 'erupsi kulit nodal', 'bersin terus menerus', 'menggigil', 'kedinginan', 'nyeri sendi', 'nyeri perut', 'asam lambung', 'luka di lidah', 'penyusutan otot', 'muntah', 'sensasi terbakar saat buang air kecil', 'bercak saat buang air kecil', 'kelelahan', 'penambahan berat badan', 'kecemasan', 'tangan dan kaki dingin', 'perubahan mood', 'penurunan berat badan', 'gelisah', 'lesu', 'bercak di tenggorokan', 'gula darah tidak teratur', 'batuk', 'demam tinggi', 'mata cekung', 'sesak napas', 'berkeringat', 'dehidrasi', 'gangguan pencernaan', 'sakit kepala', 'kulit kekuningan', 'urine gelap', 'mual', 'hilang nafsu makan', 'nyeri di belakang mata', 'sakit punggung', 'sembelit', 'sakit perut', 'diare', 'demam ringan', 'urine kuning', 'mata kuning', 'gagal hati akut', 'pembengkakan perut', 'kelenjar getah bening bengkak', 'kelelahan', 'penglihatan buram', 'dahak', 'iritasi tenggorokan', 'mata merah', 'tekanan sinus', 'pilek', 'hidung tersumbat', 'nyeri dada', 'kelemahan di anggota tubuh', 'detak jantung cepat', 'nyeri saat buang air besar', 'nyeri di area anus', 'tinja berdarah', 'iritasi di anus', 'nyeri leher', 'pusing', 'kram', 'memar', 'obesitas', 'kaki bengkak', 'pembuluh darah bengkak', 'wajah dan mata bengkak', 'tiroid membesar', 'kuku rapuh', 'pembengkakan ekstremitas', 'rasa lapar berlebihan', 'kontak di luar nikah', 'bibir kering dan bertingling', 'bicara cadel', 'nyeri lutut', 'nyeri sendi pinggul', 'kelemahan otot', 'leher kaku', 'sendi bengkak', 'kekakuan pergerakan', 'gerakan berputar', 'kehilangan keseimbangan', 'ketidakstabilan', 'kelemahan satu sisi tubuh', 'hilang indra penciuman', 'ketidaknyamanan kandung kemih', 'bau urine menyengat', 'rasa ingin buang air kecil terus', 'gas keluar', 'gatal dalam', 'penampilan toksik', 'depresi', 'iritabilitas', 'nyeri otot', 'altered sensorium', 'bintik merah di tubuh', 'nyeri perut', 'menstruasi tidak normal', 'bercak dischromic', 'mata berair', 'nafsu makan meningkat', 'poliuria', 'riwayat keluarga', 'dahak lendir', 'dahak berkarat', 'kurang konsentrasi', 'gangguan penglihatan', 'menerima transfusi darah', 'menerima suntikan tidak steril', 'koma', 'pendarahan lambung', 'pembesaran perut', 'riwayat konsumsi alkohol', 'kelebihan cairan', 'darah di dahak', 'vena menonjol di betis', 'palpitasi', 'nyeri saat berjalan', 'jerawat bernanah', 'komedo', 'bekas luka', 'kulit mengelupas', 'debu seperti perak', 'lekukan kecil di kuku', 'kuku meradang', 'lepuh', 'luka merah di hidung', 'kerak kuning mengalir']

# Validasi jumlah fitur
assert len(features) == 131, f"features harus memiliki 131 elemen, saat ini memiliki: {len(features)}"

```

---

## **2. Daftar Kelas Penyakit**

Daftar kelas penyakit yang akan diprediksi oleh model. Setiap indeks mewakili kelas tertentu.

```python
# Nama kelas penyakit
disease_classes = ['Infeksi Jamur', 'Alergi', 'Penyakit Refluks Gastroesofagus (GERD)',
 'Cholestasis Kronis', 'Reaksi Obat', 'Penyakit Tukak Lambung', 'AIDS',
 'Diabetes', 'Gastroenteritis', 'Asma Bronkial', 'Hypertension', 'Migrain',
 'Spondilosis Servikal', 'Paralisia (perdarahan otak)', 'Penyakit Kuning',
 'Malaria', 'Cacar Air', 'Dengue', 'Tifus', 'Hepatitis A', 'Hepatitis B',
 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Hepatitis Alkoholik',
 'Tuberkulosis', 'Flu Biasa', 'Pneumonia', 'Wasir Dimorfik',
 'Serangan Jantung', 'Varises', 'Hipotiroidisme', 'Hipertiroidisme',
 'Hipoglikemia', 'Osteoartritis', 'Artritis', 'Vertigo', 'Jerawat',
 'Infeksi Saluran Kemih', 'Psoriasis', 'Impetigo']

```

---

## **3. Fungsi Pemrosesan Input**

Fungsi ini mengubah input pengguna (gejala) menjadi array biner yang dapat digunakan oleh model.

```python
def process_input(user_input, features):
    """
    user_input: List fitur (gejala) dari user.
    features: List lengkap semua fitur dalam dataset.
    """
    input_array = [1 if feature in user_input else 0 for feature in features]
    return np.array([input_array], dtype=np.float32)  # Tambahkan dimensi batch

```

---

## **4. Contoh Penggunaan Fungsi Pemrosesan**

Gunakan fungsi `process_input` untuk mengubah daftar gejala pengguna menjadi array biner berdimensi `(1, 131)`.

```python
# Contoh input user
user_input = ["gatal", "ruam kulit", "bercak dischromic"]

# Proses input menjadi array biner
input_data = process_input(user_input, features)
print("Input Array Shape:", input_data.shape)  # Pastikan (1, 131)

```

Output:

```
Input Array Shape: (1, 131)

```

---

## **5. Memuat dan Menjalankan Model TFLite**

### **Langkah-Langkah**:

1. Muat model TFLite menggunakan `tensorflow.lite.Interpreter`.
2. Alokasikan tensor untuk input dan output.
3. Masukkan data input ke model.
4. Jalankan prediksi dan ambil hasil.

```python
# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Detail input dan output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Masukkan data input
interpreter.set_tensor(input_details[0]['index'], input_data)

# Jalankan prediksi
interpreter.invoke()

# Ambil hasil prediksi
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_class = np.argmax(output_data)

# Hasil prediksi
predicted_disease = disease_classes[predicted_class]
print("Predicted Class:", predicted_class)
print("Prediction Probabilities:", output_data)
print("Predicted Disease:", predicted_disease)

```

---

## **6. Hasil Prediksi**

Setelah kode di atas dijalankan, Anda akan mendapatkan:

- **Kelas Prediksi (Predicted Class):** Indeks kelas yang diprediksi.
- **Probabilitas (Prediction Probabilities):** Probabilitas untuk setiap kelas.
- **Nama Penyakit (Predicted Disease):** Nama kelas penyakit yang diprediksi.

**Contoh Output:**

```
Predicted Class: 3
Prediction Probabilities: [[0.1 0.2 0.1 0.6 0.0 ...]]
Predicted Disease: Cholestasis Kronis

```

---

## **7. Implementasi Lengkap**

Gabungan semua langkah di atas menjadi satu file Python:

```python
import numpy as np
import tensorflow as tf

# Daftar fitur lengkap dari dataset (131 fitur)
features = ['gatal', 'ruam kulit', 'erupsi kulit nodal', 'bersin terus menerus', 'menggigil', 'kedinginan', 'nyeri sendi', 'nyeri perut', 'asam lambung', 'luka di lidah', 'penyusutan otot', 'muntah', 'sensasi terbakar saat buang air kecil', 'bercak saat buang air kecil', 'kelelahan', 'penambahan berat badan', 'kecemasan', 'tangan dan kaki dingin', 'perubahan mood', 'penurunan berat badan', 'gelisah', 'lesu', 'bercak di tenggorokan', 'gula darah tidak teratur', 'batuk', 'demam tinggi', 'mata cekung', 'sesak napas', 'berkeringat', 'dehidrasi', 'gangguan pencernaan', 'sakit kepala', 'kulit kekuningan', 'urine gelap', 'mual', 'hilang nafsu makan', 'nyeri di belakang mata', 'sakit punggung', 'sembelit', 'sakit perut', 'diare', 'demam ringan', 'urine kuning', 'mata kuning', 'gagal hati akut', 'pembengkakan perut', 'kelenjar getah bening bengkak', 'kelelahan', 'penglihatan buram', 'dahak', 'iritasi tenggorokan', 'mata merah', 'tekanan sinus', 'pilek', 'hidung tersumbat', 'nyeri dada', 'kelemahan di anggota tubuh', 'detak jantung cepat', 'nyeri saat buang air besar', 'nyeri di area anus', 'tinja berdarah', 'iritasi di anus', 'nyeri leher', 'pusing', 'kram', 'memar', 'obesitas', 'kaki bengkak', 'pembuluh darah bengkak', 'wajah dan mata bengkak', 'tiroid membesar', 'kuku rapuh', 'pembengkakan ekstremitas', 'rasa lapar berlebihan', 'kontak di luar nikah', 'bibir kering dan bertingling', 'bicara cadel', 'nyeri lutut', 'nyeri sendi pinggul', 'kelemahan otot', 'leher kaku', 'sendi bengkak', 'kekakuan pergerakan', 'gerakan berputar', 'kehilangan keseimbangan', 'ketidakstabilan', 'kelemahan satu sisi tubuh', 'hilang indra penciuman', 'ketidaknyamanan kandung kemih', 'bau urine menyengat', 'rasa ingin buang air kecil terus', 'gas keluar', 'gatal dalam', 'penampilan toksik', 'depresi', 'iritabilitas', 'nyeri otot', 'altered sensorium', 'bintik merah di tubuh', 'nyeri perut', 'menstruasi tidak normal', 'bercak dischromic', 'mata berair', 'nafsu makan meningkat', 'poliuria', 'riwayat keluarga', 'dahak lendir', 'dahak berkarat', 'kurang konsentrasi', 'gangguan penglihatan', 'menerima transfusi darah', 'menerima suntikan tidak steril', 'koma', 'pendarahan lambung', 'pembesaran perut', 'riwayat konsumsi alkohol', 'kelebihan cairan', 'darah di dahak', 'vena menonjol di betis', 'palpitasi', 'nyeri saat berjalan', 'jerawat bernanah', 'komedo', 'bekas luka', 'kulit mengelupas', 'debu seperti perak', 'lekukan kecil di kuku', 'kuku meradang', 'lepuh', 'luka merah di hidung', 'kerak kuning mengalir']

# Validasi jumlah fitur
assert len(features) == 131, f"features harus memiliki 131 elemen, saat ini memiliki: {len(features)}"

# Daftar kelas penyakit
disease_classes = ['Infeksi Jamur', 'Alergi', 'Penyakit Refluks Gastroesofagus (GERD)',
 'Cholestasis Kronis', 'Reaksi Obat', 'Penyakit Tukak Lambung', 'AIDS',
 'Diabetes', 'Gastroenteritis', 'Asma Bronkial', 'Hypertension', 'Migrain',
 'Spondilosis Servikal', 'Paralisia (perdarahan otak)', 'Penyakit Kuning',
 'Malaria', 'Cacar Air', 'Dengue', 'Tifus', 'Hepatitis A', 'Hepatitis B',
 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Hepatitis Alkoholik',
 'Tuberkulosis', 'Flu Biasa', 'Pneumonia', 'Wasir Dimorfik',
 'Serangan Jantung', 'Varises', 'Hipotiroidisme', 'Hipertiroidisme',
 'Hipoglikemia', 'Osteoartritis', 'Artritis', 'Vertigo', 'Jerawat',
 'Infeksi Saluran Kemih', 'Psoriasis', 'Impetigo']

# Fungsi untuk memproses input
def process_input(user_input, features):
    input_array = [1 if feature in user_input else 0 for feature in features]
    return np.array([input_array], dtype=np.float32)

# Contoh input pengguna
user_input = ["gatal", "ruam kulit", "bercak dischromic"]

# Proses input
input_data = process_input(user_input, features)
print("Input Array Shape:", input_data.shape)  # Pastikan (1, 131)

# Load model TFLite
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Detail input dan output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Masukkan data input
interpreter.set_tensor(input_details[0]['index'], input_data)

# Jalankan prediksi
interpreter.invoke()

# Ambil hasil prediksi
output_data = interpreter.get_tensor(output_details[0]['index'])
predicted_class = np.argmax(output_data)

# Hasil prediksi
predicted_disease = disease_classes[predicted_class]
print("Predicted Class:", predicted_class)
print("Prediction Probabilities:", output_data)
print("Predicted Disease:", predicted_disease)

```

---

**Catatan Akhir:**
Sesuaikan form input dinamis saat mengimplementasikan di Mobile dan Cloud
