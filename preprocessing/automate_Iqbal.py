import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

def preprocess_data(raw_data_path, output_data_path):
    print("Memuat dataset mentah...")
    df = pd.read_csv(raw_data_path)

    # Mengidentifikasi kolom numerik yang akan distandarisasi
    numerical_cols = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls',
                      'Payment Delay', 'Total Spend', 'Last Interaction']

    # Inisialisasi StandardScaler
    scaler = StandardScaler()

    # Membuat salinan DataFrame untuk scaling
    df_scaled = df.copy()
    df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])

    # 3. Encoding Data Kategorikal (One-Hot Encoding)
    categorical_cols = ['Gender', 'Subscription Type', 'Contract Length']
    df_encoded = pd.get_dummies(df_scaled, columns=categorical_cols, drop_first=True)

    # 4. Binning (Pengelompokan Data) pada 'Age'
    # Catatan: Karena 'Age' sudah di-scale di atas, kita gunakan df asli untuk binning agar range usianya tepat
    age_bins = [18, 25, 35, 45, 55, 65]
    age_labels = ['18-24', '25-34', '35-44', '45-54', '55-65']
    df_encoded['Age_Group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, right=False)

    # 5. Encoding Kolom 'Age_Group' (One-Hot Encoding)
    df_final = pd.get_dummies(df_encoded, columns=['Age_Group'], drop_first=True)

    # Drop kolom ID dan Age numerik yang sudah tidak dipakai (jika ada CustomerID)
    columns_to_drop = ['Age']
    if 'CustomerID' in df_final.columns:
        columns_to_drop.append('CustomerID')
        
    df_final = df_final.drop(columns=columns_to_drop, errors='ignore')

    # 6. Memastikan folder output tersedia, lalu simpan ke CSV
    dir_name = os.path.dirname(output_data_path)
    if dir_name:  # Hanya membuat folder jika path memiliki folder induk
        os.makedirs(dir_name, exist_ok=True)
        
    df_final.to_csv(output_data_path, index=False)
    
    print(f"Sukses! Data bersih berhasil diekspor ke: {output_data_path}")
    return df_final

if __name__ == "__main__":
    print("Menjalankan preprocessing data secara otomatis...")

    # Menentukan base directory (lokasi folder 'preprocessing')
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    RAW_DATA_PATH = os.path.join(BASE_DIR, "..", "customer_churn_dataset-testing-master.csv")
    # Menyimpan hasil pembersihan di dalam folder preprocessing menggunakan os.path.join agar aman di Linux/Windows
    OUTPUT_DATA_PATH = os.path.join(BASE_DIR, "customer_churn_preprocessing.csv")

    # Jalankan fungsi
    df_clean = preprocess_data(RAW_DATA_PATH, OUTPUT_DATA_PATH)

    print("Bentuk dataset setelah preprocessing:", df_clean.shape)
    print("\nIntip 5 data teratas hasil preprocessing:")
    print(df_clean.head())