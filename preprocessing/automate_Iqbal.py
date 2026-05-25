import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(raw_data_path):
    # 1. Memuat Dataset
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
    age_bins = [18, 25, 35, 45, 55, 65]
    age_labels = ['18-24', '25-34', '35-44', '45-54', '55-65']
    df_encoded['Age_Group'] = pd.cut(df_encoded['Age'], bins=age_bins, labels=age_labels, right=False)

    # 5. Encoding Kolom 'Age_Group' (One-Hot Encoding)
    df_final = pd.get_dummies(df_encoded, columns=['Age_Group'], drop_first=True)

    # 6. Definisikan fitur (X) dan variabel target (y)
    X = df_final.drop(['CustomerID', 'Churn', 'Age'], axis=1)
    y = df_final['Churn']

    return X, y

if __name__ == "__main__":
    print("Menjalankan preprocessing data secara langsung...")

    # Tentukan jalur ke dataset mentah Anda
    # Sesuaikan path ini jika Anda menjalankannya di lingkungan lokal atau GitHub
    raw_data_path = "customer_churn_dataset-testing-master.csv"

    # Panggil fungsi preprocessing otomatis
    X_preprocessed, y_target = preprocess_data(raw_data_path)

    print("Bentuk X setelah preprocessing:", X_preprocessed.shape)
    print("Bentuk y setelah preprocessing:", y_target.shape)

    print("\nHead dari X_preprocessed:")
    # Menggunakan print() daripada display() karena ini untuk skrip Python biasa
    print(X_preprocessed.head())

    print("\nHead dari y_target:")
    print(y_target.head())