import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

def run_preprocessing(input_path, output_path):
    print("Memuat dataset raw...")
    df = pd.read_csv(input_path)

    print("Memulai Data Preprocessing...")
    
    # 1. Drop kolom tidak relevan
    cols_to_drop = ['Header image', 'Website', 'Support url', 'Support email', 'Notes', 'Screenshots']
    df_clean = df.drop(columns=cols_to_drop, errors='ignore')

    # 2. Handling Missing Values
    df_clean['Developers'] = df_clean['Developers'].fillna('Unknown')
    df_clean['Publishers'] = df_clean['Publishers'].fillna('Unknown')
    df_clean = df_clean.dropna(subset=['Price', 'Positive', 'Negative'])

    # 3. Normalisasi
    scaler = MinMaxScaler()
    num_cols = ['Price', 'Peak CCU', 'Positive', 'Negative', 'Average playtime forever']
    # Memastikan kolom ada sebelum di-scale
    existing_num_cols = [col for col in num_cols if col in df_clean.columns]
    df_clean[existing_num_cols] = scaler.fit_transform(df_clean[existing_num_cols])

    # Menyimpan hasil
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    print(f"Dataset bersih berhasil disimpan ke: {output_path}")

if __name__ == "__main__":
    RAW_DATA_PATH = "../steam_dataset_raw/steam_cleaned_2026.csv"
    CLEAN_DATA_PATH = "steam_dataset_preprocessing/steam_ready_to_train.csv"
    
    run_preprocessing(RAW_DATA_PATH, CLEAN_DATA_PATH)