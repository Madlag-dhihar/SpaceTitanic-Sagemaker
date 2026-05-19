import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_FILE = BASE_DIR / "data" / "raw" / "train.csv"

def load_and_split_data(random_state=42):
    """Memuat data mentah asli dan membaginya menjadi Train & Validation set"""
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"File data tidak ditemukan di: {TRAIN_FILE}. Pastikan folder data/raw/ sudah terisi.")
        
    df = pd.read_csv(TRAIN_FILE)
    
    
    X = df.drop(columns=['Transported'], errors='ignore')
    y = df['Transported'].astype(int) if 'Transported' in df.columns else None
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    return X_train, X_val, y_train, y_val