import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def raw_feature_engineering_transformer(df):
    """
    Fungsi transformasi tangguh yang bisa menerima data mentah asli (saat training)
    maupun data lepasan dari UI/Streamlit tanpa memicu error crash.
    """
    df = df.copy()
    
    
    if 'Cabin' in df.columns:
        df['Deck'] = df['Cabin'].apply(lambda x: x.split('/')[0] if pd.notna(x) else 'Unknown')
        df['Cabin_num'] = df['Cabin'].apply(lambda x: x.split('/')[1] if pd.notna(x) else -1).astype(float)
        df['Side'] = df['Cabin'].apply(lambda x: x.split('/')[2] if pd.notna(x) else 'Unknown')
    else:
        for col in ['Deck', 'Side']:
            if col not in df.columns: df[col] = 'Unknown'
        if 'Cabin_num' not in df.columns: df['Cabin_num'] = -1.0

    
    if 'PassengerId' in df.columns:
        df['Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
        df['Group_size'] = df.groupby('Group')['Group'].transform('count')
        df['Solo'] = (df['Group_size'] == 1).astype(int)
    else:
        if 'Group_size' not in df.columns: df['Group_size'] = 1
        if 'Solo' not in df.columns: df['Solo'] = 1

    
    if 'Name' in df.columns:
        df['LastName'] = df['Name'].apply(lambda x: x.split()[-1] if pd.notna(x) else 'Unknown')
        df['Family_size'] = df.groupby('LastName')['LastName'].transform('count')
    else:
        if 'Family_size' not in df.columns: df['Family_size'] = 1

    
    spending_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for col in spending_cols:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].fillna(0.0)
            
    df['TotalSpending'] = df[spending_cols].sum(axis=1)
    df['HasSpending'] = (df['TotalSpending'] > 0).astype(int)
    df['NoSpending'] = (df['TotalSpending'] == 0).astype(int)
    
    for col in spending_cols:
        df[f'{col}_ratio'] = df[col] / (df['TotalSpending'] + 1)

    
    if 'Age' not in df.columns:
        df['Age'] = 28.0  # median default
    else:
        df['Age'] = df['Age'].fillna(28.0)
        
    df['Age_group'] = pd.cut(
        df['Age'], bins=[0, 12, 18, 30, 50, 100],
        labels=['Child', 'Teen', 'Young_Adult', 'Adult', 'Senior']
    ).astype(str).fillna('Young_Adult')
    
    df['Age_missing'] = df['Age'].isna().astype(int)
    df['CryoSleep_missing'] = df['CryoSleep'].isna().astype(int) if 'CryoSleep' in df.columns else 0

    
    cat_cols = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Age_group']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)
        else:
            df[col] = 'Unknown'

    
    num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Cabin_num', 
                'Group_size', 'Solo', 'Family_size', 'TotalSpending', 'HasSpending', 'NoSpending', 
                'Age_missing', 'CryoSleep_missing', 'RoomService_ratio', 'FoodCourt_ratio', 
                'ShoppingMall_ratio', 'Spa_ratio', 'VRDeck_ratio']
                
    return df[cat_cols + num_cols]


def create_pipeline_factory(model_type="logistic_regression", random_state=42):
    """Membuat end-to-end Pipeline untuk salah satu dari 3 kandidat model"""
    
    categorical_features = ['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side', 'Age_group']
    
    
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    
    preprocessor = ColumnTransformer(
        transformers=[('cat_encoder', encoder, categorical_features)],
        remainder='passthrough'
    )
    
    
    if model_type == "logistic_regression":
        classifier = LogisticRegression(C=0.1, max_iter=500, random_state=random_state)
    elif model_type == "random_forest":
        classifier = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=random_state)
    elif model_type == "xgboost":
        classifier = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=random_state, eval_metric='logloss')
    else:
        raise ValueError(f"Model {model_type} tidak didukung.")

    
    pipeline = Pipeline([
        ('feature_engineering', FunctionTransformer(raw_feature_engineering_transformer)),
        ('categorical_encoding', preprocessor),
        ('classifier', classifier)
    ])
    
    return pipeline