import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_models(models_dict, X_val, y_val):
    """Mengevaluasi performa seluruh model dan merangkumnya dalam bentuk DataFrame ringkas"""
    records = []
    
    for name, model in models_dict.items():
        preds = model.predict(X_val)
        
        acc = accuracy_score(y_val, preds)
        prec = precision_score(y_val, preds, zero_division=0)
        rec = recall_score(y_val, preds, zero_division=0)
        f1 = f1_score(y_val, preds, zero_division=0)
        
        records.append({
            "Model Name": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1
        })
        
    summary_df = pd.DataFrame(records)
    print("\n" + "="*20 + " TABLE COMPARISON " + "="*20)
    print(summary_df.to_string(index=False))
    print("="*58 + "\n")
    
    return summary_df