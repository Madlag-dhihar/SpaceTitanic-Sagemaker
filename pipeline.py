import pickle
import tarfile
from pathlib import Path
import mlflow

from src.data import load_and_split_data
from src.models import create_pipeline_factory
from src.evaluate import evaluate_models

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Spaceship_Titanic_MultiModel")

def main():
    print("Executing Stage 1: Loading and Splitting Raw Data...")
    X_train, X_val, y_train, y_val = load_and_split_data()
    
    candidate_types = ["logistic_regression", "random_forest", "xgboost"]
    trained_models = {}
    
    print("\nExecuting Stage 2: Training 3 Candidate Models...")
    for model_type in candidate_types:
        with mlflow.start_run(run_name=f"Train_{model_type}"):
            print(f" -> Training {model_type} pipeline...")
            pipeline = create_pipeline_factory(model_type=model_type)
            pipeline.fit(X_train, y_train)
            
            trained_models[model_type] = pipeline
            mlflow.sklearn.log_model(pipeline, f"{model_type}_model")

    print("\nExecuting Stage 3: Evaluating and Comparing Models...")
    summary_df = evaluate_models(trained_models, X_val, y_val)
    
    
    best_row = summary_df.loc[summary_df['Accuracy'].idxmax()]
    winner_name = best_row['Model Name']
    print(f" WINNER MODEL SELECTED: {winner_name} with Accuracy: {best_row['Accuracy']:.4f}")
    
    best_pipeline = trained_models[winner_name]
    
    # Buat direktori output model jika belum ada
    Path("model").mkdir(exist_ok=True)
    
    
    with open("model/pipeline.pkl", "wb") as f:
        pickle.dump(best_pipeline, f)
    print("Saved winner pipeline artifacts locally at 'model/pipeline.pkl'")

    
    print("\nExecuting Stage 4: Packaging Artifacts into model.tar.gz...")
    with tarfile.open("model.tar.gz", "w:gz") as tar:
        tar.add("model/pipeline.pkl", arcname="pipeline.pkl")
        tar.add("src/inference.py", arcname="code/inference.py")
        tar.add("src/models.py", arcname="code/models.py")
        tar.add("src/requirements.txt", arcname="code/requirements.txt")
        
    print("\n SELESAI! File 'model.tar.gz' berhasil dibuat di root folder.")

if __name__ == "__main__":
    main()