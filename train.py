import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from catboost import CatBoostClassifier
from preprocessing import pre_processing
from utils import cross_validate_fold,CatBoostModel
import numpy as np
import mlflow
import pickle


def main():
    # Load data
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("catboost")
    df = pd.read_csv("data\\train.data",sep="\t")
    target = pd.read_csv("data\\train_churn.labels.txt",  header=None)
    df.dropna(how='all', axis=1, inplace=True) 

    # Preprocessing
    p_f = "preprocessing_pipeline.pkl"
    X,y,categorical_features = pre_processing(df,target,p_f)

    params=dict(
        iterations=1000,
                            depth=5,
                            learning_rate=0.05,
                            loss_function='Logloss',
                            eval_metric='F1',
                            l2_leaf_reg=8,
                            verbose=400,
                            bagging_temperature=8,
                            border_count=32,
                            random_strength=16,
                            class_weights=[1, 22],
                            random_seed=42
    )

    # Initialize CatBoost classifier
    model = CatBoostClassifier(**params)

    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    # Perform parallelized 10-fold cross-validation
    f1_scores = [cross_validate_fold(model, X, y, train_idx, test_idx, categorical_features) for train_idx, test_idx in kfold.split(X, y) ]
    avg_f1_score = np.mean([score['1']['f1-score'] for score in f1_scores])
    print("Average F1-score:", avg_f1_score)
    avg_precission = np.mean([score['1']['precision'] for score in f1_scores])
    print("Average Precission:", avg_precission)
    avg_recall = np.mean([score['1']['recall'] for score in f1_scores])
    print("Average Recall:", avg_recall)

    # Log parameters and metrics with MLflow
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics({"F1-Score":avg_f1_score,"Precision": avg_precission,"Recall": avg_recall})

        # Log the trained model
        mlflow.sklearn.log_model(model, "catboost_model")

        # Register the model in the MLflow Model Registry
        model_uri = "runs:/{}/catboost_model".format(mlflow.active_run().info.run_id)
        registered_model = mlflow.register_model(model_uri, "CatBoostClassifier")
        mlflow.log_artifact(p_f, artifact_path="preprocessing")
        print("Model version:", registered_model.version)


    # # Save the model and preprocessing pipeline
    # mlflow.pyfunc.save_model(path="model", python_model=CatBoostModel(), artifacts={"model": model_uri})


if __name__ == "__main__":
    main()
