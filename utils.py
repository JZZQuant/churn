from sklearn.metrics import classification_report
import numpy as np
# Deploy the model and preprocessing pipeline as a serving endpoint
from mlflow.pyfunc import PythonModel, load_model

# Define evaluation metrics
def evaluate_model(y_true, y_pred):
    return classification_report(y_true, y_pred, output_dict=True)

def cross_validate_fold(model, X, y, train_idx, test_idx, categorical_features):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Get indexes of columns with string data type
    categorical_features = X.select_dtypes(exclude=['int','float']).columns.to_list()

    # Fit the model
    model.fit(X_train, y_train, cat_features=categorical_features)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    f1_score = evaluate_model(y_test, y_pred)
    return f1_score

class CatBoostModel(PythonModel):
    def load_context(self, context):
        self.model = load_model(context.artifacts["model"])

    def predict(self, context, model_input):
        # Apply preprocessing pipeline to input data
        preprocessed_input = context.artifacts["preprocessor"].transform(model_input)
        return self.model.predict(preprocessed_input)