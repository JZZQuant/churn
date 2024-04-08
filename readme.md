### Building MLflow Server

To build the MLflow server, follow these steps:

1. **Pull MLflow Docker Image**: 

    ```bash
    docker pull ghcr.io/mlflow/mlflow:v2.11.3
    ```

2. **Run MLflow Server Container**: 

    ```bash
    docker run -p 5000:5000 ghcr.io/mlflow/mlflow 
    ```

### Training and Registering the Model

To train and register the model, execute the following command:

```bash
python train.py
