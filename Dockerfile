FROM python:3.8

RUN pip install mlflow
EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0"]
