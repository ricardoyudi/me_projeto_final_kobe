from sklearn.metrics import log_loss, f1_score, confusion_matrix
import pandas as pd
import mlflow
import seaborn as sns
import matplotlib.pyplot as plt

mlflow.set_tracking_uri("sqlite:///mlruns.db")

experiment_name = 'Projeto Kobe'
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    experiment_id = mlflow.create_experiment(experiment_name)
    experiment = mlflow.get_experiment(experiment_id)
experiment_id = experiment.experiment_id

print('== Pipeline Aplicacao Kobe ==')

with mlflow.start_run(experiment_id=experiment_id, run_name = 'PipelineAplicacao'):

    model_uri = f"models:/modelo_kobe@staging"
    loaded_model = mlflow.sklearn.load_model(model_uri)
    
    data_prod = pd.read_parquet('../data/raw/dataset_kobe_prod.parquet')
    data_prod = data_prod.dropna(subset=['shot_made_flag'])
    
    columns = ['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']
    
    Y = loaded_model.predict_proba(data_prod[columns].drop('shot_made_flag', axis=1))[:, 1]
    
    data_prod['predict_score'] = Y
    
    data_prod.to_parquet('../data/processed/prediction_prod.parquet')
    mlflow.log_artifact('../data/processed/prediction_prod.parquet')
    
    y_true = data_prod['shot_made_flag']
    y_pred_proba = data_prod['predict_score']

    # Cálculo do log loss
    log_loss_metric = log_loss(y_true, y_pred_proba)
    mlflow.log_metric("log_loss", log_loss_metric)

    # Calculando a predição em classes para usar no f1_score
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Cálculo do f1_score
    f1_score_metric = f1_score(y_true, y_pred)
    mlflow.log_metric("f1_score", f1_score_metric)
    
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Plottando a matriz de confusão usando Seaborn
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=["Não Acertou", "Acertou"], yticklabels=["Não Acertou", "Acertou"])
    plt.title('Matriz de Confusão - Produção')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.savefig('confusion_matrix_prod.png')
    plt.close()