import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report

prod_file = '../data/processed/prediction_prod.parquet'
dev_file = '../data/processed/prediction_test.parquet'

############################################ SIDE BAR TITLE
st.sidebar.title('Painel de Controle')
st.sidebar.markdown(f"""
 Preditor de arremessos do "Black Mamba", se acertou ou errou a cesta.
""")

df_prod = pd.read_parquet(prod_file)
df_dev = pd.read_parquet(dev_file)

# st.write(df_prod)
# st.write(df_dev)

fignum = plt.figure(figsize=(6,4))
# Saida do modelo dados dev
sns.distplot(df_dev.prediction_score_1,
             label='Teste',
             ax = plt.gca())

# Saida do modelo dados prod
sns.distplot(df_prod.predict_score,
             label='Producao',
             ax = plt.gca())

plt.title('Monitoramento Desvio de Dados da Saída do Modelo')
plt.ylabel('Densidade Estimada')
plt.xlabel('Probabilidade de acerto de cesta')
plt.xlim((0,1))
plt.grid(True)
plt.legend(loc='best')

st.pyplot(fignum)

# Gerando os relatórios de classificação como dicionários
df_prod = df_prod.dropna(subset=['shot_made_flag'])
df_prod['prediction_label'] = (df_prod['predict_score'] >= 0.5).astype(int)

report_dev = classification_report(df_dev.shot_made_flag, df_dev.prediction_label, output_dict=True)
report_prod = classification_report(df_prod.shot_made_flag, df_prod.prediction_label, output_dict=True)

# Convertendo os dicionários em DataFrames
report_df_dev = pd.DataFrame(report_dev).transpose().add_suffix('_dev')
report_df_prod = pd.DataFrame(report_prod).transpose().add_suffix('_prod')

# Juntando os DataFrames para comparação
report_df_combined = report_df_dev.join(report_df_prod)

# Opção para ajustar a exibição floating point para melhor leitura
pd.options.display.float_format = '{:0.2f}'.format

# Exibindo a tabela combinada no Streamlit
st.dataframe(report_df_combined)




