from sklearn.neural_network import MLPRegressor
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, max_error

# Obtêm os dados de teste
dataframe_test = pd.read_csv('Dataset_Test.csv', delimiter=';', encoding='latin-1')
x_test = pd.DataFrame(dataframe_test.iloc[:, 0:-1].values)
y_test = pd.DataFrame(dataframe_test.iloc[:, -1].values)

# Obtêm os dados de treinamento
dataframe_train = pd.read_csv('Dataset_Train.csv', delimiter=';', encoding='latin-1')
x_train = pd.DataFrame(dataframe_train.iloc[:, 0:-1].values)
y_train = pd.DataFrame(dataframe_train.iloc[:, -1].values)

# Configura e treina a rede neural
regressor = MLPRegressor(max_iter=15000, tol=1e-5, activation='tanh', hidden_layer_sizes=(150,150), solver='lbfgs').fit(x_train, y_train)

# Realiza os testes
y_pred = regressor.predict(x_test)

# Apresenta o desempenho
mse  = mean_squared_error(y_test, y_pred) * 100
print(f'{mse} | {mean_absolute_error(y_test, y_pred)*100} | {max_error(y_test, y_pred)*100} | {r2_score(y_test, y_pred)}')

# Exporta o resultado
results = pd.concat([y_test*91.66296, pd.DataFrame(y_pred*91.66296)], axis=1)
results.columns = ['Real', 'Predição']
results.to_csv(f'Resultados.csv', index=False, encoding='latin-1')