from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from breast_data import get_data

df = get_data()
X = df.drop(columns=['target'])
y = df['target']

param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.01],
}

grid_search = GridSearchCV(MLPClassifier(max_iter=500), param_grid, cv=5)
grid_search.fit(X, y)

print("Best Parameters:", grid_search.best_params_)
