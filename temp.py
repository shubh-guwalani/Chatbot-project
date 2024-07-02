from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from umap import UMAP
from hdbscan import HDBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import silhouette_score

# Load sample data
data, labels = load_digits(return_X_y=True)

# Define the objective function
def objective(params):
    umap_model = UMAP(n_neighbors=int(params['n_neighbors']),
                      n_components=int(params['n_components']),
                      min_dist=params['min_dist'],
                      metric=params['metric'],
                      random_state=42)
    embedding = umap_model.fit_transform(data)
    
    hdbscan_model = HDBSCAN(min_cluster_size=int(params['min_cluster_size']),
                            metric=params['metric'],
                            cluster_selection_method=params['cluster_selection_method'],
                            prediction_data=True)
    clusters = hdbscan_model.fit_predict(embedding)
    
    # Ignore noise points for silhouette score
    if len(set(clusters)) > 1:
        score = silhouette_score(embedding, clusters)
    else:
        score = -1  # Penalize single cluster solutions

    return {'loss': -score, 'status': STATUS_OK}

# Define the search space
space = {
    'n_neighbors': hp.quniform('n_neighbors', 5, 50, 1),
    'n_components': hp.quniform('n_components', 2, 20, 1),
    'min_dist': hp.uniform('min_dist', 0.0, 0.99),
    'metric': hp.choice('metric', ['euclidean', 'manhattan', 'cosine']),
    'min_cluster_size': hp.quniform('min_cluster_size', 50, 200, 1),
    'cluster_selection_method': hp.choice('cluster_selection_method', ['eom', 'leaf'])
}

# Run hyperparameter optimization
trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

print("Best parameters:", best)

# Extracting results for plotting
results = trials.results
losses = [result['loss'] for result in results]

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(losses, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss (Negative Silhouette Score)')
plt.title('Hyperparameter Optimization Results')
plt.legend()
plt.show()
