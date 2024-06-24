import umap
from hdbscan import HDBSCAN
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import silhouette_score

# Assuming 'embeddings' is the result from the optimized UMAP model
# embeddings = ...

# Define the objective function for HDBSCAN
def objective(params):
    # Unpack the parameters
    min_cluster_size = int(params['min_cluster_size'])
    min_samples = int(params['min_samples'])
    cluster_selection_method = params['cluster_selection_method']
    
    # Initialize and fit HDBSCAN
    hdb_model = HDBSCAN(min_cluster_size=min_cluster_size, 
                        min_samples=min_samples, 
                        metric='euclidean',  # Keeping metric as euclidean for simplicity
                        cluster_selection_method=cluster_selection_method, 
                        prediction_data=True)
    
    hdb_model.fit(embeddings)
    
    # Evaluate the clustering using silhouette score
    if len(set(hdb_model.labels_)) > 1:  # More than one cluster
        score = silhouette_score(embeddings, hdb_model.labels_)
    else:
        score = -1  # Poor score if only one cluster is found
    
    return {'loss': -score, 'status': STATUS_OK}  # We negate the score because Hyperopt minimizes the objective

# Define the search space
space = {
    'min_cluster_size': hp.quniform('min_cluster_size', 50, 300, 1),
    'min_samples': hp.quniform('min_samples', 1, 100, 1),
    'cluster_selection_method': hp.choice('cluster_selection_method', ['eom', 'leaf'])
}

# Run the optimization
trials = Trials()
best = fmin(fn=objective, 
            space=space, 
            algo=tpe.suggest, 
            max_evals=100, 
            trials=trials)

# Print the best parameters
print("Best parameters:", best)

# If you need to convert 'cluster_selection_method' from its index to the actual method string
cluster_selection_methods = ['eom', 'leaf']
best['cluster_selection_method'] = cluster_selection_methods[best['cluster_selection_method']]

# Use the best parameters to fit HDBSCAN
best_hdb_model = HDBSCAN(min_cluster_size=int(best['min_cluster_size']), 
                         min_samples=int(best['min_samples']), 
                         metric='euclidean', 
                         cluster_selection_method=best['cluster_selection_method'], 
                         prediction_data=True)
best_hdb_model.fit(embeddings)

print("Best HDBSCAN model labels:", best_hdb_model.labels_)
