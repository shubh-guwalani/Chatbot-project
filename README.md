import umap
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import pairwise_distances
import numpy as np

# Assuming 'embeddings' is your input data for UMAP
# Replace this with your actual data
# embeddings = ...

# Define the objective function
def objective(params):
    # Unpack the parameters
    n_neighbors = int(params['n_neighbors'])
    n_components = int(params['n_components'])
    min_dist = params['min_dist']
    metric = params['metric']
    
    # Initialize and fit UMAP
    reducer = umap.UMAP(n_neighbors=n_neighbors, 
                        n_components=n_components, 
                        min_dist=min_dist, 
                        metric=metric, 
                        random_state=42)
    embedding = reducer.fit_transform(embeddings)
    
    # Calculate the pairwise distances in the original space
    original_distances = pairwise_distances(embeddings)
    
    # Calculate the pairwise distances in the embedded space
    embedded_distances = pairwise_distances(embedding)
    
    # Compute the stress (a measure of distortion between the original and embedded spaces)
    stress = np.sum((original_distances - embedded_distances)**2)
    
    return {'loss': stress, 'status': STATUS_OK}

# Define the search space
space = {
    'n_neighbors': hp.quniform('n_neighbors', 2, 50, 1),
    'n_components': hp.quniform('n_components', 2, 10, 1),
    'min_dist': hp.uniform('min_dist', 0.001, 0.5),
    'metric': hp.choice('metric', ['euclidean', 'manhattan', 'cosine', 'chebyshev', 'minkowski'])
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

# If you need to convert 'metric' from its index to the actual metric string
metrics = ['euclidean', 'manhattan', 'cosine', 'chebyshev', 'minkowski']
best['metric'] = metrics[best['metric']]

# Use the best parameters to fit UMAP
best_umap = umap.UMAP(n_neighbors=int(best['n_neighbors']), 
                      n_components=int(best['n_components']), 
                      min_dist=best['min_dist'], 
                      metric=best['metric'], 
                      random_state=42)
best_embedding = best_umap.fit_transform(embeddings)

print("Best UMAP embedding shape:", best_embedding.shape)
