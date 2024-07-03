from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hdbscan import HDBSCAN
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Define the objective function
def objective(params):
    # Extract parameters
    min_cluster_size = params['min_cluster_size']
    metric = params['metric']
    cluster_selection_method = params['cluster_selection_method']

    # Initialize HDBSCAN model with parameters
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric=metric,
        cluster_selection_method=cluster_selection_method,
        prediction_data=True
    )
    
    # Fit the model to your data (assuming X is your dataset)
    hdbscan_model.fit(X)
    
    # Get the labels and calculate silhouette score
    labels = hdbscan_model.labels_
    
    # Silhouette score is only defined if number of labels is greater than 1 and less than number of samples
    if len(set(labels)) > 1 and len(set(labels)) < len(labels):
        score = silhouette_score(X, labels)
    else:
        score = -1  # Poor score if the model doesn't produce enough clusters
    
    # Hyperopt tries to minimize the objective function, so return the negative silhouette score
    return {'loss': -score, 'status': STATUS_OK}

# Define the parameter space
space = {
    'min_cluster_size': hp.choice('min_cluster_size', range(2, 100)),
    'metric': hp.choice('metric', ['euclidean', 'manhattan', 'cosine']),
    'cluster_selection_method': hp.choice('cluster_selection_method', ['eom', 'leaf'])
}

# Run the optimization
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=100,  # Adjust the number of evaluations as needed
    trials=trials
)

print("Best parameters:", best)

# Plot the loss function
losses = [trial['result']['loss'] for trial in trials.trials]
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Trial')
plt.ylabel('Loss (Negative Silhouette Score)')
plt.title('Hyperopt Optimization Loss')
plt.show()
