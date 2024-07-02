import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from hdbscan import HDBSCAN
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Example sentences
sentences = [
    "This is the first sentence.",
    "Here is another sentence.",
    "This is a different sentence.",
    # Add more sentences as needed
]

# Load a pre-trained sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
X = model.encode(sentences)

# Define a custom transformer to incorporate UMAP into a scikit-learn pipeline
class UMAPTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=42):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.min_dist = min_dist
        self.metric = metric
        self.random_state = random_state
        self.umap_model = UMAP()

    def fit(self, X, y=None):
        self.umap_model = UMAP(n_neighbors=self.n_neighbors,
                               n_components=self.n_components,
                               min_dist=self.min_dist,
                               metric=self.metric,
                               random_state=self.random_state).fit(X)
        return self

    def transform(self, X):
        return self.umap_model.transform(X)

# Define the parameter search space for UMAP and HDBSCAN
param_grid = {
    'umap__n_neighbors': Integer(5, 50),
    'umap__n_components': Integer(2, 50),
    'umap__min_dist': Real(0.0, 0.99, prior='uniform'),
    'umap__metric': ['cosine', 'euclidean', 'manhattan', 'chebyshev', 'minkowski'],
    'hdbscan__min_cluster_size': Integer(50, 300),
    'hdbscan__metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski'],
    'hdbscan__cluster_selection_method': ['eom', 'leaf']
}

# Create a pipeline with UMAP and HDBSCAN
pipe = Pipeline([
    ('umap', UMAPTransformer()),
    ('hdbscan', HDBSCAN(prediction_data=True))
])

# Define the objective function to minimize (negative silhouette score in this case)
@use_named_args(param_grid)
def objective(**params):
    pipe.set_params(**params)
    kf = KFold(n_splits=3)
    silhouette_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        pipe.fit(X_train)
        labels = pipe.predict(X_test)
        silhouette_scores.append(silhouette_score(X_test, labels))
    return -np.mean(silhouette_scores)

# Perform hyperparameter optimization with BayesSearchCV
opt = BayesSearchCV(
    pipe,
    param_grid,
    scoring=objective,
    n_iter=30,
    cv=3,
    n_jobs=-1,
    random_state=42,
    verbose=1
)

# Fit the optimizer with tqdm progress bar
with tqdm(total=opt.total_iterations, desc="Optimizing", position=0, leave=True) as pbar:
    opt.fit(X, callback=lambda x: pbar.update(1))

# Plot the convergence of the objective function (negative silhouette score)
plt.figure(figsize=(10, 6))
plt.plot(opt.cv_results_['total_time'], -opt.cv_results_['mean_test_score'], marker='o', linestyle='-', color='b')
plt.xlabel('Time (seconds)')
plt.ylabel('Negative Silhouette Score')
plt.title('Convergence Plot of Hyperparameter Optimization')
plt.grid(True)
plt.show()

# Best parameters
print("Best parameters:", opt.best_params_)
