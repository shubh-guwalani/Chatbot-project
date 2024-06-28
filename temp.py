import pandas as pd
from collections import defaultdict

# Example data
clusters = [
    ["This is example text", "This is test document", "More text data"],  # Cluster 0
    ["This is sample data", "Another example data"]                      # Cluster 1
]

keywords_per_doc = [
    [['example', 'text'], ['test', 'document'], ['text', 'data']],       # Keywords for Cluster 0
    [['sample', 'data'], ['example', 'data']]                            # Keywords for Cluster 1
]

# Calculate Intra-Cluster Document Frequency
intra_cluster_freq = defaultdict(lambda: defaultdict(int))

for cluster_id, documents in enumerate(keywords_per_doc):
    for keywords in documents:
        for keyword in keywords:
            intra_cluster_freq[cluster_id][keyword] += 1

intra_df = pd.DataFrame(intra_cluster_freq).fillna(0)
print("Intra-Cluster Frequencies:\n", intra_df)

# Calculate Inter-Cluster Document Frequency
inter_cluster_freq = defaultdict(int)

for documents in keywords_per_doc:
    for keywords in documents:
        for keyword in keywords:
            inter_cluster_freq[keyword] += 1

inter_df = pd.Series(inter_cluster_freq).fillna(0)
print("\nInter-Cluster Frequencies:\n", inter_df)

# Rank Keywords
intra_rankings = {cluster: freq.sort_values(ascending=False) for cluster, freq in intra_df.items()}
inter_rankings = inter_df.sort_values(ascending=False)

# Display rankings
print("\nIntra-Cluster Rankings:")
for cluster, ranking in intra_rankings.items():
    print(f"Cluster {cluster}:")
    print(ranking)

print("\nInter-Cluster Rankings:")
print(inter_rankings)
