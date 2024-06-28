import pandas as pd
from collections import defaultdict

# Example dataset
data = {
    'cluster': [0, 1, 0, 1, 0],
    'document': [
        "This is example text", 
        "This is sample data", 
        "This is test document", 
        "Another example data", 
        "More text data"
    ],
    'keywords': [
        ['example', 'text'], 
        ['sample', 'data'], 
        ['test', 'document'], 
        ['example', 'data'], 
        ['text', 'data']
    ]
}

df = pd.DataFrame(data)

# Calculate Intra-Cluster Document Frequency
intra_cluster_freq = defaultdict(lambda: defaultdict(int))

for _, row in df.iterrows():
    cluster = row['cluster']
    keywords = row['keywords']
    for keyword in keywords:
        intra_cluster_freq[cluster][keyword] += 1

intra_df = pd.DataFrame(intra_cluster_freq).fillna(0)
print("Intra-Cluster Frequencies:\n", intra_df)

# Calculate Inter-Cluster Document Frequency
inter_cluster_freq = defaultdict(int)

for keywords in df['keywords']:
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
