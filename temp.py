import itertools
import pandas as pd
import matplotlib.pyplot as plt

# Define colors for the visualization to iterate over
colors = itertools.cycle([
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', 
    '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', 
    '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', 
    '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', 
    '#ffffff', '#000000'
])

# Assuming `topic_model.topics_` and `topic_model` are defined elsewhere
# Ensure that -1 is excluded from the set of topics
topic_set = set(topic_model.topics_) - {-1}

# Assign colors to topics
color_key = {str(topic): next(colors) for topic in topic_set}

# Debug: Print the color_key to check color assignments
print("Color Key:", color_key)

# Prepare dataframe and ignore outliers
df = pd.DataFrame({
    "x": reduced_embeddings[:, 0],
    "y": reduced_embeddings[:, 1],
    "Topic": [str(t) for t in topic_model.topics_]
})
df["Length"] = [len(doc) for doc in abstracts]
df = df.loc[df.Topic != "-1"]
df = df.loc[(df.y > -10) & (df.y < 10) & (df.x < 10) & (df.x > -10), :]
df["Topic"] = df["Topic"].astype("category")

# Debug: Print the dataframe to inspect
print("DataFrame Head:\n", df.head())

# Get centroids of clusters
mean_df = df.groupby("Topic").mean().reset_index()
mean_df["Topic"] = mean_df["Topic"].astype(int)
mean_df = mean_df.sort_values("Topic")

# Debug: Print the mean dataframe to inspect
print("Mean DataFrame Head:\n", mean_df.head())

# Example plot to visualize the data
plt.figure(figsize=(10, 8))
for topic, color in color_key.items():
    topic_data = df[df["Topic"] == topic]
    plt.scatter(topic_data["x"], topic_data["y"], c=color, label=f"Topic {topic}")

plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Topic Visualization')
plt.show()
