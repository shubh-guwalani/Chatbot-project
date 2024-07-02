import itertools
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from adjustText import adjust_text
import matplotlib.patheffects as pe
import textwrap

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

# Plotting with seaborn and matplotlib
fig = plt.figure(figsize=(20, 20))
sns.scatterplot(data=df, x='x', y='y', hue='Topic', palette=color_key, alpha=0.4, sizes=(0.4, 10), size="Length")

# Annotate top 50 topics
texts, xs, ys = [], [], []
for index, row in mean_df.iterrows():
    topic = row["Topic"]
    name = textwrap.fill(topic_model.custom_labels_[int(topic)], 20)
    if int(topic) <= 50:
        xs.append(row["x"])
        ys.append(row["y"])
        texts.append(plt.text(row["x"], row["y"], name, size=10, ha="center", color=color_key[str(int(topic))],
                              path_effects=[pe.withStroke(linewidth=0.5, foreground="black")]))

# Adjust annotations such that they do not overlap
adjust_text(texts, x=xs, y=ys, time_lim=1, force_text=(0.01, 0.02), force_static=(0.01, 0.02), force_pull=(0.5, 0.5))

plt.axis('off')
plt.legend('', frameon=False)
plt.show()
