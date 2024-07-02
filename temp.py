import itertools
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from adjustText import adjust_text
import matplotlib.patheffects as pe
import textwrap

# Define vibrant colors for the visualization to iterate over, including a color for -1 topic
colors = itertools.cycle([
    '#FF5733', '#33FF57', '#5733FF', '#33D7FF', '#FF33D7', 
    '#FFD733', '#FF33A8', '#33FFB5', '#FF8C33', '#3385FF', 
    '#A833FF', '#FF5733', '#FF33F6', '#33FFF6', '#FFF633', 
    '#FF3366', '#66FF33', '#FF6633', '#3366FF', '#33FF99', 
    '#9933FF', '#FF3399', '#FF9933', '#33FF33', '#FF33FF'
])

# Assuming `topic_model.topics_` and `topic_model` are defined elsewhere
topic_set = set(topic_model.topics_)

# Assign colors to topics, including -1
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
fig, ax = plt.subplots(figsize=(20, 20))

# Set the background color to black
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

# Adjust seaborn plot style to fit black background
sns.set(style="white", rc={"axes.facecolor": "black", "figure.facecolor": "black", "grid.color": ".6", "grid.linestyle": "--"})

# Plot scatterplot
sns.scatterplot(data=df, x='x', y='y', hue='Topic', palette=color_key, alpha=0.4, sizes=(0.4, 10), size="Length", ax=ax)

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
