import pandas as pd
from collections import Counter

# Load the dataset
df = pd.read_csv('reddit_news.csv')

# Ensure creation_date is in datetime format
df['creation_date'] = pd.to_datetime(df['creation_date'])

# Extract date only (without time) if needed
df['date'] = df['creation_date'].dt.date

# Function to count and rank keywords for each date
def rank_keywords_by_date(df):
    # Create a dictionary to store the results
    result = {}

    # Group by date
    grouped = df.groupby('date')
    for date, group in grouped:
        # Combine all keywords for the date
        all_keywords = []
        for keywords in group['keywords']:
            all_keywords.extend(keywords.split(','))  # Assuming keywords are comma-separated
        
        # Count the frequency of each keyword
        keyword_counts = Counter(all_keywords)
        
        # Rank keywords based on frequency
        ranked_keywords = keyword_counts.most_common()
        
        # Store the result
        result[date] = ranked_keywords

    return result

# Rank keywords by date
ranked_keywords_by_date = rank_keywords_by_date(df)

# Convert the result to a DataFrame for better visualization
ranked_df_list = []
for date, keywords in ranked_keywords_by_date.items():
    for rank, (keyword, count) in enumerate(keywords, start=1):
        ranked_df_list.append({'date': date, 'rank': rank, 'keyword': keyword, 'count': count})

ranked_df = pd.DataFrame(ranked_df_list)

# Save the result to a CSV file
ranked_df.to_csv('ranked_keywords_by_date.csv', index=False)

# Print the result
print(ranked_df.head())
