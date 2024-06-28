# Sample data: lists of keywords and documents
keywords = ["keyword1", "keyword2", "keyword3", "keyword4"]
documents = [
    "This document contains keyword1 and keyword2.",
    "This document contains keyword2 and keyword3.",
    "This document contains keyword3 and keyword4.",
    "This document contains keyword1 and keyword4.",
]

# Function to rank keywords based on document frequency
def rank_keywords_by_doc_frequency(keywords, documents):
    # Initialize a dictionary to store keyword frequencies
    keyword_frequency = {keyword: 0 for keyword in keywords}

    # Iterate over each document
    for doc in documents:
        # Check each keyword in the document
        for keyword in keywords:
            if keyword in doc:
                keyword_frequency[keyword] += 1

    # Sort the keywords by their frequency in descending order
    sorted_keywords = sorted(keyword_frequency.items(), key=lambda item: item[1], reverse=True)

    return sorted_keywords

# Rank the keywords
ranked_keywords = rank_keywords_by_doc_frequency(keywords, documents)

# Print the ranked keywords
for keyword, frequency in ranked_keywords:
    print(f"Keyword: {keyword}, Frequency: {frequency}")
