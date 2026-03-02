from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np
import re

# 1. THE "FOOD" (Simplified Dictionary Data)
# In a real scenario, you'd load a massive JSON or TXT file of the Oxford Dictionary.
dictionary_data = [
    "Apple is a round fruit with red or green skin and a whitish inside",
    "Banana is a long curved fruit with a thick yellow skin",
    "Fruit is the sweet and fleshy product of a tree or other plant that contains seed",
    "Red is a color like that of a poppy or blood",
    "Yellow is a color like that of a lemon or gold"
]

# 2. PRE-PROCESSING (Cleaning the "food")
# We lowercase everything and remove punctuation so the AI doesn't get confused.
processed_data = [re.sub(r'[^\w\s]', '', d.lower()).split() for d in dictionary_data]

# 3. TRAINING (The "Eating" Process)
# vector_size=100 means each word is defined by 100 different "traits"
model = Word2Vec(sentences=processed_data, vector_size=100, window=5, min_count=1, workers=4)

# 4. TESTING THE "BRAIN"
word = "apple"
print(f"Words most similar to '{word}':")
print(model.wv.most_similar(word, topn=3))

from sklearn.cluster import KMeans
import numpy as np

# 1. GET THE "BRAIN MAP" (The Word Vectors)
# We extract the numerical coordinates for every word the model learned
word_vectors = model.wv.vectors 
word_labels = list(model.wv.key_to_index.keys())

# 2. DEFINE THE NUMBER OF "FOLDERS" (K)
# Let's tell the AI to find 2 distinct categories in our small example
num_clusters = 2

# 3. RUN K-MEANS (The Sorting Hat)
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(word_vectors)
clusters = kmeans.labels_

# 4. SEE THE RESULTS
for i in range(num_clusters):
    words_in_cluster = [word_labels[j] for j, cluster_id in enumerate(clusters) if cluster_id == i]
    print(f"--- Folder {i+1} ---")
    print(words_in_cluster)


# 1. DEFINE OUR "POLAR NORTH" AND "POLAR SOUTH"
# We use these to tell the AI what 'Good' and 'Bad' look like.
positive_seeds = ["sweet", "fleshy", "gold", "bright"]
negative_seeds = ["thick", "blood", "dark", "hard"]

def calculate_sentiment(word, model):
    if word not in model.wv:
        return "Word not in dictionary"
    
    # Get the vector for the word we want to test
    word_vec = model.wv[word]
    
    # Calculate average similarity to positive seeds vs negative seeds
    pos_score = np.mean([model.wv.similarity(word, seed) for seed in positive_seeds if seed in model.wv])
    neg_score = np.mean([model.wv.similarity(word, seed) for seed in negative_seeds if seed in model.wv])
    
    # Normalize into a score from -1 to 1
    sentiment_score = pos_score - neg_score
    return sentiment_score

# 2. TEST THE DICTIONARY
test_words = ["apple", "banana", "yellow", "red"]

for w in test_words:
    score = calculate_sentiment(w, model)
    label = "Positive" if score > 0 else "Negative"
    print(f"Word: {w:10} | Score: {score:.4f} | Label: {label}")

def score_sentence(sentence, model):
    # Clean the sentence
    words = re.sub(r'[^\w\s]', '', sentence.lower()).split()

    scores = []
    for word in words:
        score = calculate_sentiment(word, model)
        # We only count words that the AI actually "ate" (found in the dictionary)
        if isinstance(score, float):
            scores.append(score)

    if not scores:
        return 0.0

    # The final "mood" of the sentence is the average of its words
    return sum(scores) / len(scores)

# --- TESTING ---
sentences = [
    "The apple was yellow and sweet",
    "Blood is red and thick",
    "A banana is a sweet yellow fruit"
]

for s in sentences:
    avg_score = score_sentence(s, model)
    label = "Happy/Positive" if avg_score > 0 else "Dark/Negative"
    print(f"Sentence: '{s}'")
    print(f"Final Mood: {label} ({avg_score:.4f})\n")
