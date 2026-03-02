import pandas as pd
import re
import os
from pathlib import Path
import multiprocessing
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
import numpy as np

# --- 1. CONFIGURATION ---
INPUT_FOLDER = './txtDict1/'  # Path to your folder
COLUMN_NAME = 'definition'            # For CSVs
MODEL_NAME = "universal_dictionary.model"

# --- 2. THE MULTI-FORMAT STREAMER ---
class UniversalStreamer:
    """Iterates through CSV and TXT files, skipping empty lines and headers."""
    def __init__(self, folder_path):
        self.folder_path = Path(folder_path)

    def clean_text(self, text):
        """Removes symbols and returns a list of words."""
        if not text or not isinstance(text, str):
            return None
        text = text.strip()
        if not text: # Skip empty strings
            return None
        return re.sub(r'[^a-z\s]', '', text.lower()).split()

    def __iter__(self):
        # Scan for both .csv and .txt files
        for file_path in self.folder_path.glob('*'):
            ext = file_path.suffix.lower()
            
            # --- HANDLE CSV FILES ---
            if ext == '.csv':
                print(f"Feeding on CSV: {file_path.name}")
                for chunk in pd.read_csv(file_path, chunksize=500):
                    if COLUMN_NAME in chunk.columns:
                        for text in chunk[COLUMN_NAME]:
                            cleaned = self.clean_text(text)
                            if cleaned: yield cleaned

            # --- HANDLE TXT FILES ---
            elif ext == '.txt':
                print(f"Feeding on TXT: {file_path.name}")
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        cleaned = self.clean_text(line)
                        if cleaned: yield cleaned

# --- 3. TRAINING ENGINE ---
if __name__ == "__main__":
    # Ensure the folder exists
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"Created {INPUT_FOLDER}. Please drop your CSV/TXT files there!")
    else:
        print("Initializing the Universal Eater...")
        streamer = UniversalStreamer(INPUT_FOLDER)

        # Word2Vec training
        model = Word2Vec(
            sentences=streamer,
            vector_size=300,
            window=7,
            min_count=2,
            workers=multiprocessing.cpu_count() - 1,
            epochs=10
        )

        # Save the brain
        model.save(MODEL_NAME)
        print(f"\nTraining complete! 'Brain' saved as {MODEL_NAME}")
        
        # Quick Test
        word = "apple"
        if word in model.wv:
            print(f"\nTop 5 associations for '{word}':")
            print(model.wv.most_similar(word, topn=5))

        # 1. GET THE "BRAIN MAP" (The Word Vectors)
        # We extract the numerical coordinates for every word the model learned
        #word_vectors = model.wv.vectors 
        #word_labels = list(model.wv.key_to_index.keys())

        # 2. DEFINE THE NUMBER OF "FOLDERS" (K)
        # Let's tell the AI to find 2 distinct categories in our small example
        #num_clusters = 2

        # 3. RUN K-MEANS (The Sorting Hat)
        #kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(word_vectors)
        #clusters = kmeans.labels_

        # 4. SEE THE RESULTS
        #for i in range(num_clusters):
        #    words_in_cluster = [word_labels[j] for j, cluster_id in enumerate(clusters) if cluster_id == i]
        #    print(f"--- Folder {i+1} ---")
        #    print(words_in_cluster)
	
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
