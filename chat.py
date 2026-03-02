import os
from gensim.models import Word2Vec

# Make sure this matches the filename you used in your training script
MODEL_NAME = "universal_dictionary.model"

def chat():
    # 1. Load the model
    if not os.path.exists(MODEL_NAME):
        print(f"Error: {MODEL_NAME} not found. Did you run the training script?")
        return

    print(f"--- Loading {MODEL_NAME} ---")
    model = Word2Vec.load(MODEL_NAME)
    
    print("Ready! Type a word to find its closest neighbors (or 'quit' to stop).")

    while True:
        user_input = input("\nYour word > ").strip().lower()

        if user_input in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        # 2. Check if the word exists in the model's vocabulary
        if user_input in model.wv:
            # 3. Get the most similar words
            # topn=5 gives us the 5 highest scoring responses
            results = model.wv.most_similar(user_input, topn=5)
            
            print(f"Words most similar to '{user_input}':")
            for i, (word, score) in enumerate(results, 1):
                # The score is the Cosine Similarity (0 to 1)
                print(f"{i}. {word} (Score: {score:.4f})")
        else:
            print(f"Sorry, '{user_input}' isn't in my vocabulary.")
            print("Hint: Try words that appeared at least twice in your training data.")

if __name__ == "__main__":
    chat()
