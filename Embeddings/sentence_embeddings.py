from sentence_transformers import SentenceTransformer

def sentence_embeddings():
    # SentenceTransformer is a library for state-of-the-art sentence embeddings
    # It's based on BERT architecture and fine-tuned for generating sentence embeddings
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    # 'paraphrase-MiniLM-L6-v2' is the name of the pre-trained model being used
    sentences = ["This is an example sentence", "Each sentence is converted to a vector"]
    embeddings = model.encode(sentences)
    print("Sentence Embedding shape:", embeddings.shape)
    print("First sentence embedding:", embeddings[0][:5])  # First 5 dimensions

if __name__ == "__main__":
    sentence_embeddings()
