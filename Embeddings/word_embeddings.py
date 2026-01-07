from gensim.models import Word2Vec

def word_embeddings():
    # Word2Vec is a popular method for generating word embeddings
    # It learns vector representations of words that capture semantic relationships
    sentences = [['cat', 'say', 'meow'], ['dog', 'say', 'woof']]
    # Parameters:
    # - vector_size=10: Dimensionality of the word vectors
    # - window=5: Maximum distance between current and predicted word within a sentence
    # - min_count=1: Ignores all words with total frequency lower than this
    # - workers=4: Number of CPU cores to use for training
    model = Word2Vec(sentences, vector_size=30, window=5, min_count=1, workers=4)
    print("Word Embedding for 'cat':", model.wv['cat'])

if __name__ == "__main__":
    word_embeddings()

