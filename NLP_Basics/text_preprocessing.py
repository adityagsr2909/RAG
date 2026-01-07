import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt_tab')
nltk.download('stopwords')

sentence = "Hey Buddy I want to go to your house"
sentence = sentence.lower()

#- Tokenizer
tokens = nltk.word_tokenize(sentence)
print("Tokens:", tokens)

#- Remove Stop Words
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print("Filtered Tokens:", filtered_tokens)

#- Stemming
from nltk.stem import PorterStemmer
ps = PorterStemmer()
stemmed_tokens = [ps.stem(word) for word in filtered_tokens]
print("Stemmed Tokens:", stemmed_tokens)

#- Lemmatization
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
print("Lemmatized Tokens:", lemmatized_tokens)

#- Parts of Speech Tagging
nltk.download('averaged_perceptron_tagger_eng')
pos_tags = nltk.pos_tag(tokens)
print("POS Tags:", pos_tags)

#- Convert Words to Vectors
#- Bag of Words
#- Bag of words is a representation of text that describes the occurrence of words within a document.
#- Each unique word in the document is represented as a feature, and the value indicates the presence (1) or absence (0) of that word.

from sklearn.feature_extraction.text import CountVectorizer 
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform([sentence])
print("Bag of Words Vector:\n", X.toarray())
print("Feature Names:", vectorizer.get_feature_names_out())

#- disadvantages of Bag of Words
#- It ignores the order of words, which can lead to loss of context.
#- It can result in high-dimensional feature spaces, especially with large vocabularies.
#- It does not capture semantic relationships between words.
#- It can be sensitive to noise and irrelevant words in the text.

#- Ngrams
#- N-grams are contiguous sequences of n items (words or characters) from a given text.
#- They help capture context and relationships between words.
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(2, 2))  # Using bigrams
X = vectorizer.fit_transform([sentence])
print("Bigrams Vector:\n", X.toarray())
print("Bigrams Feature Names:", vectorizer.get_feature_names_out())

#- disadvantages of Ngrams
#- Increased dimensionality: As n increases, the number of possible n-grams grows exponentially
#- Data sparsity: Many n-grams may not appear frequently in the training data, leading to sparse representations
#- Context limitation: N-grams only capture local context within the specified n, potentially missing broader context
#- Computational complexity: Higher-order n-grams require more computational resources for storage and processing
#- Choosing the right n: Selecting an appropriate value for n can be challenging and may require experimentation

#- TF-IDF
#- TF-IDF (Term Frequency-Inverse Document Frequency) is a statistical measure used to evaluate the importance of a word in a document relative to a collection of documents (corpus).
#- Term Frequency = (Number of times a word appears in a document) / (Total number of words in the document)
#- Inverse Document Frequency = log_e(Total number of documents / Number of documents with the word in it)
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10)  # Using unigrams
X = vectorizer.fit_transform([sentence])
print("TF-IDF Vector:\n", X.toarray())
print("TF-IDF Feature Names:", vectorizer.get_feature_names_out())

#- disadvantages of TF-IDF
#- It assumes that words are independent of each other, ignoring context and word order.
#- It can be sensitive to noise and irrelevant words in the text.
#- It may not perform well on very short documents or texts with limited vocabulary.
#- It does not capture semantic relationships between words.


#- Word Embeddings
from gensim.models import Word2Vec, KeyedVectors

#- Word2Vec
#- Word2Vec is a neural network model that learns word embeddings from a large corpus of text.
#- It creates dense vector representations of words that capture semantic relationships.
#- The model uses either Skip-gram or CBOW architecture to predict context words or target words.

# Example usage:
sentences = [['this', 'is', 'a', 'sample'], ['another', 'example', 'sentence']]
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
vector = model.wv['sample']
print(f"Word2Vec Vector for 'sample': {vector}")
print(f"Most similar to 'sample': {model.wv.most_similar('sample')}")

#- disadvantages of Word Embeddings
#- Requires a large corpus of text for training to produce meaningful embeddings.   
#- Training can be computationally intensive and time-consuming.
#- Pre-trained embeddings may not capture domain-specific terminology or nuances.
#- May not effectively handle out-of-vocabulary words that were not present in the training data.

#- AvgWord2Vec
def avg_word2vec(sentence, model):
    words = sentence.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if not word_vectors:
        return None
    avg_vector = sum(word_vectors) / len(word_vectors)
    return avg_vector
avg_vector = avg_word2vec("this is a sample", model)
print(f"Average Word2Vec Vector: {avg_vector}")
