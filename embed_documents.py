from sklearn import neighbors
from scipy import stats
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

def embed_doc(doc):
    return np.mean([token.vector for token in doc], axis=0)

def load_X(docs):
    X = np.array([embed_doc(d) for i, d in enumerate(docs)])
    return X

def load_Y(corpus):
    Y_string = [corpus.get_custom(d, 'category') for d in corpus]
    le = LabelEncoder()
    Y = le.fit_transform(Y_string)
    return Y, le

def errors(Y_pred, Y):
    return np.argwhere(Y_pred != Y).flatten()

# def get_logistic_regressor()

def build_knn(X):
    knn = neighbors.NearestNeighbors(metric='cosine')
    knn.fit(X)
    return knn

def get_neighbors(doc_embed, knn, k=3, training_sample=True):
    # If the given embedding was part of the data on which the knn class was fit,
    # it will be returned trivially as the 1-nearest neighbor. Thus we remove it.
    if training_sample: 
        dists, inds = knn.kneighbors(doc_embed.reshape(1, -1), n_neighbors=k+1) 
        inds = inds.flatten()[1:]
        dists = dists.flatten()[1:]
    else:
        dists, inds = knn.kneighbors(doc_embed.reshape(1, -1), n_neighbors=k) 
        inds = inds.flatten()
        dists = dists.flatten()
    return dists, inds

def print_helper(idx, corpus):
    d = corpus[idx]
    return "Title: {}, \nCategory: {}".format(corpus.title(d), corpus.category(d))

def print_helper_nbrs(dists, inds, corpus):
    print("NEAREST NEIGHBORS")
    for ind, dist in zip(inds, dists):
        title, cat = corpus.title(corpus[ind]), corpus.category(corpus[ind])
        print(title, "(distance = {})".format(dist), '\n', cat)
    print()

def print_neighbors(doc_idx, corpus, X, knn, training_sample=True):
    print(print_helper(doc_idx, corpus))
    dists, inds = get_neighbors(X[doc_idx], knn, training_sample=training_sample)
    print_helper_nbrs(dists, inds, corpus)
    
def show_five(corpus, X, knn):
    print("Showing some documents and their neighbors...\n")
    corpus_idxs = range(len(corpus))
    sample_idxs = np.random.choice(corpus_idxs, 5)
    for i in sample_idxs:
        print_neighbors(i, corpus, X, knn)
        
        
        