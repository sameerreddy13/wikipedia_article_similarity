import numpy as np


def get_embedding_dict(words, nlp):
    word_embeddings = {token.text:token.vector/np.linalg.norm(token.vector) for token in nlp.pipe(words)}
    return word_embeddings

def get_example_embedding_dict(words, nlp):
    word_embeddings = get_embedding_dict(words, nlp)
    king_embed = word_embeddings['king']
    print("Our word embeddings are of size {} and normalized to unit norm.".format(len(king_embed)))
    print("Here's an example of the first 5 entries of the word embedding for 'king':\n{}".format(king_embed[:5]))
    return word_embeddings
    

def get_example_similarities(ed):
    word_embeddings = ed
    king_embed = word_embeddings['king']
    print("Calculating Similarities...")
    for word, embed in word_embeddings.items():
        sim_king = embed.dot(king_embed)
        print("Between '{}' and 'king' = {:.2f}".format(word, sim_king))
    print()
    subtract_embed = king_embed - word_embeddings['man']
    print("Calculating Euclidean distances...")
    for word, embed in word_embeddings.items():
        sim_subtract = embed.dot(subtract_embed)
        print("Between '{}' and ('king' - 'man') = {:.2f}".format(word, sim_subtract))