import numpy as np


def dist_vocab_embedding(token_ids, embedding_table, out):
    out[:] = embedding_table[token_ids.astype(np.int64)]
