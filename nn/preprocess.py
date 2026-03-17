# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    pos_seqs = [s for s, l in zip(seqs, labels) if l]
    neg_seqs = [s for s, l in zip(seqs, labels) if not l]

    n = max(len(pos_seqs), len(neg_seqs))

    # Oversample the minority class with replacement to match majority class size
    if len(pos_seqs) < len(neg_seqs):
        pos_sampled = list(np.random.choice(pos_seqs, size=n, replace=True))
        neg_sampled = neg_seqs
    else:
        neg_sampled = list(np.random.choice(neg_seqs, size=n, replace=True))
        pos_sampled = pos_seqs

    sampled_seqs = pos_sampled + neg_sampled
    sampled_labels = [True] * n + [False] * n

    # Shuffle so positives and negatives are interleaved
    indices = np.random.permutation(len(sampled_seqs))
    sampled_seqs = [sampled_seqs[i] for i in indices]
    sampled_labels = [sampled_labels[i] for i in indices]

    return sampled_seqs, sampled_labels

def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """
    base_map = {'A': [1,0,0,0], 'T': [0,1,0,0], 'C': [0,0,1,0], 'G': [0,0,0,1]}
    encodings = []
    for seq in seq_arr:
        enc = []
        for base in seq.upper():
            enc.extend(base_map.get(base, [0,0,0,0]))
        encodings.append(enc)
    return np.array(encodings)