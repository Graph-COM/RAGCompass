import numpy as np
import re
import string

from collections import Counter

def get_recall(gold_doc_list, top_doc_list):
    gold_doc_set = set(gold_doc_list)
    retrieved_doc_set = set(top_doc_list)
    
    if len(gold_doc_set) > 0:
        recall = len(gold_doc_set.intersection(retrieved_doc_set)) / len(gold_doc_set)
    else:
        recall = 0.0
    
    return recall

def get_ndcg(gold_doc_list, top_doc_list, k=None):
    if k is None:
        k = len(top_doc_list)

    gold_doc_set = set(gold_doc_list)

    dcg = 0.0
    for rank, doc in enumerate(top_doc_list[:k]):
        if doc in gold_doc_set:
            dcg += 1.0 / np.log2(rank + 2)

    ideal_rel_count = min(len(gold_doc_set), k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_rel_count))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def normalize_answer(answer: str) -> str:
    """
    Normalize a given string by applying the following transformations:
    1. Convert the string to lowercase.
    2. Remove punctuation characters.
    3. Remove the articles "a", "an", and "the".
    4. Normalize whitespace by collapsing multiple spaces into one.

    Args:
        answer (str): The input string to be normalized.

    Returns:
        str: The normalized string.
    """
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(answer))))

def compute_pairwise_f1(gold, predicted):
    gold_tokens = normalize_answer(gold).split()
    predicted_tokens = normalize_answer(predicted).split()
    common = Counter(predicted_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(predicted_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return 2 * (precision * recall) / (precision + recall)

def get_f1(gold_ans_set, pred_ans):
    f1_scores = [compute_pairwise_f1(gold, pred_ans) for gold in gold_ans_set]
    return np.max(f1_scores)

def get_em(gold_ans_set, pred_ans):
    normalized_pred = normalize_answer(pred_ans)
    return 1.0 if any(normalize_answer(gold) == normalized_pred for gold in gold_ans_set) else 0.0
