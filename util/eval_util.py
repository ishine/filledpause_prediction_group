import torch

def calc_score_all(output, target):
    """Calculate precision, recall, f-score, and specificity for filler position.

    Params
    ------
    output: torch.Tensor
        Output of model, shape of (batch, max_text_len, tagset_size)
    target: torch.Tensor
        Target, shape of (batch, max_text_len)
    
    Returns
    -------
    precision: torch.Tensor | None
        Precision score
    recall: torch.Tensor | None
        Recall score
    f_score: torch.Tensor | None
        F-score
    specificity: torch.Tensor | None
        Specificity score
    """

    # Precision, recall, specificity
    tp_fp = tp_fn = tn_fp = tp = tn = 0
    for i in range(len(output)):
        tp_fp += (torch.argmax(output[i], dim=1) != 0).sum()
        tp_fn += (target[i] != 0).sum()
        tn_fp += (target[i] == 0).sum()
        tp += ((torch.argmax(output[i], dim=1) != 0) & (target[i] != 0)).sum()
        tn += ((torch.argmax(output[i], dim=1) == 0) & (target[i] == 0)).sum()

    precision = tp / tp_fp if tp_fp != 0 else None
    recall = tp / tp_fn if tp_fn != 0 else None
    specificity = tn / tn_fp if tn_fp != 0 else None

    # F-score
    if precision is not None and recall is not None:
        f_score = 2 * precision * recall / (precision + recall)
    else:
        f_score = None
        
    return precision, recall, f_score, specificity

def calc_score_each_filler(output, target, filler_index):
    """Calculate precision, recall, f-score, and specificity for each filler.

    Params
    ------
    output: torch.Tensor
        Output of model, shape of (batch, max_text_len, tagset_size)
    target: torch.Tensor
        Target, shape of (batch, max_text_len)
    filler_index : int
        Filler's index in filler list

    Returns
    -------
    precision: torch.Tensor | None
        Precision score
    recall: torch.Tensor | None
        Recall score
    f_score: torch.Tensor | None
        F-score
    specificity: torch.Tensor | None
        Specificity score
    """

    # Precision, recall, specificity
    tp_fp = tp_fn = tn_fp = tp = tn = 0
    for i in range(len(output)):
        tp_fp += (torch.argmax(output[i], dim=1) == filler_index).sum()
        tp_fn += (target[i] == filler_index).sum()
        tn_fp += (target[i] != filler_index).sum()
        tp += ((torch.argmax(output[i], dim=1) == filler_index) & (target[i] == filler_index)).sum()
        tn += ((torch.argmax(output[i], dim=1) != filler_index) & (target[i] != filler_index)).sum()

    precision = tp / tp_fp if tp_fp != 0 else None
    recall = tp / tp_fn if tp_fn != 0 else None
    specificity = tn / tn_fp if tn_fp != 0 else None

    # F-score
    if precision is not None and recall is not None:
        f_score = 2 * precision * recall / (precision + recall)
    else:
        f_score = None
        
    return precision, recall, f_score, specificity

