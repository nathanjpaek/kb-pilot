import torch
import numpy as np
from typing import Iterable
from typing import Optional
import torch.nn as nn


def find_closest_span_pairs(head: 'Iterable', tail: 'Iterable', backtrace:
    'Optional[bool]'=True):
    """
    Find all span pairs.

    Args:
        head: list of start position predictions, either 1 or 0
        tail: list of end position predictions, either 1 or 0
        backtrace: if there are more tail predictions than head predictions,
            then backtrace to find a closest head position to get a span pair

    Examples:
        >>> head = torch.tensor([1, 0, 0, 1, 0, 0, 1], dtype=torch.long)
        >>> tail = torch.tensor([0, 1, 0, 1, 0, 1, 1], dtype=torch.long)
        >>> find_closest_span_pairs(head, tail, backtrace=False)
        [(0, 1), (3, 3), (6, 6)]
        >>> find_closest_span_pairs(head, tail, backtrace=True)
        [(0, 1), (3, 3), (6, 6), (3, 5)]
    """
    if isinstance(head, torch.Tensor):
        head = head.detach().cpu()
    if isinstance(tail, torch.Tensor):
        tail = tail.detach().cpu()
    head_valid_poses = np.where(head == 1)[0]
    tail_valid_poses = np.where(tail == 1)[0]
    tail_used_poses = {pos: (False) for pos in tail_valid_poses.tolist()}
    pairs = []
    for head_i in head_valid_poses:
        tail_js = tail_valid_poses[tail_valid_poses >= head_i]
        if len(tail_js) > 0:
            tail_j = tail_js[0]
            tail_used_poses[tail_j] = True
            pairs.append((head_i, tail_j))
    if backtrace:
        for tail_j in tail_used_poses:
            if tail_used_poses[tail_j] is False:
                head_is = head_valid_poses[head_valid_poses <= tail_j]
                if len(head_is) > 0:
                    head_i = head_is[-1]
                    pairs.append((head_i, tail_j))
    return pairs


def find_closest_span_pairs_with_index(heads: 'Iterable', tails: 'Iterable',
    backtrace: 'Optional[bool]'=True):
    """
    Find all possible pairs with indexes,
    useful for object discoveries with class idx.

    Args:
        heads: batch of torch.Tensor
        tails: batch of torch.Tensor
        backtrace: if there are more tail predictions than head predictions,
            then backtrace to find a closest head position to get a span pair

    Examples:
        >>> heads = torch.tensor([[1, 0, 0, 1, 0, 0, 1], [1, 0, 0, 1, 0, 0, 1]], dtype=torch.long)
        >>> tails = torch.tensor([[0, 1, 0, 1, 0, 1, 1], [0, 1, 0, 0, 0, 1, 0]], dtype=torch.long)
        >>> find_closest_span_pairs(heads, tails, backtrace=False)
        [(0, 0, 1), (0, 3, 3), (0, 6, 6), (1, 0, 1), (1, 3, 5)]
        >>> find_closest_span_pairs(heads, tails, backtrace=True)
        [(0, 0, 1), (0, 3, 3), (0, 6, 6), (0, 3, 5), (1, 0, 1), (1, 3, 5)]
    """
    results = []
    for idx, (head, tail) in enumerate(zip(heads, tails)):
        pairs = find_closest_span_pairs(head, tail, backtrace=backtrace)
        for pair in pairs:
            results.append((idx, pair[0], pair[1]))
    return results


class SubjObjSpan(nn.Module):
    """
    Inputs:
        hidden: (batch_size, seq_len, hidden_size)
        one_subj_head: object golden head with one subject (batch_size, hidden_size)
        one_subj_tail: object golden tail with one subject (batch_size, hidden_size)
    """

    def __init__(self, hidden_size, num_classes, threshold:
        'Optional[float]'=0.5):
        super().__init__()
        self.threshold = threshold
        self.subj_head_ffnn = nn.Linear(hidden_size, 1)
        self.subj_tail_ffnn = nn.Linear(hidden_size, 1)
        self.obj_head_ffnn = nn.Linear(hidden_size, num_classes)
        self.obj_tail_ffnn = nn.Linear(hidden_size, num_classes)

    def get_objs_for_specific_subj(self, subj_head_mapping,
        subj_tail_mapping, hidden):
        subj_head = torch.matmul(subj_head_mapping, hidden)
        subj_tail = torch.matmul(subj_tail_mapping, hidden)
        sub = (subj_head + subj_tail) / 2
        encoded_text = hidden + sub
        pred_obj_heads = self.obj_head_ffnn(encoded_text)
        pred_obj_tails = self.obj_tail_ffnn(encoded_text)
        return pred_obj_heads, pred_obj_tails

    def build_mapping(self, subj_heads, subj_tails):
        """
        Build head & tail mapping for predicted subjects,
        for each instance in a batch, for a subject in all
        the predicted subjects, return a single subject
        and its corresponding mappings.
        """
        for subj_head, subj_tail in zip(subj_heads, subj_tails):
            subjs = find_closest_span_pairs(subj_head, subj_tail)
            seq_len = subj_head.shape[0]
            for subj in subjs:
                subj_head_mapping = torch.zeros(seq_len, device=subj_head.
                    device)
                subj_tail_mapping = torch.zeros(seq_len, device=subj_tail.
                    device)
                subj_head_mapping[subj[0]] = 1.0
                subj_tail_mapping[subj[1]] = 1.0
                yield subj, subj_head_mapping, subj_tail_mapping

    def build_batch_mapping(self, subj_head, subj_tail):
        """
        Build head & tail mapping for predicted subjects,
        for each instance in a batch, return all the predicted
        subjects and mappings.
        """
        subjs = find_closest_span_pairs(subj_head, subj_tail)
        seq_len = subj_head.shape[0]
        if len(subjs) > 0:
            subjs_head_mapping = torch.zeros(len(subjs), seq_len, device=
                subj_head.device)
            subjs_tail_mapping = torch.zeros(len(subjs), seq_len, device=
                subj_tail.device)
            for subj_idx, subj in enumerate(subjs):
                subjs_head_mapping[subj_idx, subj[0]] = 1.0
                subjs_tail_mapping[subj_idx, subj[1]] = 1.0
            return subjs, subjs_head_mapping, subjs_tail_mapping
        else:
            return None, None, None

    def forward(self, hidden, subj_head, subj_tail):
        subj_head_out = self.subj_head_ffnn(hidden)
        subj_tail_out = self.subj_tail_ffnn(hidden)
        obj_head_out, obj_tail_out = self.get_objs_for_specific_subj(subj_head
            .unsqueeze(1), subj_tail.unsqueeze(1), hidden)
        return subj_head_out.squeeze(-1), subj_tail_out.squeeze(-1
            ), obj_head_out, obj_tail_out

    def predict(self, hidden):
        if hidden.shape[0] != 1:
            raise RuntimeError(
                f'eval batch size must be 1 x hidden_size, while hidden is {hidden.shape}'
                )
        subj_head_out = self.subj_head_ffnn(hidden)
        subj_tail_out = self.subj_tail_ffnn(hidden)
        subj_head_out = torch.sigmoid(subj_head_out)
        subj_tail_out = torch.sigmoid(subj_tail_out)
        pred_subj_head = subj_head_out.ge(self.threshold).long()
        pred_subj_tail = subj_tail_out.ge(self.threshold).long()
        triples = []
        subjs, subj_head_mappings, subj_tail_mappings = (self.
            build_batch_mapping(pred_subj_head.squeeze(0).squeeze(-1),
            pred_subj_tail.squeeze(0).squeeze(-1)))
        if subjs:
            obj_head_out, obj_tail_out = self.get_objs_for_specific_subj(
                subj_head_mappings.unsqueeze(1), subj_tail_mappings.
                unsqueeze(1), hidden)
            obj_head_out = torch.sigmoid(obj_head_out)
            obj_tail_out = torch.sigmoid(obj_tail_out)
            obj_head_out = obj_head_out.ge(self.threshold).long()
            obj_tail_out = obj_tail_out.ge(self.threshold).long()
            for subj_idx, subj in enumerate(subjs):
                objs = find_closest_span_pairs_with_index(obj_head_out[
                    subj_idx].permute(1, 0), obj_tail_out[subj_idx].permute
                    (1, 0))
                for relation_idx, obj_pair_start, obj_pair_end in objs:
                    triples.append(((subj[0], subj[1] + 1), relation_idx, (
                        obj_pair_start, obj_pair_end + 1)))
        return [triples]


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4]), torch.rand(
        [4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'hidden_size': 4, 'num_classes': 4}]
