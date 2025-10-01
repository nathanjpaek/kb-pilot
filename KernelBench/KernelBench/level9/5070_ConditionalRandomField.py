import torch
from torch import nn
import torch.utils.data
import torch.nn.init as init


def initial_parameter(net, initial_method=None):
    """A method used to initialize the weights of PyTorch models.

    :param net: a PyTorch model
    :param initial_method: str, one of the following initializations

            - xavier_uniform
            - xavier_normal (default)
            - kaiming_normal, or msra
            - kaiming_uniform
            - orthogonal
            - sparse
            - normal
            - uniform

    """
    if initial_method == 'xavier_uniform':
        init_method = init.xavier_uniform_
    elif initial_method == 'xavier_normal':
        init_method = init.xavier_normal_
    elif initial_method == 'kaiming_normal' or initial_method == 'msra':
        init_method = init.kaiming_normal_
    elif initial_method == 'kaiming_uniform':
        init_method = init.kaiming_uniform_
    elif initial_method == 'orthogonal':
        init_method = init.orthogonal_
    elif initial_method == 'sparse':
        init_method = init.sparse_
    elif initial_method == 'normal':
        init_method = init.normal_
    elif initial_method == 'uniform':
        init_method = init.uniform_
    else:
        init_method = init.xavier_normal_

    def weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d) or isinstance(m
            , nn.Conv3d):
            if initial_method is not None:
                init_method(m.weight.data)
            else:
                init.xavier_normal_(m.weight.data)
            init.normal_(m.bias.data)
        elif isinstance(m, nn.LSTM):
            for w in m.parameters():
                if len(w.data.size()) > 1:
                    init_method(w.data)
                else:
                    init.normal_(w.data)
        elif hasattr(m, 'weight') and m.weight.requires_grad:
            init_method(m.weight.data)
        else:
            for w in m.parameters():
                if w.requires_grad:
                    if len(w.data.size()) > 1:
                        init_method(w.data)
                    else:
                        init.normal_(w.data)
    net.apply(weights_init)


def log_sum_exp(x, dim=-1):
    max_value, _ = x.max(dim=dim, keepdim=True)
    res = torch.log(torch.sum(torch.exp(x - max_value), dim=dim, keepdim=True)
        ) + max_value
    return res.squeeze(dim)


class ConditionalRandomField(nn.Module):

    def __init__(self, tag_size, include_start_end_trans=False,
        initial_method=None):
        """
        :param tag_size: int, num of tags
        :param include_start_end_trans: bool, whether to include start/end tag
        """
        super(ConditionalRandomField, self).__init__()
        self.include_start_end_trans = include_start_end_trans
        self.tag_size = tag_size
        self.trans_m = nn.Parameter(torch.randn(tag_size, tag_size))
        if self.include_start_end_trans:
            self.start_scores = nn.Parameter(torch.randn(tag_size))
            self.end_scores = nn.Parameter(torch.randn(tag_size))
        initial_parameter(self, initial_method)

    def reset_parameter(self):
        nn.init.xavier_normal_(self.trans_m)
        if self.include_start_end_trans:
            nn.init.normal_(self.start_scores)
            nn.init.normal_(self.end_scores)

    def _normalizer_likelihood(self, logits, mask):
        """
        Computes the (batch_size,) denominator term for the log-likelihood, which is the
        sum of the likelihoods across all possible state sequences.
        :param logits:FloatTensor, max_len x batch_size x tag_size
        :param mask:ByteTensor, max_len x batch_size
        :return:FloatTensor, batch_size
        """
        seq_len, batch_size, n_tags = logits.size()
        alpha = logits[0]
        if self.include_start_end_trans:
            alpha += self.start_scores.view(1, -1)
        for i in range(1, seq_len):
            emit_score = logits[i].view(batch_size, 1, n_tags)
            trans_score = self.trans_m.view(1, n_tags, n_tags)
            tmp = alpha.view(batch_size, n_tags, 1) + emit_score + trans_score
            alpha = log_sum_exp(tmp, 1) * mask[i].view(batch_size, 1
                ) + alpha * (1 - mask[i]).view(batch_size, 1)
        if self.include_start_end_trans:
            alpha += self.end_scores.view(1, -1)
        return log_sum_exp(alpha, 1)

    def _glod_score(self, logits, tags, mask):
        """
        Compute the score for the gold path.
        :param logits: FloatTensor, max_len x batch_size x tag_size
        :param tags: LongTensor, max_len x batch_size
        :param mask: ByteTensor, max_len x batch_size
        :return:FloatTensor, batch_size
        """
        seq_len, batch_size, _ = logits.size()
        batch_idx = torch.arange(batch_size, dtype=torch.long, device=
            logits.device)
        seq_idx = torch.arange(seq_len, dtype=torch.long, device=logits.device)
        trans_score = self.trans_m[tags[:seq_len - 1], tags[1:]] * mask[1:, :]
        emit_score = logits[seq_idx.view(-1, 1), batch_idx.view(1, -1), tags
            ] * mask
        score = trans_score + emit_score[:seq_len - 1, :]
        score = score.sum(0) + emit_score[-1] * mask[-1]
        if self.include_start_end_trans:
            st_scores = self.start_scores.view(1, -1).repeat(batch_size, 1)[
                batch_idx, tags[0]]
            last_idx = mask.long().sum(0) - 1
            ed_scores = self.end_scores.view(1, -1).repeat(batch_size, 1)[
                batch_idx, tags[last_idx, batch_idx]]
            score += st_scores + ed_scores
        return score

    def forward(self, feats, tags, mask):
        """
        Calculate the neg log likelihood
        :param feats:FloatTensor, batch_size x max_len x tag_size
        :param tags:LongTensor, batch_size x max_len
        :param mask:ByteTensor batch_size x max_len
        :return:FloatTensor, batch_size
        """
        feats = feats.transpose(0, 1)
        tags = tags.transpose(0, 1).long()
        mask = mask.transpose(0, 1).float()
        all_path_score = self._normalizer_likelihood(feats, mask)
        gold_path_score = self._glod_score(feats, tags, mask)
        return all_path_score - gold_path_score

    def viterbi_decode(self, data, mask, get_score=False):
        """
        Given a feats matrix, return best decode path and best score.
        :param data:FloatTensor, batch_size x max_len x tag_size
        :param mask:ByteTensor batch_size x max_len
        :param get_score: bool, whether to output the decode score.
        :return: scores, paths
        """
        batch_size, seq_len, n_tags = data.size()
        data = data.transpose(0, 1).data
        mask = mask.transpose(0, 1).data.float()
        vpath = data.new_zeros((seq_len, batch_size, n_tags), dtype=torch.long)
        vscore = data[0]
        if self.include_start_end_trans:
            vscore += self.start_scores.view(1, -1)
        for i in range(1, seq_len):
            prev_score = vscore.view(batch_size, n_tags, 1)
            cur_score = data[i].view(batch_size, 1, n_tags)
            trans_score = self.trans_m.view(1, n_tags, n_tags).data
            score = prev_score + trans_score + cur_score
            best_score, best_dst = score.max(1)
            vpath[i] = best_dst
            vscore = best_score * mask[i].view(batch_size, 1) + vscore * (1 -
                mask[i]).view(batch_size, 1)
        if self.include_start_end_trans:
            vscore += self.end_scores.view(1, -1)
        batch_idx = torch.arange(batch_size, dtype=torch.long, device=data.
            device)
        seq_idx = torch.arange(seq_len, dtype=torch.long, device=data.device)
        lens = mask.long().sum(0) - 1
        idxes = (lens.view(1, -1) - seq_idx.view(-1, 1)) % seq_len
        ans = data.new_empty((seq_len, batch_size), dtype=torch.long)
        ans_score, last_tags = vscore.max(1)
        ans[idxes[0], batch_idx] = last_tags
        for i in range(seq_len - 1):
            last_tags = vpath[idxes[i], batch_idx, last_tags]
            ans[idxes[i + 1], batch_idx] = last_tags
        if get_score:
            return ans_score, ans.transpose(0, 1)
        return ans.transpose(0, 1)


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4]), torch.rand([4, 4])]


def get_init_inputs():
    return [[], {'tag_size': 4}]
