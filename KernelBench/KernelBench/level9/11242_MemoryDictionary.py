import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *


class MemoryDictionary(nn.Module):
    """このクラスでは
        M_1 -> M_2 
    という写像を生成します。
    この記憶辞書の最もシンプルな場合である、二層の全結合層によって作成されます。
    
    
    """

    def __init__(self, num_memory: 'int', num_dims: 'int', device:
        'torch.device'='cpu', dtype: 'torch.dtype'=torch.float,
        connection_margin: 'float'=0.1, max_lr: 'float'=10.0, num_search:
        'int'=10) ->None:
        """
        初期重みを生成します。初期重みは恒等写像にするために
        weight1.T = weight2という関係があります。
            D(M) = M
        
        Generate initial weights.
        The initial weights are always initialized with the relation 
            weight1.T = weight2 
        to make it an indentical mapping. 

        """
        super().__init__()
        assert 0 < connection_margin < 0.5
        self.num_memory = num_memory
        self.num_dims = num_dims
        self.connection_threshold = 0.5 + connection_margin
        self.max_lr = max_lr
        self.num_search = num_search
        self.memory_indices = torch.arange(self.num_memory, dtype=torch.int64)
        weight1 = torch.randn((num_memory, num_dims), device=device, dtype=
            dtype)
        weight2 = weight1.T.clone()
        self.weight1 = torch.nn.Parameter(weight1, True)
        self.weight2 = torch.nn.Parameter(weight2, True)

    def forward(self, input_indices: 'torch.Tensor') ->torch.Tensor:
        """
        Caluculate connections.

        input_indices: (N,)
        return -> (N, num_memories)
        """
        return torch.matmul(self.weight1[input_indices], self.weight2)

    def add_memories(self, num: 'int') ->None:
        """
        numだけ重みを拡大します。idxがnum_memoryよりも大きい場合にtraceやconnectを
        実行する場合はこのメソッドを必ず呼び出してください。

        Expand the weight by num.
        Be sure to call this method when executing trace or connect
        when input idx is greater than num_memory.
        """
        new = torch.randn((num, self.num_dims)).type_as(self.weight1)
        newT = new.T
        new_ids = torch.arange(self.num_memory, self.num_memory + num,
            dtype=torch.int64)
        self.num_memory += num
        self.weight1.data = torch.cat([self.weight1.data, new], 0)
        self.weight2.data = torch.cat([self.weight2.data, newT], 1)
        self.memory_indices = torch.cat([self.memory_indices, new_ids])

    @torch.no_grad()
    def trace(self, indices: 'torch.Tensor', roop: 'bool'=True) ->torch.Tensor:
        """
        indicesの中の記憶からつながれた記憶を取り出し、重複のない結果を返します

        Trace and extract connected memories from the memory in indices,
        and returns non-duplicated result.
        """
        connected = torch.zeros(self.num_memory, dtype=torch.bool)
        if roop:
            for i in indices:
                next_id = self.trace_a_memory(i)
                connected[next_id] = True
            result = self.memory_indices[connected]
        else:
            next_indices = self.trace_memories(indices)
            connected[next_indices] = True
            result = self.memory_indices[connected]
        return result

    def trace_a_memory(self, idx: 'int') ->int:
        assert 0 <= idx < self.num_memory
        out = self.forward(idx).view(-1)
        out_idx = torch.argmax(out, dim=0).item()
        return out_idx

    def trace_memories(self, indices: 'torch.Tensor') ->torch.Tensor:
        return torch.argmax(self.forward(indices), dim=1)

    def get_memory_vector(self, indices: 'torch.Tensor', requires_grad:
        'bool'=False) ->torch.Tensor:
        """ returns memory vector from 1st layer V."""
        vec = self.weight1[indices]
        if not requires_grad:
            vec = vec.detach()
        return vec

    def connect(self, src_idx: 'int', tgt_idx: 'int') ->None:
        """
        connect M_srcidx to M_tgtidx.
        
        重みを更新するときにちょうどよく更新されるようにするために、
        Softmaxの分母を分子で割ったものが 1/connection_marginよりも
        小さくなるように学習率を探しています。
        
        searching for a learning rate so that the denominator of Softmax 
        divided by the numerator is less than 1 / connection_margin 
        to ensure that the weights are updated just right when they are updated, 
        """
        v, ngU = self.weight1[src_idx], self.weight2.detach()
        output = torch.matmul(v, self.weight2)
        out_prob = F.softmax(output, dim=0)
        if out_prob[tgt_idx] > self.connection_threshold:
            return
        log_prob = F.log_softmax(output, dim=0)
        loss = -log_prob[tgt_idx]
        loss.backward()
        g = self.weight1.grad[src_idx]
        v = v.detach()
        H = self.weight2.grad
        lr = self.calculate_lr(v, g, ngU, H, tgt_idx)
        if lr != 0.0:
            self.weight1.data[src_idx] = v - lr * g
            self.weight2.data -= lr * H

    @torch.no_grad()
    def calculate_lr(self, v: 'torch.Tensor', g: 'torch.Tensor', U:
        'torch.Tensor', H: 'torch.Tensor', tgt_idx: 'int') ->float:
        """
        connection_threshold付近をモデルが出力するようにするための学習率を計算します。
        Calculate the learning rate to get the model to output around connection_threshold.
        """
        A, B, C = g.matmul(H), g.matmul(U) + v.matmul(H), v.matmul(U)
        A, B, C = A, B, C
        alpha = (B[tgt_idx] / (2 * A[tgt_idx])).item()
        if A[tgt_idx] < 0 and alpha < self.max_lr and 0 < alpha:
            max_lr = alpha
        else:
            max_lr = self.max_lr
        A_p, B_p, C_p = A - A[tgt_idx], B - B[tgt_idx], C - C[tgt_idx]
        lr = self.binary_search_lr(0.0, max_lr, A_p, B_p, C_p, self.num_search)
        return lr

    def binary_search_lr(self, min_lr: 'float', max_lr: 'float', A_p:
        'torch.Tensor', B_p: 'torch.Tensor', C_p: 'torch.Tensor', num_steps:
        'int') ->float:
        """
        バイナリサーチをして漸近的に最適な学習率を求めます。
        Calculate the optimal lr asympototically by binary search.
        """
        assert min_lr < max_lr
        max_out = self._calc_sum_softmax(max_lr, A_p, B_p, C_p)
        min_out = self._calc_sum_softmax(min_lr, A_p, B_p, C_p)
        inv_ct = 1 / self.connection_threshold
        if max_out > inv_ct and min_out > inv_ct:
            return max_lr
        elif max_out < inv_ct and min_out < inv_ct:
            return min_lr
        for _ in range(num_steps):
            m_lr = (min_lr + max_lr) / 2
            denom = self._calc_sum_softmax(m_lr, A_p, B_p, C_p)
            if denom > 1 / inv_ct:
                min_lr = m_lr
            else:
                max_lr = m_lr
        return m_lr

    def _calc_sum_softmax(self, lr: 'float', A_p: 'torch.Tensor', B_p:
        'torch.Tensor', C_p: 'torch.Tensor') ->float:
        return torch.sum(torch.exp(lr ** 2 * A_p - lr * B_p + C_p)).item()

    def __getitem__(self, indices: 'Union[int, torch.Tensor]') ->Union[int,
        torch.Tensor]:
        """
        記憶配列の場合は通常通りself.traceを実行します。
        単一の記憶に対してはそのまま参照結果を返します。
        For a memory array, execute self.trace as usual.
        For a single memory, it returns the reference result as is.
        """
        if type(indices) is torch.Tensor:
            if indices.dim() > 0:
                return self.trace(indices, True)
        return self.trace_a_memory(indices)

    def __setitem__(self, src_idx: 'int', tgt_idx: 'int') ->None:
        """
        execute self.connect
        """
        self.connect(src_idx, tgt_idx)


def get_inputs():
    return [torch.ones([4], dtype=torch.int64)]


def get_init_inputs():
    return [[], {'num_memory': 4, 'num_dims': 4}]
