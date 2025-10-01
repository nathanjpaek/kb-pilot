import torch
import torch.distributed as dist


class SinkhornKnopp(torch.nn.Module):

    def __init__(self, num_iters: 'int'=3, epsilon: 'float'=0.05,
        world_size: 'int'=1):
        """Approximates optimal transport using the Sinkhorn-Knopp algorithm.

        A simple iterative method to approach the double stochastic matrix is to alternately rescale
        rows and columns of the matrix to sum to 1.

        Args:
            num_iters (int, optional):  number of times to perform row and column normalization.
                Defaults to 3.
            epsilon (float, optional): weight for the entropy regularization term. Defaults to 0.05.
            world_size (int, optional): number of nodes for distributed training. Defaults to 1.
        """
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.world_size = world_size

    @torch.no_grad()
    def forward(self, Q: 'torch.Tensor') ->torch.Tensor:
        """Produces assignments using Sinkhorn-Knopp algorithm.

        Applies the entropy regularization, normalizes the Q matrix and then normalizes rows and
        columns in an alternating fashion for num_iter times. Before returning it normalizes again
        the columns in order for the output to be an assignment of samples to prototypes.

        Args:
            Q (torch.Tensor): cosine similarities between the features of the
                samples and the prototypes.

        Returns:
            torch.Tensor: assignment of samples to prototypes according to optimal transport.
        """
        Q = torch.exp(Q / self.epsilon).t()
        B = Q.shape[1] * self.world_size
        K = Q.shape[0]
        sum_Q = torch.sum(Q)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q
        for it in range(self.num_iters):
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B
        Q *= B
        return Q.t()


def get_inputs():
    return [torch.rand([4, 4])]


def get_init_inputs():
    return [[], {}]
