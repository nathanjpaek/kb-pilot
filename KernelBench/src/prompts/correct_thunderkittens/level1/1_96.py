# model_new.py
import torch
import tk_kernels


class ModelNew(torch.nn.Module):
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor):
        predictions = predictions.contiguous().cuda().to(torch.float16)
        targets = targets.contiguous().cuda().to(torch.float16)

        M, N = predictions.shape
        output = torch.empty_like(predictions)

        tk_kernels.dispatch_micro(
            predictions,
            targets,
            output,
            int(M),
            int(N)
        )

        return output.mean()