import torch


class DistillationLoss(torch.nn.Module):

    def __init__(self, temperature: 'float'=1.0):
        super().__init__()
        self.temperature = 1.0

    def forward(self, student_logits, teacher_logits):
        teacher_prediction = torch.exp(torch.log_softmax(teacher_logits /
            self.temperature, dim=-1))
        student_prediction = torch.log_softmax(student_logits / self.
            temperature, dim=-1)
        loss = torch.mean(torch.sum(-teacher_prediction *
            student_prediction, dim=-1))
        return loss


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {}]
