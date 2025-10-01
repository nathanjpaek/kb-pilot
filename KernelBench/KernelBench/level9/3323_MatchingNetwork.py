from _paritybench_helpers import _mock_config
import torch
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import torch.nn.parallel


class MatchingNetwork(nn.Module):

    def __init__(self, opt):
        super(MatchingNetwork, self).__init__()
        scale_cls = opt['scale_cls'] if 'scale_cls' in opt else 10.0
        self.scale_cls = nn.Parameter(torch.FloatTensor(1).fill_(scale_cls),
            requires_grad=True)

    def forward(self, features_test, features_train, labels_train):
        """Recognize novel categories based on the Matching Nets approach.

        Classify the test examples (i.e., `features_test`) using the available
        training examples (i.e., `features_test` and `labels_train`) using the
        Matching Nets approach.

        Args:
            features_test: A 3D tensor with shape
                [batch_size x num_test_examples x num_channels] that represents
                the test features of each training episode in the batch.
            features_train: A 3D tensor with shape
                [batch_size x num_train_examples x num_channels] that represents
                the train features of each training episode in the batch.
            labels_train: A 3D tensor with shape
                [batch_size x num_train_examples x nKnovel] that represents
                the train labels (encoded as 1-hot vectors) of each training
                episode in the batch.

        Return:
            scores_cls: A 3D tensor with shape
                [batch_size x num_test_examples x nKnovel] that represents the
                classification scores of the test feature vectors for the
                nKnovel novel categories.
        """
        assert features_train.dim() == 3
        assert labels_train.dim() == 3
        assert features_test.dim() == 3
        assert features_train.size(0) == labels_train.size(0)
        assert features_train.size(0) == features_test.size(0)
        assert features_train.size(1) == labels_train.size(1)
        assert features_train.size(2) == features_test.size(2)
        batch_size, num_test_examples, _num_channels = features_test.size()
        num_train_examples = features_train.size(1)
        labels_train.size(2)
        features_test = F.normalize(features_test, p=2, dim=features_test.
            dim() - 1, eps=1e-12)
        features_train = F.normalize(features_train, p=2, dim=
            features_train.dim() - 1, eps=1e-12)
        cosine_similarities = self.scale_cls * torch.bmm(features_test,
            features_train.transpose(1, 2))
        cosine_similarities = cosine_similarities.view(batch_size *
            num_test_examples, num_train_examples)
        cosine_scores = F.softmax(cosine_similarities)
        cosine_scores = cosine_scores.view(batch_size, num_test_examples,
            num_train_examples)
        scores_cls = torch.bmm(cosine_scores, labels_train)
        scores_cls = torch.log(torch.clamp(scores_cls, min=1e-07))
        return scores_cls


def get_inputs():
    return [torch.rand([4, 4, 4]), torch.rand([4, 4, 4]), torch.rand([4, 4, 4])
        ]


def get_init_inputs():
    return [[], {'opt': _mock_config(scale_cls=1.0)}]
