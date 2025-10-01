import torch
import typing
from torch import Tensor
from collections import Counter
from typing import List
from typing import Optional
from typing import Union
from torch.utils.data import Dataset
import torch.utils.data.dataloader
from torch import nn
import torch.nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


def dot_product(a: 'torch.Tensor', b: 'torch.Tensor', normalize=False):
    """
    Computes dot product for pairs of vectors.
    :param normalize: Vectors are normalized (leads to cosine similarity)
    :return: Matrix with res[i][j]  = dot_product(a[i], b[j])
    """
    if len(a.shape) == 1:
        a = a.unsqueeze(0)
    if len(b.shape) == 1:
        b = b.unsqueeze(0)
    if normalize:
        a = torch.nn.functional.normalize(a, p=2, dim=1)
        b = torch.nn.functional.normalize(b, p=2, dim=1)
    return torch.mm(a, b.transpose(0, 1))


def arccosh(x):
    """Compute the arcosh, numerically stable."""
    x = torch.clamp(x, min=1 + EPSILON)
    a = torch.log(x)
    b = torch.log1p(torch.sqrt(x * x - 1) / x)
    return a + b


def _iter_dataset(dataset: 'Optional[Dataset]') ->typing.Iterable:
    if dataset is None:
        return []
    return map(lambda x: x[0], DataLoader(dataset, batch_size=1, num_workers=0)
        )


def identify_dynamic_embeddings(data_point: 'DataPoint'):
    dynamic_embeddings = []
    if isinstance(data_point, Sentence):
        first_token = data_point[0]
        for name, vector in first_token._embeddings.items():
            if vector.requires_grad:
                dynamic_embeddings.append(name)
    for name, vector in data_point._embeddings.items():
        if vector.requires_grad:
            dynamic_embeddings.append(name)
    return dynamic_embeddings


def store_embeddings(data_points: 'Union[List[DT], Dataset]', storage_mode:
    'str', dynamic_embeddings: 'Optional[List[str]]'=None):
    if isinstance(data_points, Dataset):
        data_points = list(_iter_dataset(data_points))
    if storage_mode == 'none':
        dynamic_embeddings = None
    elif not dynamic_embeddings:
        dynamic_embeddings = identify_dynamic_embeddings(data_points[0])
    for data_point in data_points:
        data_point.clear_embeddings(dynamic_embeddings)
    if storage_mode == 'cpu':
        str(flair.device) != 'cpu'
        for data_point in data_points:
            data_point


def mdot(x, y):
    """Compute the inner product."""
    m = x.new_ones(1, x.size(1))
    m[0, 0] = -1
    return torch.sum(m * x * y, 1, keepdim=True)


def dist(x, y):
    """Get the hyperbolic distance between x and y."""
    return arccosh(-mdot(x, y))


class CosineDistance(torch.nn.Module):

    def forward(self, a, b):
        return -dot_product(a, b, normalize=True)


class EuclideanDistance(nn.Module):
    """Implement a EuclideanDistance object."""

    def forward(self, mat_1: 'Tensor', mat_2: 'Tensor') ->Tensor:
        """Returns the squared euclidean distance between each
        element in mat_1 and each element in mat_2.

        Parameters
        ----------
        mat_1: torch.Tensor
            matrix of shape (n_1, n_features)
        mat_2: torch.Tensor
            matrix of shape (n_2, n_features)

        Returns
        -------
        dist: torch.Tensor
            distance matrix of shape (n_1, n_2)

        """
        _dist = [torch.sum((mat_1 - mat_2[i]) ** 2, dim=1) for i in range(
            mat_2.size(0))]
        dist = torch.stack(_dist, dim=1)
        return dist


class HyperbolicDistance(nn.Module):
    """Implement a HyperbolicDistance object."""

    def forward(self, mat_1: 'Tensor', mat_2: 'Tensor') ->Tensor:
        """Returns the squared euclidean distance between each
        element in mat_1 and each element in mat_2.

        Parameters
        ----------
        mat_1: torch.Tensor
            matrix of shape (n_1, n_features)
        mat_2: torch.Tensor
            matrix of shape (n_2, n_features)

        Returns
        -------
        dist: torch.Tensor
            distance matrix of shape (n_1, n_2)

        """
        mat_1_x_0 = torch.sqrt(1 + mat_1.pow(2).sum(dim=1, keepdim=True))
        mat_2_x_0 = torch.sqrt(1 + mat_2.pow(2).sum(dim=1, keepdim=True))
        left = mat_1_x_0.mm(mat_2_x_0.t())
        right = mat_1[:, 1:].mm(mat_2[:, 1:].t())
        return arccosh(left - right).pow(2)


class LogitCosineDistance(torch.nn.Module):

    def forward(self, a, b):
        return torch.logit(0.5 - 0.5 * dot_product(a, b, normalize=True))


class NegativeScaledDotProduct(torch.nn.Module):

    def forward(self, a, b):
        sqrt_d = torch.sqrt(torch.tensor(a.size(-1)))
        return -dot_product(a, b, normalize=False) / sqrt_d


class PrototypicalDecoder(torch.nn.Module):

    def __init__(self, num_prototypes: 'int', embeddings_size: 'int',
        prototype_size: 'Optional[int]'=None, distance_function: 'str'=
        'euclidean', use_radius: 'Optional[bool]'=False, min_radius:
        'Optional[int]'=0, unlabeled_distance: 'Optional[float]'=None,
        unlabeled_idx: 'Optional[int]'=None, learning_mode: 'Optional[str]'
        ='joint', normal_distributed_initial_prototypes: 'bool'=False):
        super().__init__()
        if not prototype_size:
            prototype_size = embeddings_size
        self.prototype_size = prototype_size
        self.metric_space_decoder: 'Optional[torch.nn.Linear]' = None
        if prototype_size != embeddings_size:
            self.metric_space_decoder = torch.nn.Linear(embeddings_size,
                prototype_size)
            torch.nn.init.xavier_uniform_(self.metric_space_decoder.weight)
        self.prototype_vectors = torch.nn.Parameter(torch.ones(
            num_prototypes, prototype_size), requires_grad=True)
        if normal_distributed_initial_prototypes:
            self.prototype_vectors = torch.nn.Parameter(torch.normal(torch.
                zeros(num_prototypes, prototype_size)))
        self.prototype_radii: 'Optional[torch.nn.Parameter]' = None
        if use_radius:
            self.prototype_radii = torch.nn.Parameter(torch.ones(
                num_prototypes), requires_grad=True)
        self.min_radius = min_radius
        self.learning_mode = learning_mode
        assert (unlabeled_idx is None) == (unlabeled_distance is None
            ), "'unlabeled_idx' and 'unlabeled_distance' should either both be set or both not be set."
        self.unlabeled_idx = unlabeled_idx
        self.unlabeled_distance = unlabeled_distance
        self._distance_function = distance_function
        self.distance: 'Optional[torch.nn.Module]' = None
        if distance_function.lower() == 'hyperbolic':
            self.distance = HyperbolicDistance()
        elif distance_function.lower() == 'cosine':
            self.distance = CosineDistance()
        elif distance_function.lower() == 'logit_cosine':
            self.distance = LogitCosineDistance()
        elif distance_function.lower() == 'euclidean':
            self.distance = EuclideanDistance()
        elif distance_function.lower() == 'dot_product':
            self.distance = NegativeScaledDotProduct()
        else:
            raise KeyError(f'Distance function {distance_function} not found.')
        self

    @property
    def num_prototypes(self):
        return self.prototype_vectors.size(0)

    def forward(self, embedded):
        if self.learning_mode == 'learn_only_map_and_prototypes':
            embedded = embedded.detach()
        if self.metric_space_decoder is not None:
            encoded = self.metric_space_decoder(embedded)
        else:
            encoded = embedded
        prot = self.prototype_vectors
        radii = self.prototype_radii
        if self.learning_mode == 'learn_only_prototypes':
            encoded = encoded.detach()
        if self.learning_mode == 'learn_only_embeddings_and_map':
            prot = prot.detach()
            if radii is not None:
                radii = radii.detach()
        distance = self.distance(encoded, prot)
        if radii is not None:
            distance /= self.min_radius + torch.nn.functional.softplus(radii)
        if self.unlabeled_distance:
            distance[..., self.unlabeled_idx] = self.unlabeled_distance
        scores = -distance
        return scores

    def enable_expectation_maximization(self, data: 'FlairDataset', encoder:
        'DefaultClassifier', exempt_labels: 'List[str]'=[], mini_batch_size:
        'int'=8):
        """Applies monkey-patch to train method (which sets the train flag).

        This allows for computation of average prototypes after a training
        sequence."""
        decoder = self
        unpatched_train = encoder.train

        def patched_train(mode: 'bool'=True):
            unpatched_train(mode=mode)
            if mode:
                logger.info('recalculating prototypes')
                with torch.no_grad():
                    decoder.calculate_prototypes(data=data, encoder=encoder,
                        exempt_labels=exempt_labels, mini_batch_size=
                        mini_batch_size)
        encoder.train = patched_train

    def calculate_prototypes(self, data: 'FlairDataset', encoder:
        'DefaultClassifier', exempt_labels: 'List[str]'=[], mini_batch_size=32
        ):
        """
        Function that calclues a prototype for each class based on the euclidean average embedding over the whole dataset
        :param data: dataset for which to calculate prototypes
        :param encoder: encoder to use
        :param exempt_labels: labels to exclude
        :param mini_batch_size: number of sentences to embed at same time
        :return:
        """
        with torch.no_grad():
            dataloader = DataLoader(data, batch_size=mini_batch_size)
            new_prototypes = torch.zeros(self.num_prototypes, self.
                prototype_size, device=flair.device)
            counter: 'Counter' = Counter()
            for batch in tqdm(dataloader):
                logits, labels = encoder.forward_pass(batch)
                if len(labels) > 0:
                    if self.metric_space_decoder is not None:
                        logits = self.metric_space_decoder(logits)
                    for logit, label in zip(logits, labels):
                        counter.update(label)
                        idx = encoder.label_dictionary.get_idx_for_item(label
                            [0])
                        new_prototypes[idx] += logit
                store_embeddings(batch, storage_mode='none')
            for label, count in counter.most_common():
                average_prototype = new_prototypes[encoder.label_dictionary
                    .get_idx_for_item(label)] / count
                new_prototypes[encoder.label_dictionary.get_idx_for_item(label)
                    ] = average_prototype
            for label in exempt_labels:
                label_idx = encoder.label_dictionary.get_idx_for_item(label)
                new_prototypes[label_idx] = self.prototype_vectors[label_idx]
            self.prototype_vectors.data = new_prototypes


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'num_prototypes': 4, 'embeddings_size': 4}]
