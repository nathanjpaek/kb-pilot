import torch
from torch import nn


class LayerNorm(nn.Module):
    """
	Construye la capa de normalización
	"""

    def __init__(self, features, eps=1e-06):
        """
		Constructor LayerNorm

		Parámetros:
		features: Tamaño del vector
		eps: Diferencia para la división
		"""
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        """
		Paso hacia delante de la capa de normalización
		
		Parámetros
		x: Entradas (salidas del encoder)
		"""
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'features': 4}]
