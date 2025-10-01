import torch
from torch import nn
import torch.nn.functional as F


class Generator(nn.Module):
    """
	Define el paso de generación lineal + softmax
	"""

    def __init__(self, d_model, vocab):
        """
		Constructor del generador lineal

		Parámetros:
		d_model: Dimensión del modelo
		vocab: Tamaño del vocabulario
		"""
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        """
		Paso hacia delante del generador
		
		Parámetros:
		x: Entradas para el generador (la salida de los
		decodificadores)

		return probabilidades generadas por softmax
		"""
        return F.log_softmax(self.proj(x), dim=-1)


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'vocab': 4}]
