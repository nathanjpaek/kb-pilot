import torch
from torch import nn
import torch.nn.functional as F


class PositionwiseFeedForward(nn.Module):
    """
	Implementa la ecuación del feed_forward networks para el transformer
	"""
    """
	Se implementa una red de dos capas sencillas con una ReLU en medio

	FFN(x) = max(0, xW_1 + b_1)W_x + b_2 
	"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
		Constructor del bloque o capa FFN
		
		Parámetros
		d_model: Dimensión del modelo
		d_ff: Dimensión del feed_forward
		dropout: Probabilidad de dropout
		"""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
		Paso hacia delante 

		Parámetros:
		x: Entradas
		"""
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


def get_inputs():
    return [torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'d_model': 4, 'd_ff': 4}]
