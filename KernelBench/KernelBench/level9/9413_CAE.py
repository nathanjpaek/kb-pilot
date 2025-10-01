import torch
import torch.nn as nn


class CAE(nn.Module):
    """
    The Cobb Angle Estimator (CAE), which :
        1. maps #nDense1 landmark features to #nDense2 angle features
        2. adds the #nDense2 angle features (from step 1) to  #nDense2 landmarks features (from previous layer)
        3. maps summed #nDense2 angle features (from step 2) to #nDense3 angles estimations
    (Paper: Automated comprehensive Adolescent Idiopathic Scoliosis assessment using MVC-Net)
    """

    def __init__(self, nDense1, nDense2, nDense3):
        """
        Initialize the building blocks of CAE.
        :param nDense1: #features from the previous SLE (#features is arbitrary) (see 'SLE' class for details)
        :param nDense2: (#features = 2 * #landmarks)
        :param nDense3: (#features = #angles)
        """
        super(CAE, self).__init__()
        self.dense1 = nn.Linear(nDense1, nDense2)
        self.tanh = nn.Tanh()
        self.dense2 = nn.Linear(nDense2, nDense3)

    def forward(self, lm_features, lm_coordinates):
        """
        The forwarding pass of CAE, which make cobb angles estimations from two landmark features.
        :param lm_features: the output 'y1' of SLE (see 'SLE' class for details)
        :param lm_coordinates: the output 'y2' of SLE (see 'SLE' class for details)
        :return angs: #nDense3 angle estimations
        """
        out_dense1 = self.dense1(lm_features)
        ang_features = self.tanh(out_dense1)
        ang_sumFeatures = ang_features + lm_coordinates
        angs = self.dense2(ang_sumFeatures)
        return angs


def get_inputs():
    return [torch.rand([4, 4, 4, 4]), torch.rand([4, 4, 4, 4])]


def get_init_inputs():
    return [[], {'nDense1': 4, 'nDense2': 4, 'nDense3': 4}]
