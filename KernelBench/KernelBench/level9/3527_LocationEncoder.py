import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class LocationEncoder(nn.Module):

    def __init__(self, pedestrian_num, input_size, hidden_size, batch_size):
        super(LocationEncoder, self).__init__()
        self.pedestrian_num = pedestrian_num
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, self.hidden_size)
        self.soft = nn.Softmax(dim=1)
        pass

    def forward(self, data):
        outputs = self.get_hidden_output(data)
        output = self.Attention(outputs, outputs)
        return output

    def get_hidden_output(self, data):
        output_list = []
        for idx in range(0, self.pedestrian_num):
            output = F.relu(self.fc1(data[:, idx]))
            output = F.relu(self.fc2(output))
            output = self.fc3(output)
            output_list.append(output)
        outputs = torch.stack(output_list, 1)
        return outputs

    def Attention(self, input_data, target_data):
        Attn = torch.bmm(target_data, input_data.transpose(1, 2))
        Attn_size = Attn.size()
        Attn = Attn - Attn.max(2)[0].unsqueeze(2).expand(Attn_size)
        exp_Attn = torch.exp(Attn)
        Attn = exp_Attn / exp_Attn.sum(2).unsqueeze(2).expand(Attn_size)
        return Attn

    def get_spatial_affinity(self, data):
        output = torch.zeros(self.batch_size, self.pedestrian_num, self.
            pedestrian_num)
        for batch in range(0, self.batch_size):
            for i in range(0, self.pedestrian_num):
                row_data = torch.Tensor([])
                for j in range(0, i + 1):
                    row_data = torch.cat([row_data, torch.dot(data[batch][i
                        ], data[batch][j]).unsqueeze(0)], dim=0)
                output[batch, i, 0:i + 1] = row_data
            for i in range(0, self.pedestrian_num):
                col_data = output[batch, :, i].view(1, -1)
                output[batch, i, :] = col_data
            output[batch] = self.soft(output[batch])
        """
        outputs will be like this :
        <h1, h1>, <h2, h1>, <h3, h1> ...
        <h2, h1>, <h2, h2>, <h3, h2> ...
        <h3, h1>, <h3, h2>, <h3, h3> ...
        ......
        """
        return output

    def softmax(self, data):
        output = torch.zeros(self.batch_size, self.pedestrian_num, self.
            pedestrian_num)
        exp_data = torch.exp(data)
        for batch in range(0, self.batch_size):
            for i in range(0, self.pedestrian_num):
                count = 0
                for j in range(0, self.pedestrian_num):
                    count += exp_data[batch][max(i, j)][min(i, j)].item()
                for j in range(0, self.pedestrian_num):
                    output[batch][i][j] = exp_data[batch][max(i, j)][min(i, j)
                        ].item() / count
        return output


def get_inputs():
    return [torch.rand([4, 4, 4])]


def get_init_inputs():
    return [[], {'pedestrian_num': 4, 'input_size': 4, 'hidden_size': 4,
        'batch_size': 4}]
