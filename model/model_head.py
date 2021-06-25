import torch
import torch.nn as nn

from model_parts import SincConv, SpeechEmbedder


class D_vector(nn.Module):
    def __init__(self):
        super(D_vector, self).__init__()
        self.sconv = SincConv(in_channels=1, out_channels=80, kernel_size=251, sample_rate=16000)

    def forward(self, x):
        out = self.sconv(x)
        print(out.shape)
        raise NotImplementedError


if __name__ == "__main__":
    x = torch.rand([4, 64, 10])
    model = D_vector()
