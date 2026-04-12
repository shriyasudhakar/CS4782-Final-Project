import torch
import torch.nn as nn

class Decoder(nn.Module):
    """
    Spec: See Section 3.2.2 (Equation 6)
    Input Size: (N, 3, H, W)
    Output Size: (N, D, H, W)

    Experiments: Ablate over D={1, 3, 6}
    """

    def __init__(self, D, hidden_dim=32):
        super(Decoder, self).__init__()
        # conv_D-->D' blocks: (page 3, 4 in the paper)
        # (1) Conv2d with in_channel D, out_channel D', kernel size 3, stride 1, padding same (so padding = 1)
        # (2) LeakyRelU activation
        # (3) BatchNormalization
        # omit activation and batch norm if convolution block is last block in network

        self.hidden_dim = hidden_dim
        # cat operation: concat along the depth axis (the channel axis)
        self.a = nn.Conv2d(3, self.hidden_dim, 3, 1, 1)
        self.b = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1, 1)
        self.c = nn.Conv2d(2 * self.hidden_dim, self.hidden_dim, 3, 1, 1) # 64 because we concat a and b which each have 32 channels
        self.d = nn.Conv2d(3 * self.hidden_dim, D, 3, 1, 1) # output channels is D, the hidden_dim
        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.batchnorm = nn.BatchNorm2d(self.hidden_dim) # a, b, c, d have same hidden size (output channel size)

    
    def forward(self, x):
        a_x = self.batchnorm(self.leakyrelu(self.a(x)))
        b_x = self.batchnorm(self.leakyrelu(self.b(a_x)))
        c_x = self.batchnorm(self.leakyrelu(self.c(torch.concat([a_x, b_x], dim=1)))) # concat along the chanel dim
        d_x = self.d(torch.concat([a_x, b_x, c_x], dim=1))

        return d_x