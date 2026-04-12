import torch
import torch.nn as nn

class Critic(nn.Module):
    """
    Spec: See Section 3.2.3 (Equation 7)
    Input Size: (N, 3, H, W)
    Output Size: (N, 1, H, W)
    """

    def __init__(self, hidden_dim=32):
        super(Critic, self).__init__()
        # (same as decoder)
        # conv_D-->D' blocks: (page 3, 4 in the paper)
        # (1) Conv2d with in_channel D, out_channel D', kernel size 3, stride 1, padding same (so padding = 1)
        # (2) LeakyRelU activation
        # (3) BatchNormalization
        # omit activation and batch norm if convolution block is last block in network

        self.hidden_dim = hidden_dim
        # cat operation: concat along the depth axis (the channel axis)
        self.a = nn.Conv2d(3, self.hidden_dim, 3, 1, 1) # Conv3->32
        self.b = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1, 1) # Conv32->32
        self.c = nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1, 1) # Conv32->32
        self.d = nn.Conv2d(self.hidden_dim, 1, 3, 1, 1) # Conv32->1

        self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.batchnorm = nn.BatchNorm2d(self.hidden_dim)

    
    def forward(self, x):
        a_x = self.batchnorm(self.leakyrelu(self.a(x)))
        b_x = self.batchnorm(self.leakyrelu(self.b(a_x)))
        c_x = self.batchnorm(self.leakyrelu(self.c(b_x)))
        d_x = self.d(c_x) # N x 1 x H x W
        d_x = d_x.mean(dim=[2, 3]).squeeze(1) # find single mean for each channel (mean across H x W dimension --> N x 1 --> (N, ) as a result of squeeze)

        return d_x # dimension: (N, )