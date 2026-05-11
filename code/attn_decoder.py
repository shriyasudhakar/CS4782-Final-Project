import torch
import torch.nn as nn

class AttentionDecoder(nn.Module):
    def __init__(self, D: int, window_size=8, num_heads=4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, "same"),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, "same"),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32 * 2, 32, 3, 1, "same"),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32)
        )

        self.attn = WindowAttention(32, window_size=window_size, num_heads=num_heads)
        self.norm = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32 * 3, D, 3, 1, "same")

    def forward(self, stego_image):
        xs = []
        x = self.conv1(stego_image)
        xs.append(x)
        x = self.conv2(torch.cat(xs, dim=1))
        xs.append(x)
        x = self.conv3(torch.cat(xs, dim=1))
        # attention on the same scale as the encoder
        x = x + self.norm(self.attn(x))
        xs.append(x)
        return self.conv4(torch.cat(xs, dim=1))