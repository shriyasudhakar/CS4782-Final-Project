import torch
from torch import nn

class BasicEncoder(nn.Module):
    def __init__(self, D:int):
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3,1,"same"),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32+D, 32, 3,1,"same"),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3,1,"same"),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        # omit leaky relu and batch norm for last block
        self.conv4 = nn.Conv2d(32,3,3,1,"same")

    def pre_forward(self, cover_image, message):
        """
        Similar architecture building across all encoders.
        """
        a = self.conv1(cover_image)
        a = torch.cat(a, message) # becomes N x 32 + D x H x W
        b = self.conv2(a)
        return b

    def forward(self, cover_image, message):
        """
        Parameters
        ----------
        cover_image : N x 3 x H x W Tensor
            Cover image C to input to Encoder network, H x W is size of image and 3 RGB channels. N is batch size
        message: {0, 1} N x D x H x W Tensor
            Binary data tensor, secret message M. D is the number of bits to hide in each pixel of cover image.

        Returns
        -------
        output: N x 3 x H x W Tensor
            Output encoded steganographic image S
        """
        x = self.pre_forward(cover_image,message)
        x = self.conv3(x)
        x = self.conv4(x)


class ResidualEncoder(BasicEncoder):
    def forward(self, cover_image, message):
        x = super().forward(cover_image,message)
        return cover_image + x
        
          
