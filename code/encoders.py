import torch
from torch import nn

class BasicEncoder(nn.Module):
    def __init__(self, D:int):
        """
        Parameters
        ----------
        
        D: int
            the number of bits to hide in each pixel of cover image

        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3,1,"same"),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32+D, 32, 3,1,"same"),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, 3,1,"same"),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        # omit leaky relu and batch norm for last block
        self.conv3 = nn.Conv2d(32,3,3,1,"same")

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
        x = self.conv1(cover_image)
        x = torch.cat([x, message], dim = 1) # becomes N x 32 + D x H x W
        # pass concatenated input into the next two blocks sequentially
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.tanh(x) # get back to expected image output range [-1,1] since this predicts entire image
        return x


class ResidualEncoder(nn.Module):
    # def forward(self, cover_image, message):
    #     x = super().forward(cover_image,message)
    #     return cover_image + x

    def __init__(self, D:int):
        """
        Parameters
        ----------
        
        D: int
            the number of bits to hide in each pixel of cover image

        """
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3,1,"same"),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32+D, 32, 3,1,"same"),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, 3,1,"same"),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        # omit leaky relu and batch norm for last block
        self.conv3 = nn.Conv2d(32,3,3,1,"same")

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
        x = self.conv1(cover_image)
        x = torch.cat([x, message], dim = 1) # becomes N x 32 + D x H x W
        # pass concatenated input into the next two blocks sequentially
        x = self.conv2(x)
        x = self.conv3(x)
        # no Tanh since it now predicts a residual from the input image
        return cover_image + x
    
class DenseEncoder(nn.Module):
    def __init__(self, D:int):
        """
        Parameters
        ----------
        
        D: int
            the number of bits to hide in each pixel of cover image

        """
        super().__init__()
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
            nn.Conv2d(32*2 + D, 32, 3,1,"same"),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        # omit leaky relu and batch norm for last block
        self.conv4 = nn.Conv2d(32*3 + D,3,3,1,"same")

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
        # list of intermediate outputs
        xs = []
        x = self.conv1(cover_image)
        xs.append(x)
        x = torch.cat(xs + [message], dim = 1)
        x = self.conv2(x)
        xs.append(x)
        x = torch.cat(xs + [message], dim=1)
        x = self.conv3(x)
        xs.append(x)
        x = torch.cat(xs + [message], dim=1)
        x = self.conv4(x)
        # no Tanh since it now predicts a residual from the input image
        return cover_image + x