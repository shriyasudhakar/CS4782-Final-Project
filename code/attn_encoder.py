from torch import nn
import torch.nn.functional as F

class WindowAttention(nn.Module):
    def __init__(self, channels, window_size=8, num_heads=4):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        N, C, H, W = x.shape
        ws = self.window_size

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x.shape

        x = x.view(N, C, Hp // ws, ws, Wp // ws, ws)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(-1, C, ws, ws)

        # QKV
        qkv = self.qkv(x)
        qkv = qkv.view(-1, 3, self.num_heads, self.head_dim, ws * ws)
        qkv = qkv.permute(1, 0, 2, 4, 3)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = out.permute(0, 1, 3, 2).contiguous()
        out = out.view(-1, C, ws, ws)
        out = self.proj(out)

        out = out.view(N, Hp // ws, Wp // ws, C, ws, ws)
        out = out.permute(0, 3, 1, 4, 2, 5).contiguous()
        out = out.view(N, C, Hp, Wp)

        out = out[:, :, :H, :W]
        return out


class AttentionEncoder(nn.Module):
    def __init__(self, D: int, window_size=8, num_heads=4):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, "same"),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32 + D, 32, 3, 1, "same"),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32 * 2 + D, 32, 3, 1, "same"),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(32)
        )

        self.attn = WindowAttention(32, window_size=window_size, num_heads=num_heads)
        self.norm = nn.BatchNorm2d(32)

        self.conv4 = nn.Conv2d(32 * 3 + D, 3, 3, 1, "same")

    def forward(self, cover_image, message):
        xs = []
        x = self.conv1(cover_image)
        xs.append(x)

        x = self.conv2(torch.cat(xs + [message], dim=1))
        xs.append(x)

        x = self.conv3(torch.cat(xs + [message], dim=1))
        # apply window attention + residual on the third feature map
        x = x + self.norm(self.attn(x))
        xs.append(x)

        x = self.conv4(torch.cat(xs + [message], dim=1))
        return cover_image + x