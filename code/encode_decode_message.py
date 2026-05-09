"""
End-to-end English-message round trip through a trained SteganoGAN encoder + decoder.
`python encode_decode_message.py`
"""

import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from reedsolo import RSCodec

from decoder import Decoder
from encoders import BasicEncoder, ResidualEncoder, DenseEncoder


MODEL_PATH = Path("../checkpoints/basic_D6.pt")
ENCODER_CLS = ResidualEncoder   # BasicEncoder | ResidualEncoder | DenseEncoder
DEPTH = 6                       # must match what the checkpoint was trained with
IMAGE_PATH = Path("../data/kilian.png")
MESSAGE = "The rain began three hours before dusk and showed no intention of stopping. By the time the last bus left the mountain road, the village of Black Hollow had disappeared behind a curtain of silver water. Roofs blurred. Pine trees bent like listeners leaning toward a secret. The river beyond the bridge swelled dark and restless beneath the storm. Inside the station café, Mira Vale counted the same coins three times. Not because she needed to. Because counting kept her from thinking. Youll wear grooves into them, said the old owner from behind the counter. Mira slid the coins into her pocket. Wouldn’t be the worst thing I’ve ruined. The owner snorted softly. You’re twenty-four. You talk like a widow. Maybe I’m practicing. The café smelled of wet wool, burnt coffee, and cedar smoke drifting from the stove. Outside, thunder rolled through the valley with enough force to rattle the windows. No more customers would come tonight. No one traveled to Black Hollow during flood season unless they had nowhere else to go. Mira knew the feeling. She rose from her booth and gathered the stack of untouched newspapers from the corner table. Her shift had ended an hour ago, but she lingered every evening now, unwilling to return to the little house at the edge of the woods. It had been her father’s house once. Then his grave. Then her inheritance. Funny how places changed names without changing shape. You should head home before the river climbs higher, the owner warned. I’ll survive. That confidence is how mountains kill people. Mira pulled on her coat and stepped into the storm. Rain hammered the earth so hard it bounced."
STEGO_OUT = Path("../data/kilian_encoded.png")
RESIDUAL_OUT = Path("../data/kilian_residual.png")
USE_RS = False  # set False to skip Reed-Solomon (faster, less error-resilient)


# ---------- 1. build encoder + decoder, load trained weights ----------

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"[device] {device}")

encoder = ENCODER_CLS(DEPTH).to(device)
decoder = Decoder(DEPTH).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
encoder.load_state_dict(checkpoint["encoder_state_dict"])
decoder.load_state_dict(checkpoint["decoder_state_dict"])
encoder.eval()
decoder.eval()
print(f"Encoder: {ENCODER_CLS} Depth={DEPTH}")


# ---------- 2. load the cover image as a (1, 3, H, W) tensor in [-1, 1] ----------

img = Image.open(IMAGE_PATH).convert("RGB")
#convert pixel values from [0,255] to [-1,1]
arr = np.asarray(img, dtype=np.float32) / 127.5 - 1.0
#convert to pytorch tensor, reorder to (3,H,W), add batch dimension 1
cover = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
_, _, H, W = cover.shape
print(f"image: {IMAGE_PATH.name}  H={H}  W={W}  capacity={H * W * DEPTH} bits")


# ---------- 3. turn MESSAGE into a bit list: utf8 -> zlib compression -> reed-solomon -> bits ----------

#Reed-Soloman error corrector with 250 parity bytes
#so that decoder can recover message even if 125 bytes get corrupted
compressed = bytearray(zlib.compress(MESSAGE.encode("utf-8")))
if USE_RS:
    rs = RSCodec(250)
    encoded_bytes = rs.encode(compressed)
else:
    rs = None
    encoded_bytes = compressed

#convert the RS-encoded bytes into flat list of bits
message_bits = []
for byte in encoded_bytes:
    bits_str = bin(byte)[2:].rjust(8, "0") #[2:] gets rid of '0b', then pads with 0s at the beginning
    message_bits.extend(int(b) for b in bits_str)
#each byte is exxactly 8 bits

# append 32 zeros at the end, used as the split marker between repeats on decode
message_bits += [0] * 32
print(f"[msg] '{MESSAGE}' -> {len(message_bits) - 32} bits (+32 zeros)")


# ---------- 4. tile the message bits repeatedly to fill (D, H, W), then run the encoder ----------

capacity = H * W * DEPTH
print(f"image capacity: {capacity} bits  |  message requires: {len(message_bits)} bits  |  tiles: {capacity // len(message_bits)}x")
if len(message_bits) > capacity:
    raise ValueError(f"message needs {len(message_bits)} bits, image only fits {capacity}")

# tile the message bits
payload_bits = (message_bits * (capacity // len(message_bits) + 1))[:capacity]
# convert flat list of bits to the shape of the image
payload = torch.FloatTensor(payload_bits).view(1, DEPTH, H, W).to(device)
print(f"[tile] message payload shape {tuple(payload.shape)}")

#produce the stego image. clip any pixels that drift outside the valid range
with torch.no_grad():
    stego = encoder(cover, payload).clamp(-1.0, 1.0)


# ---------- 5. save the stego PNG (uint8 quantization happens here) ----------

#convert stego tensor to a saveable image
# remove the batch dimension, move to CPU, reshape to (H,W,3), convert to numpy
stego_arr = stego[0].detach().cpu().permute(1, 2, 0).numpy()
#convert [-1,1] back to [0,255]
stego_arr = ((stego_arr + 1.0) * 127.5).round().astype(np.uint8)
Image.fromarray(stego_arr).save(STEGO_OUT)
print(f"stego image saved to {STEGO_OUT}")

# residual between cover and stego: amplify differences by 10 so small changes are visible
cover_v = (cover[0].cpu().permute(1, 2, 0).numpy() + 1.0) * 0.5
stego_v = (stego[0].detach().cpu().permute(1, 2, 0).numpy() + 1.0) * 0.5
residual_arr = (np.abs(stego_v - cover_v) * 10).clip(0, 1)
residual_arr = (residual_arr * 255).round().astype(np.uint8)
Image.fromarray(residual_arr).save(RESIDUAL_OUT)
print(f"residual image saved to {RESIDUAL_OUT}")


# ---------- 6. reload the PNG and run the decoder (mimics a real recipient) ----------

#convert image to a tensor with values in [-1,1] 
img2 = Image.open(STEGO_OUT).convert("RGB")
arr2 = np.asarray(img2, dtype=np.float32) / 127.5 - 1.0
stego_reloaded = torch.from_numpy(arr2).permute(2, 0, 1).unsqueeze(0).to(device)

#decoder outputs one logit per bit position
# logit > 0 means bit 1, logit <= 0 means bit 0
with torch.no_grad():
    logits = decoder(stego_reloaded)
    decoded_bits = (logits.view(-1) > 0).int().cpu().numpy().tolist()


# ---------- 7. bits -> bytes -> split on 4 zero bytes -> RS-decode each -> majority vote ----------

decoded_bytes = bytearray()
for i in range(len(decoded_bits) // 8):
    byte = decoded_bits[i * 8:(i + 1) * 8]
    decoded_bytes.append(int("".join(str(b) for b in byte), 2))

candidates = Counter()
for chunk in decoded_bytes.split(b"\x00\x00\x00\x00"):
    if not chunk:
        continue
    try:
        if USE_RS:
            fixed = rs.decode(bytearray(chunk))
            if isinstance(fixed, tuple):  # reedsolo >=1.5 returns (msg, msg_with_ecc, errata_pos)
                fixed = fixed[0]
        else:
            fixed = chunk
        text = zlib.decompress(bytes(fixed)).decode("utf-8")
        candidates[text] += 1
    except Exception:
        continue

recovered = candidates.most_common(1)[0][0] if candidates else None


# ---------- 8. report ----------

print()
correct_bits = sum(a == b for a, b in zip(payload_bits, decoded_bits))
total_bits = len(payload_bits)
print("=" * 60)
print(f"original:   {MESSAGE!r}")
print(f"recovered:  {recovered!r}")
print(f"match:      {recovered == MESSAGE}")
print(f"bit accuracy: {correct_bits}/{total_bits} ({100 * correct_bits / total_bits:.2f}%)")
print("=" * 60)
