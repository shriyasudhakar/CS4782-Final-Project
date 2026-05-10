# Paper: SteganoGAN: High Capacity Image Steganography with GANs
Shriya Sudhakar, Kiran Mitra, Nidhi Soma, Niti Goyal

## Introduction
The purpose of this repository is a reimplementation of [SteganoGAN: High Capcaity Image Steganography with GANs](https://arxiv.org/pdf/1901.03892) by Zhang et. al, 2019. The main contribution of this paper is to introduce a new state-of-the-art deep learning technique for Steganography. The authors were able to acheive a record payload of 4.4 bits per
pixel for natural images from multiple datasets that evade detection by tradiditonal and deep-learning beased steganalysis tools.

## Chosen Result
We chose to replicate the paper's Div2K resulting metrics on depths=1,3,6 for all Basic, Residual, and Dense encoders.
<img width="1378" height="277" alt="image" src="https://github.com/user-attachments/assets/d5aafd99-0c9b-4385-ad47-46f776041a88" />

We also used StegExpose, a classic open-source steganalysis tool, to evaluate the detection performance of all model types implemented, replicating Figure 5. Note that we only used 50 samples to replicate this result and plotted depths=1,3,6 while the paper used 1000 samples and plotted depth 6.
<img width="708" height="796" alt="image" src="https://github.com/user-attachments/assets/67dd2b7b-5fc8-420c-87fd-d6a10ed8d7bf" />


## GitHub Contents
Important folders:
/code - contains main components of model like encoders, decoders, and critic

/data - DIV2K train, validation, and test data used for evaluation

/results - separate folder to replicate results demonstrated in paper 

## Re-implementation Details


## Reproduction Steps
Run our training script in /code/train.ipynb which will download data locally and install necessary libraries, preprocess images, and run the training loop. It also has all our model architecture and supports wandb logging for evaluation metrics. Our notebook does multiple runs with different Encoders and different depths so you can reproduce our table of results.

To reproduce the StegExpose plot, you will have to download the tool code: https://github.com/b3dk7/stegexpose. You can use the 100 validation images in /data/coco_val_images. Run /data/crop_real_images.py to process the real images, and code/generate_eval_images.py to generate steganographic versions. Then combine the first 50 real images and the last 50 steganographic images to create your test set. You can then run StegExpose with your the test image folder.

We used a T4 GPU in Google Colab. Training for 32 epochs took about 80 minutes.

## Conclusion
We learned:
- The model is able to adapt to OOD images through our grayscale/blurred ablations.
- At higher depths, the model struggles with encoding more information impercievably
- There is a tradeoff between accuracy metrics (accuracy of bits recovered, RS-BPP) and fidelity metrics (PSNR, SSIM)
- AttentionEncoder likely will outperform the paper's other encoder methods
## References
[1] Zhang, Kevin Alex, et al. "SteganoGAN: High Capacity Image Steganography with GANs." arXiv preprint arXiv:1901.03892 (2019).

[2] Boehm, Benedikt. "Stegexpose - A Tool for Detecting LSB Steganography." arXiv preprint arXiv:1410.6656 (2014).

[3] Liu, Ze e, et al. “Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.” arXiv preprint arXiv:2103.14030 (2021)

## Acknowledgements
This project was undertaken for the final course project in CS 4782: Introduction to Deep Learning at Cornell University in Spring 2026. 
