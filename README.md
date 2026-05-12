# Paper: SteganoGAN: High Capacity Image Steganography with GANs
Shriya Sudhakar (ss3576), Kiran Mitra (km936), Nidhi Soma (ns848), Niti Goyal (ng459)

## Introduction
This repository is a re-implementation of [SteganoGAN: High Capacity Image Steganography with GANs](https://arxiv.org/pdf/1901.03892) by Zhang et al. as part of CS 4782. SteganoGAN uses a GAN-based encoder decoder and critic to hide binary message inside images, achieving a record payload of 4.4 bits per pixel while evading detection by traditional and deep-learning-based steganalysis tools. Additionally, this repo introduces a novel attention encoder and decoder (surpassing the paper's results), loss ablations, and model robustness experiments.

## Chosen Result
We chose to replicate the paper's Div2K resulting metrics on depths=1,3,6 for all Basic, Residual, and Dense encoders. This table represents the paper's core empirical claims.
<img width="1378" height="277" alt="image" src="https://github.com/user-attachments/assets/d5aafd99-0c9b-4385-ad47-46f776041a88" />
Table: Comparison of 4 key metrics against different encoders, performed on the DIV2K dataset [Zhang et al. 2019]

We also used StegExpose, a classic open-source steganalysis tool, to evaluate the detection performance of all model types implemented, replicating Figure 5. Note that we only used 50 samples to replicate this result and plotted depths=1,3,6 while the paper used 1000 samples and plotted depth 6.
<img width="708" height="796" alt="image" src="https://github.com/user-attachments/assets/67dd2b7b-5fc8-420c-87fd-d6a10ed8d7bf" />

## GitHub Contents
**`requirements.txt`** - all Python dependencies; install with `pip install -r requirements.txt`

Important folders:
**/code** - contains main components of model like encoders, decoders, and critic
**- `train.ipynb` - Python notebook in which users can train configurations of models to replicate Table 1 from the original paper. It contains all necessary code, including encoders, decoders, critics, train loop, etc.**
- `train_attentionencoder.ipynb` - notebook specifically meant for our encoder and decoder ablation using Attention.
- `encoders.py` - contains all 3 main types of decoders from the paper: Basic, Dense, and Residual
- `attn_encoder.py` - contains the implementation of our novel attention encoder
- `attn_decoder.py` - contains the implementation of our novel attention decoder
- `decoder.py` - contains the main decoder that the paper described
- `crop_real_images.py` - processes real images to be normalized and center cropped to 360x360 to match steganographic generation. Should be run when making folders for StegExpose
- `critic.py` - contains the complete critic implementation
- `datasetloader.py` - loads in data (from a given data directory), and performs transformations (such as cropping) as described in the paper
- `evaluate.py` - contains evaluation functions for the 4 metrics: Decoding Accuracy, Reed Solomon Bits Per Pixel (RS-BPP), Peak Signal to Noise Ratio (PSNR), and Structural Similarity Index (SSIM)
- `encode_decode_message.py` - end-to-end demonstration that encodes an English message into a cover image using a trained encoder, saves the stego image and residual, then decodes and verifies recovery with bit accuracy reporting
- `generate_eval_images.py` - given a directory of cover images, generates steganographic versions and residuals for visualization
- `grayscale_blur_ablation.py` - performs our model robustness experiment, where we provide grayscale and blurred images to the model to understand (1) whether the model is able to perform on OOD images and (2) how the model hides messages in these new types of images
- `plot_stegexpose_rocauc.py` - given output CSV from StegExpose of detection results, plots the ROC AUC curve for the Dense model at different depths. May have to change path to match local location of StegExpose

**/data** - mainly contains DIV2K validation, and test data used for evaluation
- since the full datasets are too large to upload to github, we've included instructions on how to download the data in the README.md in the /data folder
- `/DIV2K_valid_HR` - contains our validation dataset of 100 images from DIV2K used to evaluate our main table of metrics
- `/coco_val_images` - contains the 100 images from COCO used for StegExpose evaluation. Also contains a notebook `coco_100_images.ipynb` to download these images.
- `profs` - contains images of our professors! (used in our poster)
- `profs_outputs/Dense_D=1` - contains steganographic images and residuals for our professors with the Dense model at Depth=1 (also used in our poster)
- `test_dense1`, `test_dense3`, `test_dense6` - contains the processed COCO cover and steganographic images used for our StegExpose evaluation

**/checkpoints** - contains pre-trained model weights for all encoder types (Basic, Residual, Dense, Attention) at depths 1, 3, 6, including ablation variants (`/with_scaling`, `/no_critic`). Scripts in `/code` load from this folder by default.

**/results
**- contains our main replication of table 1 (`table_metrics.png`) along with various training experiments (scaling MSE and removing critic), as well as results from the StegExpose tool (`stegexpose.png`).

**/poster
**- `SteganoGAN CS 4782.pdf` contains our poster that was presented at the final poster session

**/report
**- `steganogan_2page_report.pdf` contains our final report and analysis

## Re-implementation Details

We implemented Basic, Residual, and Dense encoders, DenseDecoder, and Critic across data depths D={1, 3, 6}, trained on DIV2K (800 train, 100 val) for 32 epochs with Adam (lr=1e-4). We also introduce a novel attention-based encoder/decoder (Window Multi-Head Self-Attention, performed within 8×8 windows across 4 heads) that outperforms the paper's proposed architecture. The key fix we made to match the paper's main result on PSNR and SSIM (image fidelity) was scaling the MSE loss term by 100 MSE loss. Lastly we evaluated the robustness of our models to image distribution shifts (StegExpose on COCO images), and tested adaptability on grayscale and blurred images.

## Reproduction Steps

1. **Dependencies:** Install required libraries with:
```bash
pip install -r requirements.txt
```
2. Train: run `code/train.ipynb`, which downloads DIV2K train and val splits, trains all configs (Basic, Dense, Residual). run `code/train_attentionencoder.ipynb`. You will need a GPU. All experiments were run on a T4 GPU in Google Colab for 32 epochs. Takes 80 minutes per model config.
3. StegExpose: First, download the tool code: https://github.com/b3dk7/stegexpose. Run code/crop_real_images.py to perform image pre-processing, and then run `code/generate_eval_images.py` to generate the steganographic images. You can use `data/coco_val_images` as your image set. Then, run `code/generate_eval_images.py` to generate steganographic images (will need to change paths to your own local paths, and also point to the created stego images). Combine the first 50 real images and the last 50 steganographic images to create your test set. We also provide the test sets we used for our ROC AUC plot as `data/test_dense1`, `data/test_dense3`, `data/test_dense6`. You can then run StegExpose with your test image folder and use `/code/plot_stegexpose_rocauc.py` to create your ROC AUC plot.

## Results/Insights
Our re-implementation matches the paper's trends: Dense depth 6 performs best, and our attention encoder beats the paper's results across all four metrics. We noted an important tradeoff: higher RS-BPP/Accuracy comes at the cost of PSNR/SSIM, a tradeoff between being able to detect and decode the message and the 'realness' of the image. Scaling the MSE loss by 100 helped us reach the paper's results, bringing our image realness metric closer to that of the paper's. StegExpose auROC was 0.76 (model trained on DIV2K, evaluated on COCO) (vs. 0.59 reported; model trained and evaluated on COCO), indicating that the model is not robust to distribution shift.

On blurred and grayscale image ablations, we noted that it was (1) easier for the model to hide messages in the blurred images and (2) message location changed as compared to the RGB image: grayscale hid messages across channels and focused on edges, while blurred images relied more on color channels and avoided sharp edges.

## Conclusion
Our re-implementation successfully replicates and improves upon SteganoGAN's results, with our Attention Encoder/Decoder achieving the best metrics overall. The key lessons: loss scaling is critical for image fidelity, attention mechanisms improve steganographic quality via global image understanding, and the decodability vs imperceptibility tradeoff is quite prominent: higher accuracy (better decodability) produces visibly grainier images (message easier to spot). Models also generalized well to OOD inputs (grayscale, blurred, somewhat to COCO), hiding messages differently across image types.

## References
[1] Zhang, Kevin Alex, et al. "SteganoGAN: High Capacity Image Steganography with GANs." arXiv preprint arXiv:1901.03892 (2019).

[2] Boehm, Benedikt. "Stegexpose - A Tool for Detecting LSB Steganography." arXiv preprint arXiv:1410.6656 (2014).

[3] Liu, Ze, et al. “Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.” arXiv preprint arXiv:2103.14030 (2021).

[4] Ignatov, Andrey and Timofte, Radu, et al. “PIRM challenge on perceptual image enhancement on smartphones: report”. In European Conference on Computer Vision (ECCV) Workshops (2019).

[5] Lin, T., et al. “Microsoft COCO: common objects in context.” arXiv preprint arXiv:1405.0312 (2014). 

## Acknowledgements
This project was undertaken for the final course project in CS 4782: Introduction to Deep Learning at Cornell University in Spring 2026. We thank the professors and course staff for their time and advising throughout the semester. We also thank the authors of SteganoGAN for setting up a SoTA result in steganography, and also setting the stage for future deep-learning papers in the field, inspiring our own additional works with novel encoders, decoders, and experiments.
