# Paper: SteganoGAN: High Capacity Image Steganography with GANs
Shriya Sudhakar (ss3576), Kiran Mitra (km936), Nidhi Soma (ns848), Niti Goyal (ng459)

## Introduction
This repository is a re-implementation of SteganoGAN undertaken as part of CS 4782: Introduction to Deep Learning at Cornell University (Spring 2026).

Image steganography is at the intersection of security and computer vision, where the main goal is to be able to hide secret messages inside images imperceptibly. Encoding binary data into natural images is quite difficult due to the need of balancing message capacity, visual authenticity, and evasion of steganographic detection tools.

This repository reimplements [SteganoGAN: High Capacity Image Steganography with GANs](https://arxiv.org/pdf/1901.03892) by Zhang et al., 2019, which introduces a GAN-based method for high-capacity image steganography. The encoder hides a message in the cover image, while a decoder recovers it. The critic encourages steganographic images to look indistinguishable from the original cover image. The authors test three types of encoders (Basic, Residual, Dense), and achieve a record payload of 4.4 bits per pixel for natural images from multiple datasets that evade detection by traditional and deep-learning-based steganalysis tools.

In addition to reproducing their main result on the DIV2K dataset, we introduce a novel attention-based encoder and decoder that surpasses the paper's initial results, and also perform loss ablations and model robustness experiments.

## Chosen Result
We chose to replicate the paper's Div2K resulting metrics on depths=1,3,6 for all Basic, Residual, and Dense encoders.
<img width="1378" height="277" alt="image" src="https://github.com/user-attachments/assets/d5aafd99-0c9b-4385-ad47-46f776041a88" />
Table: Comparison of 4 key metrics against different encoders, performed on the DIV2K dataset [Zhang et al. 2019]

We also used StegExpose, a classic open-source steganalysis tool, to evaluate the detection performance of all model types implemented, replicating Figure 5. Note that we only used 50 samples to replicate this result and plotted depths=1,3,6 while the paper used 1000 samples and plotted depth 6.
<img width="708" height="796" alt="image" src="https://github.com/user-attachments/assets/67dd2b7b-5fc8-420c-87fd-d6a10ed8d7bf" />



## GitHub Contents
`requirements.txt` - all Python dependencies; install with `pip install -r requirements.txt`

Important folders:
/code - contains main components of model like encoders, decoders, and critic
- `train.ipynb` - Python notebook in which users can train configurations of models to replicate Table 1 from the original paper. It contains all necessary code, including encoders, decoders, critics, train loop, etc.
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

/data - mainly contains DIV2K validation, and test data used for evaluation
- since the full datasets are too large to upload to github, we've included instructions on how to download the data in the README.md in the /data folder
- `/DIV2K_valid_HR` - contains our validation dataset of 100 images from DIV2K used to evaluate our main table of metrics
- `/coco_val_images` - contains the 100 images from COCO used for StegExpose evaluation. Also contains a notebook `coco_100_images.ipynb` to download these images.
- `profs` - contains images of our professors! (used in our poster)
- `profs_outputs/Dense_D=1` - contains steganographic images and residuals for our professors with the Dense model at Depth=1 (also used in our poster)
- `test_dense1`, `test_dense3`, `test_dense6` - contains the processed COCO cover and steganographic images used for our StegExpose evaluation

/checkpoints - contains pre-trained model weights for all encoder types (Basic, Residual, Dense, Attention) at depths 1, 3, 6, including ablation variants (`/with_scaling`, `/no_critic`). Scripts in `/code` load from this folder by default.

/results
- contains our main replication of table 1 (`table_metrics.png`) along with various training experiments (scaling MSE and removing critic), as well as results from the StegExpose tool (`stegexpose.png`).

/poster
- `SteganoGAN CS 4782.pdf` contains our poster that was presented at the final poster session

/report
- `steganogan_2page_report.pdf` contains our final report and analysis

## Re-implementation Details
We implemented the SteganoGAN core encoder, decoder, critic framework. The encoder hides a binary message inside a provided cover image, the decoder recovers said message, and a critic (discriminator in GANs), pushes images to look like real images. 

We trained three encoder variants: Basic (sequential convolutions), Residual (adds cover image as skip connection), and Dense (concatenates intermediate feature maps). We implemented the DenseDecoder (only one decoder in paper, uses convolutions), Critic (also uses convolutions). We trained on the DIV2K dataset (800 train, 100 validation images) for 32 epochs with Adam at lr=1e-4. 

We evaluated 4 different accuracy metrics (Decoder Accuracy, RS-BPP, PSNR, SSIM) across 3 data depths D={1, 3, 6}. Initially during training, we noticed that while our Acc and RS-BPP metrics were high, PSNR and SSIM (image realness) were lower than expected (as compared to the paper). We performed ablations modifying the loss (100x MSE scaling, critic removal) and noticed much improved performance, meeting the paper's results on all 4 metrics. Beyond reproducing these main results from the paper, we also introduce a novel Attention encoder and decoder, which extends the existing Dense architecture with Window Multi-Head Self Attention (8x8 windows, 3 heads), and also performs the best compared to the 3 architectures proposed initially. 

We fnally evaluated the robustness of our models to image distribution shifts (StegExpose on COCO images), and tested adaptability on grayscale and blurred images.

## Reproduction Steps

**Dependencies:** Install required libraries with:
```bash
pip install -r requirements.txt
```

**Computational resources:** A GPU is required. All experiments were run on a T4 GPU in Google Colab. Training for 32 epochs takes approximately 80 minutes per model config.

**Training:** To train your own SteganoGAN, run our training script in /code/train.ipynb which will download the DIV2K data locally, preprocess images, and run the training loop. To train with our novel attention encoder/decoder, use /code/train_attentionencoder.ipynb. It also has all our model architecture and supports wandb logging for evaluation metrics. Our notebook does multiple runs with different Encoders and different depths so you can reproduce our table of results or make your own by changing the list of configs in the config loop in the last cell. You will need a wandb login and API key before beginning and to change SAVE_DIR to the correct GDrive path in which you would like to save the model weights.

**StegExpose evaluation:** To reproduce the StegExpose plot, you will have to download the tool code: https://github.com/b3dk7/stegexpose. You can use the 100 validation images in /data/coco_val_images. Run /code/crop_real_images.py to process the real images, and /code/generate_eval_images.py to generate steganographic versions (you may have to modify the paths for your local setup). Then combine the first 50 real images and the last 50 steganographic images to create your test set. We also provide the test sets we used for our ROC AUC plot as test_dense1, test_dense3, test_dense6. You can then run StegExpose with your test image folder and use /code/plot_stegexpose_rocauc.py to create your ROC AUC plot.

## Results/Insights

Our re-implementation reproduces the main trends from the original SteganoGAN paper. Like the paper, we found that deeper Dense models perform well, especially at depth 6. However, our best overall results came from our novel Attention-based architecture.

Running this repo will train models that hide binary messages inside images and decode them back, while reporting decoding accuracy, RS-BPP, PSNR, and SSIM. In general, users should expect a tradeoff: higher decoding accuracy and RS-BPP often come with lower PSNR and SSIM, meaning better message recovery can make images look more visibly altered.

We found that scaling the image MSE loss by 100, as done in the original codebase, greatly improved image realness and brought our results closer to the paper’s reported metrics. We also tested removing the critic and found similar results at depths 1 and 6, suggesting the adversarial loss may not be essential when the MSE term dominates.

For steganalysis, the paper reported a StegExpose auROC of 0.59 for Dense depth 6. Our model achieved 0.76 on COCO images after training on DIV2K, meaning it was easier to detect under distribution shift. This suggests broader training data may be needed for real-world robustness.

For the blurred and grayscale image ablations, we visualized residuals to see where the model stored the hidden message. Grayscale images spread the message more evenly across channels and focused on edges, while blurred images relied more on color channels and avoided sharp edge artifacts. Overall, the model generalized well. It was also easier to hide messages in blurry images while maintaining the realness.

Overall, this repo provides a working SteganoGAN re-implementation with training, evaluation, StegExpose testing, and additional architecture/ablation experiments.


## Conclusion
Our implementation of SteganoGAN successfully replicates and even improves upon the original paper's results with our new Attention Encoder and Decoder. We achieved the highest metrics across the board on Acc, RS-BPP, PSNR, and SSIM compared to the paper's original implementation. We learned that appropriate scaling of the losses is necessary to achieve high image realness, while maintaining a level of decodability of messages. As a result, one of the most important things we discovered during this project was a tradeoff at the pareto-optimal frontier between accuracy metrics with respect to how well a message can be decoded (Accuracy and RS-BPP) and the fidelity metrics (PSNR and SSIM). These results can also be interpreted visually (in addition to metrics): images with higher decode accuracy tend to look tampered (grainy), while images with higher fidelity look more real, but make it harder to locate and decode messages. We also noticed that at higher depths (larger message), the model struggles with encoding more information imperceptibly, which makes sense since the model's task is to now hide more bits per pixel, and the more modification to an image, the more likely it looks tampered with. Finally, through our evaluation ablations, we noted that our models are able to adapt to out of distribution (OOD) images, through evaluation on COCO, grayscale, and blurred images, and hid the message differently in each type of image.

## References
[1] Zhang, Kevin Alex, et al. "SteganoGAN: High Capacity Image Steganography with GANs." arXiv preprint arXiv:1901.03892 (2019).

[2] Boehm, Benedikt. "Stegexpose - A Tool for Detecting LSB Steganography." arXiv preprint arXiv:1410.6656 (2014).

[3] Liu, Ze, et al. “Swin Transformer: Hierarchical Vision Transformer using Shifted Windows.” arXiv preprint arXiv:2103.14030 (2021).

[4] Ignatov, Andrey and Timofte, Radu, et al. “PIRM challenge on perceptual image enhancement on smartphones: report”. In European Conference on Computer Vision (ECCV) Workshops (2019).

[5] Lin, T., et al. “Microsoft COCO: common objects in context.” arXiv preprint arXiv:1405.0312 (2014). 

## Acknowledgements
This project was undertaken for the final course project in CS 4782: Introduction to Deep Learning at Cornell University in Spring 2026. We thank the professors and course staff for their time and advising throughout the semester. We also thank the authors of SteganoGAN for setting up a SoTA result in steganography, and also setting the stage for future deep-learning papers in the field, inspiring our own additional works with novel encoders, decoders, and experiments.
