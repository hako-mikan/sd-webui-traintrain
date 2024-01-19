Here is the translation of the provided text:

# TrainTrain
- This is an extension for [AUTOMATIC1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).
- You can create LoRA, iLECO, and differential LoRA.

[<img src="https://img.shields.io/badge/lang-Egnlish-red.svg?style=plastic" height="25" />](README.md)
[<img src="https://img.shields.io/badge/言語-日本語-green.svg?style=plastic" height="25" />](#overview)
[<img src="https://img.shields.io/badge/Support-%E2%99%A5-magenta.svg?logo=github&style=plastic" height="25" />](https://github.com/sponsors/hako-mikan)

# Overview
This is a tool for training LoRA for Stable Diffusion. It operates as an extension of the Stable Diffusion Web-UI and does not require setting up a training environment. It accelerates the training of regular LoRA, iLECO (instant-LECO), which speeds up the learning of LECO (removing or emphasizing a model's concept), and differential learning that creates slider LoRA from two differential images.

# Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
    - [LoRA](#lora)
    - [iLECO](#ileco)
    - [Difference](#difference)
- [Settings](#settings)
    - [Mandatory Parameters](#mandatory-parameters)
    - [Optional Parameters](#optional-parameters)
- [Queue (Reserving Training)](#queue)
- [Plot](#plot)
- [Acknowledgments & References](#acknowledgments)

## Requirements
   Operates with Web-UI 1.7.

## Installation
   Enter `https://github.com/hako-mikan/sd-webui-traintrain` in the Web-UI's Install From URL and press the Install button, then restart. The first startup may take a little time (a few seconds to tens of seconds).

## Usage
   Enter the mandatory parameters for each mode and press the Start Training button to begin training. The created LoRA will be saved in the LoRA folder.
## LoRA
Learn LoRA from images.
### Input Images
   Supports `jpg`, `jpeg`, `png`, `gif`, `tif`, `tiff`, `bmp`, `webp`, `pcx`, `ico` formats. The size does not need to be the one specified by `image size`, but it will be cropped during training, so it's better to format the images to some extent to avoid inconsistencies with the captions. Images are classified by aspect ratio. For example, if you set the `image size` to 768x512, several resolution sets (buckets) will be created with a maximum pixel size of 768x512. By default, it classifies into three types: 768x512, 512x512, and 512x768, and images are sorted into the closest classification by aspect ratio. This is because the training only accepts images of the same size. During this process, images are resized and cropped. The cropping is centered on the image's center. To refine the classification, decrease the value of `image buckets step`.

### Image Resizing & Mirroring
   Training the same image repeatedly can lead to overfitting, where the image itself appears. If there are few training images, we deal with overfitting by resizing and flipping images to increase the number of training images. If you set `image size` to `768,512` and `image buckets step` to `128`, the frames `(384, 768), (512, 768), (512, 640), (512, 512), (640, 512), (768, 512), (768, 384)` are created. Additionally, setting `image min length` to `256` creates frames for resizing such as `(256, 512), (384, 640), (256, 384), (384, 512), (384, 384), (256, 256), (512, 384), (384, 256), (640, 384), (512, 256)`. Images are first sorted into normal frames, but if `sub image num` is set, they are also resized and stored in resizing frames with a similar aspect ratio. For instance, if an image is stored in a `(512, 640)` frame and `sub image num` is set to 3, it is also resized and stored in `(384, 640)`, `(256, 384)`, and `(384, 512)`. If `image mirroring` is enabled, mirrored images are also stored,

 resulting in 8 training images from one image.

### Captions, Trigger Words
   If there are `txt` or `caption` files with the same filename as the image, the text in these files is used for training. If both exist, the `txt` file takes precedence. If `trigger word` is set, it is inserted before all captions, including when there is no caption file.

### Approach to Captions
   Let's say you're training a character named A. A has twin tails, wears a blue shirt, and a red skirt. If there's a picture of A against a white background, the caption should include A's name, the direction they're facing, and that the background is white. Elements unique to A, like twin tails, blue shirt, and red skirt, shouldn't be included in the caption as they are specific to A and you want to train for them. However, direction, background, and composition, which you don't want to learn, should be included.

## iLECO
   iLECO (instant-LECO) is a faster version of LECO training, transforming the concept specified in Original Prompt closer to the concept in Target Prompt. If nothing is entered in Target Prompt, it becomes training to remove that concept.
   For example, let's erase the Mona Lisa, which appears robustly in any model. Enter "Mona Lisa" in Original Prompt and leave Target Prompt blank. It converges with about 500 `train iterations`. The value of `alpha` is usually set smaller than rank, but in the case of iLECO, a larger value than rank may be better.
   We succeeded in erasing the Mona Lisa. Next, enter "Vincent van Gogh Sunflowers" in Target Prompt. Now, the Mona Lisa turns into sunflowers in the LoRA.
   Try entering "red" in Original Prompt and "blue" in Target Prompt. You get a LoRA that turns red into blue.

## Difference
   Creates LoRA from two differential images. This is known as the copy machine learning method. First, create a copy machine LoRA (which only produces the same image), then apply LoRA and train for the difference to create a differential LoRA. Set images in Original and Target. The image size should be the same.
   First, training for the copy machine begins, followed by training for the difference. For example, let's make a LoRA for closing eyes using the following two images.
   Use Difference_Use2ndPassSettings. Set `train batch size` to 1-3. A larger value does not make much difference. We succeeded. Other than closing the eyes, there is almost no impact on the painting style or composition. This is because the rank(dim) is set to 4, which is small in the 2ndPass. If you set this to the same 16 as the copy machine, it will affect the painting style and composition.

> [!TIP]
> If you don't have enough VRAM, enable `gradient checkpointing`. It will slightly extend the computation time but reduce VRAM usage. In some cases, activating `gradient checkpointing` and increasing the batch size can shorten the computation time. In copy machine learning, increasing the batch size beyond 3 makes little difference, so it's better to keep it at 3 or less. The batch size is the number of images learned at once, but doubling the batch size doesn't mean you can halve the `iterations`. In one learning step, the weights are updated once, but doubling the batch size does not double the number of updates, nor does it double the efficiency of learning.

## Settings
## Mandatory Parameters

|Parameter| Details  | 
|----|----|
|network type | lierla is regular LoRA. c3lier (commonly known as LoCON) and loha (commonly known as LyCORIS) increase the area of learning. If you choose c3lier or loha, setting the optional `conv rank` and `conv alpha` adjusts the dim of the additional area| 
|network rank | The size of LoRA. Also known as dim. Don't set it too high; start with about 16| 
|network alpha   | The reduction width of LoRA. Usually, the same or a smaller value than rank is set. For iLECO, a larger value than rank may be better| 
|lora data directory| Specify the folder where the images for LoRA training are saved. Includes subfolders| 
|lora trigger word| If you don't use a caption file, training is performed associated with the text written here. See Usage for details| 
|network blocks| Used for layer-specific training. BASE refers to the TextEncoder. BASE is not used in iLECO, Difference| 
|train iterations  | Number of training cycles. 500 to 1000 is suitable for iLECO, Difference| 
