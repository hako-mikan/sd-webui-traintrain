# TrainTrain
- This is an extension for [AUTOMATIC1111's stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui).
- You can create LoRA, iLECO, and differential LoRA.

[<img src="https://img.shields.io/badge/lang-Egnlish-red.svg?style=plastic" height="25" />](#overview)
[<img src="https://img.shields.io/badge/言語-日本語-green.svg?style=plastic" height="25" />](README_ja.md)
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

| Parameter | Details |
|-----------|---------|
| network type | lierla is a standard LoRA. c3lier (commonly known as LoCON) and loha (commonly known as LyCORIS) increase the learning area. If you choose c3lier or loha, you can adjust the dimensions of the additional area by setting the `conv rank` and `conv alpha` options. |
| network rank | The size of LoRA, also known as dim. It's not good to be too large, so start around 16. |
| network alpha | The reduction width of LoRA. Usually set to the same or a smaller value than rank. For iLECO, a larger value than rank may be better. |
| lora data directory | Specifies the folder where image files for LoRA learning are stored. Subfolders are also included. |
| lora trigger word | When not using caption files, learning is performed associated with the text written here. Details in the learning section. |
| network blocks | Used for layer-specific learning. BASE refers to the TextEncoder. BASE is not used in iLECO, Difference. |
| train iterations | Number of learning iterations. 500 to 1000 is appropriate for iLECO, Difference. |
| image size | The resolution during learning. The order of height and width is only valid for iLECO. |
| train batch size | How many images are learned at once. Set to an efficient level so that VRAM does not overflow the shared memory. |
| train learning rate | The learning rate. 1e-3 to 1e-4 for iLECO, and about 1e-3 for Difference is appropriate. |
| train optimizer | Setting for the optimization function. adamw is recommended. adamw8bit reduces accuracy. Especially in Difference, adamw8bit does not work well. |
| train lr scheduler | Setting to change the learning rate during learning. Just set it to cosine. If you choose adafactor as the optimizer, the learning rate is automatically determined, and this item is deactivated. |
| save lora name | The file name when saving. If not set, it becomes untitled. |
| use gradient checkpointing | Reduces VRAM usage at the expense of slightly slower learning. |

## Optional Parameters
Optional, so they work even if not specified.
| Parameter | Details |
|-----------|---------|
| network conv rank | Rank of the conv layer when using c3lier, loha. If set to 0, the value of network rank is used. |
| network conv alpha | Reduction width of the conv layer when using c3lier, loha. If set to 0, the value of network alpha is used. |
| network element | Specifies the learning target in detail. Does not work with loha.<br>Full: Same as the normal LoRA.<br>CrossAttention: Only activates layers that process generation based on prompts.<br>SelfAttention: Only activates layers that process generation without prompts. |
| train lr step rules | Specifies the steps when the lr scheduler is set to step. |
| train lr scheduler num cycles | Number of repetitions for cosine with restart. |
| train lr scheduler power | Exponent when the lr scheduler is set to linear. |
| train lr warmup steps | Specifies the number of effective steps to gradually increase lr at the beginning of learning. |
| train textencoder learning rate | Learning rate of the Text Encoder. If 0, the value of train learning rate is used. |
| image buckets step | Specifies the detail of classification when classifying images into several aspect ratios. |
| image min length | Specifies the minimum resolution. |
| image max ratio | Specifies the maximum aspect ratio. |
| sub image num | The number of times the image is reduced to different resolutions. |
| image mirroring | Mirrors the image horizontally. |
| save per steps | Saves LoRA at specified steps. |
| save overwrite | Whether to overwrite when saving. |
| save as json | Whether to save the settings during learning execution. The settings are saved by date in the json folder of the extension. |
| model v pred | Whether the SD2.X model uses v-pred. |
| train model precision | Precision of non-learning targets during learning. fp16 is fine. |
| train lora precision | Precision of the learning target during learning. fp32 is fine. |
| save precision | Precision when saving. fp16 is fine. |
| train seed | The seed used during learning. |
| diff save 1st pass | Whether to save the copier LoRA. |
| diff 1st pass only | Learn only the copier LoRA. |
| diff load 1st pass | Load the copier LoRA from a file. |
| train snr gamma | Whether to add timestep correction

. Set a value between 0 and 20. The recommended value is 5. |
| logging verbose | Outputs logs to the command prompt. |
| logging_save_csv | Records step, loss, learning rate in csv format. |

## Presets, Saving and Loading of Settings
You can call up the settings with a button. The settings are handled in a json file. Presets are stored in the preset folder.

## Queue
You can reserve learning. Pressing the `Add to Queue` button reserves learning with the current settings. If you press this button during learning, the next learning will start automatically after the learning. If pressed before learning, after learning ends with the settings when `Start Training` was pressed, the learning in the Queue list is processed in order. You cannot add settings with the same `save lora name`.

## Plot
When the logging_save_csv option is enabled, you can graph the progress of learning. If you don't enter anything in `Name of logfile`, the results of learning in progress or the most recent learning are displayed. If you enter a csv file name, those results are displayed. Only the file name is needed, not the full path. The file must be in the logs folder.

## Acknowledgments
This code is based on [Plat](https://github.com/p1atdev)'s [LECO](https://github.com/p1atdev/LECO), [laksjdjf](https://github.com/laksjdjf)'s [learning code](https://github.com/laksjdjf/sd-trainer), [kohya](https://github.com/kohya-ss)'s [learning code](https://github.com/kohya-ss/sd-scripts), and [KohakuBlueleaf](https://github.com/KohakuBlueleaf)'s [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS).

## Reference
- https://github.com/rohitgandikota/erasing

- https://github.com/cloneofsimo/lora

- https://github.com/laksjdjf/sd-trainer

- https://github.com/kohya-ss/sd-scripts

- https://github.com/KohakuBlueleaf/LyCORIS

- https://github.com/ntc-ai/conceptmod
