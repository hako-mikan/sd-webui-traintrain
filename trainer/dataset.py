from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from tqdm import tqdm
import random
import torch.nn.functional as F

test = False

def make_dataloaders(t):
    find_filesets(t)                               #画像、テキスト、キャプションのパスを取得
    make_buckets(t)                            #画像サイズのリストを作成
    load_resize_image_and_text(t)       #画像を読み込み、画像サイズごとに振り分け、リサイズ、テキストの読み込み
                                                        #t.image_bucketsは画像サイズをkeyとしたimage,txt, captionのリスト
    encode_image_text(t)                    #画像とテキストをlatentとembeddingに変換

    dataloaders = []                             #データセットのセットを作成
    for key in t.image_buckets:
        if test: save_images(t, key, t.image_buckets_raw[key])
        dataset = LatentsConds(t, t.image_buckets[key])
        if dataset.__len__() > 0:
            dataloaders.append(DataLoader(dataset, batch_size=t.train_batch_size, shuffle=True))
        
    return dataloaders

class ContinualRandomDataLoader:
    def __init__(self, dataloaders):
        self.original_dataloaders = dataloaders
        self.epoch = 0
        self.data = len(self.original_dataloaders) > 0
        self._reset_iterators()

    def _reset_iterators(self):
        # すべての DataLoader から新しいイテレータを生成
        self.dataloaders = list(self.original_dataloaders)
        self.iterators = [iter(dataloader) for dataloader in self.dataloaders]

    def __iter__(self):
        return self

    def __next__(self):
        if not self.iterators:
            # すべての DataLoader が終了したらリセット
            self._reset_iterators()

        while self.iterators:
            # ランダムに DataLoader を選択
            idx = random.randrange(len(self.iterators))
            try:
                return next(self.iterators[idx])
            except StopIteration:
                # 終了した DataLoader をリストから削除
                self.iterators.pop(idx)
                self.dataloaders.pop(idx)

        # すべての DataLoader が終了した場合
        self.epoch += 1
        raise StopIteration
                                               
class LatentsConds(Dataset):
    def __init__(self, t, latents_conds):
        self.latents_conds = latents_conds
        self.batch_size = t.train_batch_size
        self.isxl = t.isxl

    def __len__(self):
        return len(self.latents_conds)

    def __getitem__(self, i):
        batch = {}
        latent, mask, cond1, cond2 = self.latents_conds[i]
        batch["latent"] = latent.squeeze()
        batch["cond1"] = cond1 if isinstance(cond1, str) else cond1.squeeze() if cond1 is not None else None

        if self.isxl:
            batch["cond2"] = cond2 if isinstance(cond2, str) else cond2.squeeze() if cond2 is not None else None
        if mask is not None:
            batch["mask"] = mask.squeeze() 

        return batch

TARGET_IMAGEFILES = ["jpg", "jpeg", "png", "gif", "tif", "tiff", "bmp", "webp", "pcx", "ico"]

def make_buckets(t):
    increment = t.image_buckets_step # default : 256
    # 最大ピクセル数 resolutionは[x ,y]の配列。 y >= x
    max_pixels = t.image_size[0]*t.image_size[1] 

    # 正方形は手動で追加
    max_buckets = set()
    max_buckets.add((t.image_size[0], t.image_size[0]))

    # 最小値から～
    width = t.image_min_length
    # ～最大値まで
    while width <= max(t.image_size):
        # 最大ピクセル数と最大長を越えない最大の高さ
        height = min(max(t.image_size), (max_pixels // width) - (max_pixels // width) % increment)
        ratio = width/height

        # アスペクト比が極端じゃなかったら追加、高さと幅入れ替えたものも追加。
        if 1 / t.image_max_ratio <= ratio <= t.image_max_ratio:
            max_buckets.add((width, height))
            max_buckets.add((height, width))
        width += increment  # 幅を大きくして次のループへ

    sub_buckets = set()

    # 最小サイズから最大サイズまでの範囲で枠を生成
    for width in range(t.image_min_length, max(t.image_size) + 1, increment):
        for height in range(t.image_min_length, max(t.image_size) + 1, increment):
            if width * height <= max_pixels:
                ratio = width / height
                if 1 / t.image_max_ratio <= ratio <= t.image_max_ratio:
                    if (width, height) not in max_buckets:
                        sub_buckets.add((width, height))
                    if (height, width) not in max_buckets:
                        sub_buckets.add((height, width))

    # アスペクト比に基づいて枠を並べ替え
    max_buckets = list(max_buckets)
    max_ratios = [w / h for w, h in max_buckets]
    max_buckets = np.array(max_buckets)[np.argsort(max_ratios)]
    max_buckets = [tuple(x) for x in max_buckets]
    max_ratios = np.sort(max_ratios)

    sub_buckets = list(sub_buckets)
    sub_ratios = [w / h for w, h in sub_buckets]
    sub_buckets = np.array(sub_buckets)[np.argsort(sub_ratios)]
    sub_buckets = [tuple(x) for x in sub_buckets]
    sub_ratios = np.sort(sub_ratios)

    t.image_max_buckets_sizes = max_buckets
    t.image_max_ratios = max_ratios
    t.image_sub_buckets_sizes = sub_buckets
    t.image_sub_ratios = sub_ratios
    t.image_buckets_raw = {}
    t.image_buckets = {}
    print("max bucket sizes : ", max_buckets)
    #t.db("max bucket sizes : ", max_ratios)
    print("sub bucket sizes : ", sub_buckets)
    #t.db("sub bucket sizes : ", sub_ratios)
    for bucket in max_buckets + sub_buckets:
        t.image_buckets_raw[bucket] = []
        t.image_buckets[bucket] = []

def find_filesets(t):
    """
    Create two lists: 
    1. Absolute paths of image files in the specified folder and subfolders.
    2. Absolute paths of corresponding text files, or 'None' if no corresponding text file exists.

    :param folder_path: Path to the folder to search in.
    :param image_extensions: List of image file extensions to look for.
    :return: Tuple of two lists (image_paths, text_paths)
    """
    pathsets = []
    
    # Walk through the folder and subfolders
    for root, dirs, files in os.walk(t.lora_data_directory):
        for file in files:
            if any(file.endswith(ext) for ext in TARGET_IMAGEFILES):
                image_path = os.path.join(root, file)

                filename = os.path.splitext(os.path.basename(image_path))[0]
                filename = filename.split("_id_")[0]
                filename = filename.replace("_", ",")  

                # Check for corresponding text file
                text_file = os.path.splitext(image_path)[0] + '.txt'
                text_file = text_file if os.path.isfile(text_file) else None

                # Check for corresponding caption file
                caption_file = os.path.splitext(image_path)[0] + '.caption'
                caption_file = caption_file if os.path.isfile(caption_file) else None
                pathsets.append([image_path, text_file, caption_file, filename])

    t.db("Images : ", len(pathsets))
    t.db("Texts : " , sum(1 for _, text, _, _ in pathsets if text is not None))
    t.db("Captions : " , sum(1 for _, _, caption, _ in pathsets if caption is not None))

    t.image_pathsets = pathsets

def load_resize_image_and_text(t):
    for img_path, txt_path, cap_path, filename in t.image_pathsets:
        image = Image.open(img_path)
        usealpha = image.mode == "RGBA"

        #max sizes
        ratio = image.width / image.height
        ar_errors = t.image_max_ratios - ratio
        indice = np.argmin(np.abs(ar_errors))  # 一番近いアスペクト比のインデックス
        max = t.image_max_buckets_sizes[indice]
        ar_error = ar_errors[indice]

        def resize_and_crop(ar_error, image, bucket_width, bucket_height, disable_upscale):
            if (ar_error > 0 and image.width < bucket_width or 
                ar_error <= 0 and image.height < bucket_height) and disable_upscale:
                return None

            if ar_error <= 0:  # 幅＜高さなら高さを合わせる
                temp_width = int(image.width*bucket_height/image.height)
                image = image.resize((temp_width, bucket_height))  # アスペクト比を変えずに高さだけbucketに合わせる
                left = (temp_width - bucket_width) / 2  # 切り取り境界左側
                right = bucket_width + left  # 切り取り境界右側
                image = image.crop((left, 0, right, bucket_height))  # 左右切り取り
            else:  # 幅高さを逆にしたもの
                temp_height = int(image.height*bucket_width/image.width)
                image = image.resize((bucket_width, temp_height))
                upper = (temp_height - bucket_height) / 2
                lower = bucket_height + upper
                image = image.crop((0, upper, bucket_width, lower))

            if usealpha:
                tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
                # アルファチャンネルの抽出
                alpha_channel = tensor[3]
                # 透明な部分を0、透明でない部分を1に設定
                alpha_mask = (alpha_channel > 0.1).float()
                # マスクのサイズを取得
                H, W = alpha_mask.shape
                # 新しいサイズを計算（縦横8分の1）
                new_H, new_W = H // 8, W // 8
                # マスクを縦横8分の1にリサイズ
                mask = F.interpolate(alpha_mask.unsqueeze(0).unsqueeze(0), size=(new_H, new_W), mode='nearest')
                mask = torch.cat([mask]*4, dim=1)
            else:
                # アルファチャンネルがない場合の画像サイズの取得
                tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
                _, H, W = tensor.shape  # アルファチャンネルがない場合のためにRGBチャンネルを無視
                new_H, new_W = H // 8, W // 8  # 新しいサイズを計算（縦横8分の1）
                # すべて1のテンソルを作成
                mask = torch.ones((1, 4, new_H, new_W))

            image = image.convert("RGB")

            return image, mask

        resized, alpha_mask = resize_and_crop(ar_error, image, *max, t.image_disable_upscale)
        if resized is not None:
            t.image_buckets_raw[max].append([resized, alpha_mask, load_text_files(txt_path), load_text_files(cap_path), filename])
            if t.image_mirroring:
                flipped = resized.transpose(Image.FLIP_LEFT_RIGHT)
                flipped_mask = torch.flip(alpha_mask, [1]) if alpha_mask is not None else None
                t.image_buckets_raw[max].append([flipped, flipped_mask, load_text_files(txt_path), load_text_files(cap_path), filename])

        ar_errors = t.image_sub_ratios - ratio

        try:
            for _ in range(t.sub_image_num):
                indice = np.argmin(np.abs(ar_errors))  # 一番近いアスペクト比のインデックス
                sub = t.image_sub_buckets_sizes[indice]
                ar_error = ar_errors[indice]
                resized, alpha_mask  = resize_and_crop(ar_error, image, *sub, t.image_disable_upscale)
                if resized is not None:
                    t.image_buckets_raw[sub].append([resized, alpha_mask, load_text_files(txt_path), load_text_files(cap_path), filename])
                    if t.image_mirroring:
                        flipped = resized.transpose(Image.FLIP_LEFT_RIGHT)
                        flipped_mask = torch.flip(alpha_mask, [1]) if alpha_mask is not None else None
                        t.image_buckets_raw[sub].append([flipped, flipped_mask, load_text_files(txt_path), load_text_files(cap_path), filename])
                
                ar_errors[indice] = ar_errors[indice] + 1
        except:
            print("Failed to make sub-buckets; image bucket step or image minimum length is too big?")

    for key in t.image_buckets_raw:
        print(f"bucket {key} has {len(t.image_buckets_raw[key])} images")
        t.total_images += len(t.image_buckets_raw[key])
    
def load_text_files(file_path):
    if file_path is None:
        return None
    with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()

def encode_image_text(t):
    with torch.no_grad(), t.a.autocast():
        emp1, emp2 = t.text_model.encode_text(t.lora_trigger_word)
        bar = tqdm(total = t.total_images)
        for key in t.image_buckets_raw:
            for image, mask, text, caption, filename in t.image_buckets_raw[key]:
                latent = t.image2latent(t,image)
                if t.image_use_filename_as_tag:
                    prompt = t.lora_trigger_word + "," + filename
                elif text is not None:
                    prompt = t.lora_trigger_word + ", " + text
                elif caption is not None:
                    prompt = t.lora_trigger_word + ", " + caption
                else:
                    prompt = t.lora_trigger_word
                t.tagcount(prompt)
                if "BASE" not in t.network_blocks:
                    emb1, emb2 = (emp1, emp2) if prompt is None else t.text_model.encode_text(prompt)
                else:
                    emb1 = emb2 = prompt
                t.image_buckets[key].append([latent, mask, emb1, emb2])
                bar.update(1)

def save_images(t,key,images):
    if not images: return
    path = os.path.join(t.lora_data_directory,"x".join(map(str, list(key))))
    os.makedirs(path, exist_ok=True)
    for i, image in enumerate(images):
        ipath = os.path.join(path, f"{i}.jpg")
        image[0].save(ipath)
