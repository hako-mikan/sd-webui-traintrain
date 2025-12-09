import torch, json, os, ast, warnings, gc, sys, subprocess, safetensors.torch
import torch.nn as nn
import gradio as gr
from safetensors.torch import load_file
from pipelines.transformer_z_image import ZImageTransformer2DModel
from pipelines.pipeline_z_image import ZImagePipeline
from datetime import datetime
from typing import Literal
from diffusers.optimization import get_scheduler
from transformers.optimization import AdafactorSchedule
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, CosineAnnealingWarmRestarts, StepLR, MultiStepLR, ReduceLROnPlateau, CyclicLR, OneCycleLR
from pprint import pprint
from accelerate import Accelerator
from pathlib import Path
from torchao.quantization import int8_weight_only, quantize_
from diffusers.models import AutoencoderKL
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
    StableDiffusion3Pipeline, 
    FluxPipeline,
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler
)

warnings.filterwarnings("ignore", category=FutureWarning)

CUDA = torch.device("cuda:0")
CPU = torch.device("cpu")

try:
    from modules.scripts import basedir
    from modules import shared, paths
    standalone = False
    path_root = basedir()
    path_trainer = os.path.join(path_root, "trainer")
    lora_dir = shared.cmd_opts.lora_dir
except:
    path_root = Path.cwd()
    lora_dir = os.path.join(path_root,"output")
    path_root = os.path.join(path_root, "traintrain")
    path_trainer = os.path.join(path_root, "trainer")
    standalone = True
    from modules.launch_utils import args

all_configs = []

PASS2 = "2nd pass"
POs = ["came", "tiger", "adammini"]
EXTENTIONS = ("ckpt", "pt", "pth", "bin", "safetensors", "sft", "gguf")

jsonspath = os.path.join(path_root,"jsons")
logspath = os.path.join(path_root,"logs")
presetspath = os.path.join(path_root,"presets")

SEP = "--------------------------"
OPTIMIZERS = ["AdamW", "AdamW8bit", "AdaFactor", "Lion", "Prodigy", SEP,
              "DadaptAdam","DadaptLion", "DAdaptAdaGrad", "DAdaptAdan", "DAdaptSGD",SEP,
               "Adam8bit", "SGDNesterov8bit", "Lion8bit", "PagedAdamW8bit", "PagedLion8bit",  SEP, 
               "RAdamScheduleFree", "AdamWScheduleFree", "SGDScheduleFree", SEP, 
               "CAME", "Tiger", "AdamMini",
               "PagedAdamW", "PagedAdamW32bit", "SGDNesterov", "Adam",]

class Trainer():
    def __init__(self, jsononly, model, vae, te, mode, values):
        if type(jsononly) is list:
            paths = jsononly
            jsononly = False
        else:
            if standalone:
                paths = [args.models_dir, args.ckpt_dir, args.vae_dir, args.lora_dir]
            else:
                paths = None
                
        self.values = values
        self.mode = mode
        self.use_8bit = False
        self.read_model_dtype = False
        self.count_dict = {}
        self.metadata = {}

        self.save_dir = lora_dir
        if not os.path.exists(lora_dir):
            os.makedirs(lora_dir)
        self.setpass(0)

        self.image_size = [int(x) for x in self.image_size.split(",")]
        if len(self.image_size) == 1:
            self.image_size = self.image_size * 2
        self.image_size.sort()
        
        self.gradient_accumulation_steps = 1
        self.train_repeat = 1
        self.total_images = 0
        
        if self.diff_1st_pass_only:
            self.save_1st_pass = True

        self.checkfile()

        clen = len(all_configs) * (len(values) // len(all_configs))

        self.prompts = values[clen:clen + 3]
        self.images =  values[clen + 3:]
        
        self.add_dcit = {"mode": mode, "model": model, "vae": vae,"te": te, "original prompt": self.prompts[0],"target prompt": self.prompts[1]}

        self.export_json(jsononly)
        if paths is not None:
            self.setpaths(paths)

    def setpass(self, pas, set = True):
        values_0 = self.values[:len(all_configs)]
        values_1 = self.values[len(all_configs):len(all_configs) * 2]
        if pas == 1:
            if values_1[-1]:
                if set: print("Use 2nd pass settings")
            else:
                return
        jdict = {}
        for i, (sets, value) in enumerate(zip(all_configs, values_1 if pas > 0 else values_0)):
            jdict[sets[0]] = value
    
            if pas > 0:
                if not sets[5][3]:
                    value = values_0[i]

            if not isinstance(value, sets[4]):
                try:
                    value = sets[4](value)
                except:
                    if not sets[0] == "train_textencoder_learning_rate":
                        print(f"ERROR, input value for {sets[0]} : {sets[4]} is invalid, use default value {sets[3]}")
                    value = sets[3]
            if "precision" in sets[0]:
                if sets[0] == "train_model_precision" and value == "int8":
                    self.use_8bit = True
                    print("Use 8bit Model Precision")
                if sets[0] == "train_model_precision" and value == "model":
                    self.read_model_dtype = True
                value = parse_precision(value)

            if "diff_load_1st_pass" == sets[0]:
                found = False
                value = value if ".safetensors" in value else value + ".safetensors"
                for root, dirs, files in os.walk(self.save_dir):
                    if value in files:
                        value = os.path.join(root, value)
                        found = True
                if not found:
                    value = ""
            
            if "train_optimizer" == sets[0]:
                value = value.lower()
            
            if "train_optimizer_settings" == sets[0] or "train_lr_scheduler_settings" == sets[0]:
                dvalue = {}
                if value is not None and len(value.strip()) > 0:
                    # 改行や空白を取り除きつつ処理
                    value = value.replace(" ", "").replace(";","\n")
                    args = value.split("\n")
                    for arg in args:
                        if "=" in arg:  # "=" が存在しない場合は無視
                            key, val = arg.split("=", 1)
                            val = ast.literal_eval(val)  # リテラル評価で型を適切に変換
                            dvalue[key] = val
                value = dvalue
            if set: 
                setattr(self, sets[0].split("(")[0], value)
        
        self.mode_fixer()
        return jdict

    savedata = ["model", "vae", ]
    
    def export_json(self, jsononly):
        current_time = datetime.now()
        outdict = self.setpass(0, set = False)
        if self.mode == "Difference":
            outdict[PASS2] = self.setpass(1, set = False)
        outdict.update(self.add_dcit)
        today = current_time.strftime("%Y%m%d")
        time = current_time.strftime("%Y%m%d_%H%M%S")
        add = "" if jsononly else f"-{time}"
        jsonpath = os.path.join(presetspath, self.save_lora_name + add + ".json")  if jsononly else os.path.join(jsonspath, today, self.save_lora_name + add + ".json")
        self.csvpath = os.path.join(logspath ,today, self.save_lora_name + add + ".csv")
        
        if self.save_as_json:
            directory = os.path.dirname(jsonpath)
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(jsonpath, 'w') as file:
                json.dump(outdict, file, indent=4)
        
        if jsononly:
            with open(jsonpath, 'w') as file:
                json.dump(outdict, file, indent=4)  

    def db(self, *obj, pp = False):
        if self.logging_verbose:
            if pp:
                pprint(*obj)
            else:
                print(*obj)

    def checkfile(self):
        if self.save_lora_name == "":
            self.save_lora_name = "untitled"

        filename = os.path.join(self.save_dir, f"{self.save_lora_name}.safetensors")

        self.isfile = os.path.isfile(filename) and not self.save_overwrite
    
    def tagcount(self, prompt):
        tags = [p.strip() for p in prompt.split(",")]

        for tag in tags:
            if tag in self.count_dict:
                self.count_dict[tag] += 1
            else:
                self.count_dict[tag] = 1
    
    #["LoRA", "iLECO", "Difference","ADDifT", "Multi-ADDifT"]
    def mode_fixer(self):
        if self.mode == "LoRA":
            pass

        if self.mode == "iLECO":
            self.network_resume  = ""

        if self.mode == "ADDifT":
            if self.diff_load_1st_pass:
                self.network_resume = self.diff_load_1st_pass
            if self.lora_trigger_word == "" and "BASE" in self.network_blocks:
                self.network_blocks.remove("BASE")

        if self.mode == "Multi-ADDifT":
            if self.diff_load_1st_pass:
                self.network_resume = self.diff_load_1st_pass

    def sd_typer(self, ver = None):
        if ver is None:
            model = shared.sd_model
            print(type(model).__name__ )
            self.is_sd1 = type(model).__name__ == "StableDiffusion" or getattr(model,'is_sd1', False)
            self.is_sd2 = type(model).__name__ == "StableDiffusion2" or getattr(model,'is_sd2', False)
            self.is_sdxl = type(model).__name__ == "StableDiffusionXL" or getattr(model,'is_sdxl', False)
            self.is_sd3 = type(model).__name__ == "StableDiffusion3" or getattr(model,'is_sd2', False)
            self.is_flux = type(model).__name__ == "Flux" and getattr(model,'is_flux', False)
            self.is_zimage = type(model).__name__ == "ZImage" and getattr(model,'is_flux', False)
        else:
            self.is_sd1 = ver == 0
            self.is_sd2 = ver == 1
            self.is_sdxl = ver == 2
            self.is_sd3 = ver == 3
            self.is_flux = ver == 4
            self.is_zimage = ver == 5

        #sdver: 0:sd1 ,1:sd2, 2:sdxl, 3:sd3, 4:flux
        if self.is_sdxl:
            self.model_version = "sdxl_base_v1-0"
            self.vae_scale_factor = 0.13025
            self.vae_shift_factor = 0
            self.sdver = 2
        elif self.is_sd2:
            self.model_version = "sd_v2"
            self.vae_scale_factor = 0.18215
            self.vae_shift_factor = 0
            self.sdver = 1
        elif self.is_sd3:
            self.model_version = "sd_v3"
            self.vae_scale_factor = 1.5305
            self.vae_shift_factor = 0.0609
            self.sdver = 3
        elif self.is_flux:
            self.model_version = "flux"
            self.vae_scale_factor = 0.3611
            self.vae_shift_factor = 0.1159
            self.sdver = 4
        elif self.is_zimage:
            self.model_version = "zimage"
            self.vae_scale_factor = 0.3611
            self.vae_shift_factor = 0.1159
            self.sdver = 5
        else:
            self.model_version = "sd_v1"
            self.vae_scale_factor = 0.18215
            self.vae_shift_factor = 0
            self.sdver = 0
        
        self.is_dit = self.is_sd3 or self.is_flux or self.is_zimage
        self.is_te2 = self.is_sdxl or self.is_sd3

        print("Base Model : ", self.model_version)
    
    def setpaths(self, paths):
        if paths[0] is not None:#root
            self.save_dir = os.path.join(paths[0],"lora")
            self.models_dir = os.path.join(paths[0],"StableDiffusion")
            self.vae_dir = os.path.join(paths[0],"VAE")
        if paths[1] is not None:#model
            self.models_dir = paths[1]
        if paths[2] is not None:#vae
            self.vae_dir = paths[2]
        if paths[3] is not None:#lora
            self.save_dir = paths[3]

def import_json(name, preset = False, cli = False):
    def find_files(file_name):
        for root, dirs, files in os.walk(jsonspath):
            if file_name in files:
                return os.path.join(root, file_name)
        return None
    if preset:
        filepath = os.path.join(presetspath, name + ".json")
    elif cli:
        filepath = name
    else:
        filepath = find_files(name if ".json" in name else name + ".json")

    output = []

    if filepath is None:
        return [gr.update()] * len(all_configs)
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    def setconfigs(data, output):
        for key, gtype ,_ ,default , dtype, _ in all_configs:
            if key in data:
                if key == "train_optimizer":
                    for optim in OPTIMIZERS:
                        if data[key].lower() == optim.lower():
                            data[key] = optim 
                if key == PASS2: continue
                if gtype == "DD" or "learning rate" in key:
                    dtype = str
                try:
                    output.append(dtype(data[key]))
                except:
                    output.append(default)
            else:
                output.append(default)
    setconfigs(data, output)

    if PASS2 in data and data[PASS2]:
        setconfigs(data[PASS2], output)
    else:
        output = output * 2

    output.append(data["original prompt"] if "original prompt" in data else "")
    output.append(data["target prompt"] if "target prompt" in data else "")
    output.append("")
    if cli:
        output.append(data["original image"] if "original image" in data else "")
        output.append(data["target image"] if "target image" in data else "")
    
    head = []
    head.append(data["mode"] if "mode" in data else "LoRA")
    head.append(data["model"] if "model" in data else None)
    head.append(data["vae"] if "vae" in data else None)

    return head + output

AVAILABLE_SCHEDULERS = Literal["ddim", "ddpm", "lms", "euler_a"]

def get_optimizer(name: str, trainable_params, lr, optimizer_kwargs, network):
    name = name.lower()
    if name.startswith("dadapt"):
        import dadaptation
        if name == "dadaptadam":
            optim = dadaptation.DAdaptAdam
        elif name == "dadaptlion":
            optim = dadaptation.DAdaptLion               
        elif name == "DAdaptAdaGrad".lower():
            optim = dadaptation.DAdaptAdaGrad
        elif name == "DAdaptAdan".lower():
            optim = dadaptation.DAdaptAdan
        elif name == "DAdaptSGD".lower():
            optim = dadaptation.DAdaptSGD

    elif name.endswith("8bit"): 
        import bitsandbytes as bnb
        try:
            if name == "adam8bit":
                optim = bnb.optim.Adam8bit
            elif name == "adamw8bit":  
                optim = bnb.optim.AdamW8bit
            elif name == "SGDNesterov8bit".lower():
                optim = bnb.optim.SGD8bit
                if "momentum" not in optimizer_kwargs:
                    optimizer_kwargs["momentum"] = 0.9
                optimizer_kwargs["nesterov"] = True
            elif name == "Lion8bit".lower():
                optim = bnb.optim.Lion8bit
            elif name == "PagedAdamW8bit".lower():
                optim = bnb.optim.PagedAdamW8bit
            elif name == "PagedLion8bit".lower():
                optim  = bnb.optim.PagedLion8bit

        except AttributeError:
            raise AttributeError(
                f"No {name}. The version of bitsandbytes installed seems to be old. Please install newest. / {name}が見つかりません。インストールされているbitsandbytesのバージョンが古いようです。最新版をインストールしてください。"
            )

    elif name.lower() == "adafactor":
        import transformers
        optim = transformers.optimization.Adafactor

    elif name == "PagedAdamW".lower():
        import bitsandbytes as bnb
        optim = bnb.optim.PagedAdamW
    elif name == "PagedAdamW32bit".lower():
        import bitsandbytes as bnb
        optim = bnb.optim.PagedAdamW32bit

    elif name == "SGDNesterov".lower():
        if "momentum" not in optimizer_kwargs:
            optimizer_kwargs["momentum"] = 0.9
        optimizer_kwargs["nesterov"] = True
        optim = torch.optim.SGD

    elif name.endswith("schedulefree".lower()):
        import schedulefree as sf
        if name == "RAdamScheduleFree".lower():
            optim = sf.RAdamScheduleFree
        elif name == "AdamWScheduleFree".lower():
            optim = sf.AdamWScheduleFree
        elif name == "SGDScheduleFree".lower():
            optim = sf.SGDScheduleFree

    elif name in POs:
        import pytorch_optimizer as po    
        if name == "CAME".lower():
            optim = po.CAME        
        elif name == "Tiger".lower():
            optim = po.Tiger        
        elif name == "AdamMini".lower():
            optim = po.AdamMini   
        
    else:
        if name == "adam":
            optim = torch.optim.Adam
        elif name == "adamw":
            optim = torch.optim.AdamW  
        elif name == "lion":
            from lion_pytorch import Lion
            optim = Lion
        elif name == "prodigy":
            import prodigyopt
            optim = prodigyopt.Prodigy

    
    if name.startswith("DAdapt".lower()) or name == "Prodigy".lower():
    # check lr and lr_count, and logger.info warning
        actual_lr = lr
        lr_count = 1
        if type(trainable_params) == list and type(trainable_params[0]) == dict:
            lrs = set()
            actual_lr = trainable_params[0].get("lr", actual_lr)
            for group in trainable_params:
                lrs.add(group.get("lr", actual_lr))
            lr_count = len(lrs)

        if actual_lr <= 0.1:
            print(
                f"learning rate is too low. If using D-Adaptation or Prodigy, set learning rate around 1.0 / 学習率が低すぎるようです。D-AdaptationまたはProdigyの使用時は1.0前後の値を指定してください: lr={actual_lr}"
            )
            print("recommend option: lr=1.0 / 推奨は1.0です")
        if lr_count > 1:
            print(
                f"when multiple learning rates are specified with dadaptation (e.g. for Text Encoder and U-Net), only the first one will take effect / D-AdaptationまたはProdigyで複数の学習率を指定した場合（Text EncoderとU-Netなど）、最初の学習率のみが有効になります: lr={actual_lr}"
            )

    elif name == "Adafactor".lower():
        # 引数を確認して適宜補正する
        if "relative_step" not in optimizer_kwargs:
            optimizer_kwargs["relative_step"] = True  # default
        if not optimizer_kwargs["relative_step"] and optimizer_kwargs.get("warmup_init", False):
            print(
                f"set relative_step to True because warmup_init is True / warmup_initがTrueのためrelative_stepをTrueにします"
            )
            optimizer_kwargs["relative_step"] = True
 
        if optimizer_kwargs["relative_step"]:
            print(f"relative_step is true / relative_stepがtrueです")
            if lr != 0.0:
                print(f"learning rate is used as initial_lr / 指定したlearning rateはinitial_lrとして使用されます")
  

            # trainable_paramsがgroupだった時の処理：lrを削除する
            if type(trainable_params) == list and type(trainable_params[0]) == dict:
                has_group_lr = False
                for group in trainable_params:
                    p = group.pop("lr", None)
                    has_group_lr = has_group_lr or (p is not None)

            lr = None
        #TODO

        # else:
        #     if args.max_grad_norm != 0.0:
        #         logger.warning(
        #             f"because max_grad_norm is set, clip_grad_norm is enabled. consider set to 0 / max_grad_normが設定されているためclip_grad_normが有効になります。0に設定して無効にしたほうがいいかもしれません"
        #         )
        #     if args.lr_scheduler != "constant_with_warmup":
        #         logger.warning(f"constant_with_warmup will be good / スケジューラはconstant_with_warmupが良いかもしれません")
        #     if optimizer_kwargs.get("clip_threshold", 1.0) != 1.0:
        #         logger.warning(f"clip_threshold=1.0 will be good / clip_thresholdは1.0が良いかもしれません")


    return optim(network, lr = lr, **optimizer_kwargs) if name == "AdamMini".lower() else  optim(trainable_params, lr = lr, **optimizer_kwargs) 


def get_random_resolution_in_bucket(bucket_resolution: int = 512) -> tuple[int, int]:
    max_resolution = bucket_resolution
    min_resolution = bucket_resolution // 2

    step = 64

    min_step = min_resolution // step
    max_step = max_resolution // step

    height = torch.randint(min_step, max_step, (1,)).item() * step
    width = torch.randint(min_step, max_step, (1,)).item() * step

    return height, width

def load_noise_scheduler(name: str,v_parameterization: bool):
    sched_init_args = {}
    name = name.lower().replace(" ", "_")
    if name == "ddim":
        scheduler_cls = DDIMScheduler
    elif name == "ddpm":
        scheduler_cls = DDPMScheduler
    elif name == "pndm":
        scheduler_cls = PNDMScheduler
    elif name == "lms" or name == "k_lms":
        scheduler_cls = LMSDiscreteScheduler
    elif name == "euler" or name == "k_euler":
        scheduler_cls = EulerDiscreteScheduler
    elif name == "euler_a" or name == "k_euler_a":
        scheduler_cls = EulerAncestralDiscreteScheduler
    elif name == "dpmsolver" or name == "dpmsolver++":
        scheduler_cls = DPMSolverMultistepScheduler
        sched_init_args["algorithm_type"] = name
    elif name == "dpmsingle":
        scheduler_cls = DPMSolverSinglestepScheduler
    elif name == "heun":
        scheduler_cls = HeunDiscreteScheduler
    elif name == "dpm_2" or name == "k_dpm_2":
        scheduler_cls = KDPM2DiscreteScheduler
    elif name == "dpm_2_a" or name == "k_dpm_2_a":
        scheduler_cls = KDPM2AncestralDiscreteScheduler
    else:
        scheduler_cls = DDIMScheduler
        "Selected scheduler is not in list, use DDIMScheduler."

    if v_parameterization:
        sched_init_args["prediction_type"] = "v_prediction"

    scheduler = scheduler_cls(
        num_train_timesteps = 1000,
        beta_start = 0.00085,
        beta_end = 0.0120, 
        beta_schedule = "scaled_linear",
        **sched_init_args,
    )

    # clip_sample=Trueにする
    if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is False:
        # print("set clip_sample to True")
        scheduler.config.clip_sample = True

    prepare_scheduler_for_custom_training(scheduler)

    return scheduler

def prepare_scheduler_for_custom_training(noise_scheduler):
    if hasattr(noise_scheduler, "all_snr"):
        return

    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    alpha = sqrt_alphas_cumprod
    sigma = sqrt_one_minus_alphas_cumprod
    all_snr = (alpha / sigma) ** 2

    noise_scheduler.all_snr = all_snr.to("cuda")
    

def load_checkpoint_model(checkpoint_path, t, clip_skip = None, vae = None):
    pipe = StableDiffusionPipeline.from_single_file(checkpoint_path,upcast_attention=True if t.is_sd2 else False)

    text_encoder = pipe.text_encoder
    unet = pipe.unet
    if vae is None and hasattr(pipe, "vae"):
        vae = pipe.vae

    if clip_skip is not None:
        if t.is_sd2:
            text_encoder.config.num_hidden_layers = 24 - (clip_skip - 1)
        else:
            text_encoder.config.num_hidden_layers = 12 - (clip_skip - 1)

    text_model = TextModel(
        pipe.tokenizer,
        pipe.text_encoder,
        None, None, None, None,
        t.sdver,
        clip_skip
    )
    
    if t.use_8bit:
        quantize_(unet, int8_weight_only())
        t.train_model_precision = unet.dtype
    
    del pipe
    return text_model, unet, vae

def load_checkpoint_model_xl(checkpoint_path, t, vae = None):
    pipe = StableDiffusionXLPipeline.from_single_file(checkpoint_path)

    unet = pipe.unet
    if vae is None and hasattr(pipe, "vae"):
        vae = pipe.vae

    text_model = TextModel(
        pipe.tokenizer,
        pipe.text_encoder,
        pipe.tokenizer_2,
        pipe.text_encoder_2,
        None, None,
        t.sdver,
    )
    del pipe
    
    if t.use_8bit:
        quantize_(unet, int8_weight_only())
        t.train_model_precision = unet.dtype
    
    return text_model, unet, vae

def load_checkpoint_model_sd3(checkpoint_path, t, vae = None, clip_l = None, clip_g = None, t5 = None):
    pipe = StableDiffusion3Pipeline.from_single_file(checkpoint_path)

    unet = pipe.transformer
    vae = pipe.vae

    assert clip_l is None and pipe.text_encoder, "ERROR, No CLIP L"
    assert clip_g is None and pipe.text_encoder2, "ERROR, No CLIP g"
    assert t5 is None and pipe.text_encoder3, "ERROR, No T5"

    text_model = TextModel(
        pipe.tokenizer,
        pipe.text_encoder, 
        pipe.tokenizer_2,
        pipe.text_encoder_2,
        pipe.tokenizer_3,
        pipe.text_encoder_3,
        t.sdver
    )
    
    del pipe
    
    if t.use_8bit:
        quantize_(unet, int8_weight_only())
        t.train_model_precision = unet.dtype
        TextModel.quantize()
    
    return text_model, unet, vae

def load_checkpoint_model_flux(checkpoint_path, t, vae = None, clip_l = None, clip_g = None, t5 = None):
    pipe = FluxPipeline.from_single_file(checkpoint_path)

    unet = pipe.transformer
    vae = pipe.vae

    assert clip_l is None and pipe.text_encoder, "ERROR, No CLIP L"
    assert t5 is None and pipe.text_encoder3, "ERROR, No T5"

    text_model = TextModel(
        pipe.tokenizer,
        pipe.text_encoder, 
        None, None,
        pipe.tokenizer_2,
        pipe.text_encoder_2,
        t.sdver
    )
    
    del pipe
    
    if t.use_8bit:
        quantize_(unet, int8_weight_only())
        t.train_model_precision = unet.dtype
        TextModel.quantize()
    
    return text_model, unet, vae

try:
    with open(os.path.join(path_root, "pipelines", "configs.json"), 'r', encoding='utf-8') as f:
        master_config = json.load(f)
except:
    master_config = {}

def materialize_remaining_meta_tensors(model, device="cpu", dtype=torch.bfloat16):
    for name, module in model.named_modules():
        for p_name, param in module.named_parameters(recurse=False):
            if param.device.type == 'meta':
                print(f"Materializing missing parameter: {name}.{p_name}")
                new_param = torch.nn.Parameter(torch.empty(param.size(), device=device, dtype=dtype))
                setattr(module, p_name, new_param)

        for b_name, buffer in module.named_buffers(recurse=False):
            if buffer.device.type == 'meta':
                print(f"Materializing missing buffer: {name}.{b_name}")
                new_buffer = torch.empty(buffer.size(), device=device, dtype=dtype)
                setattr(module, b_name, new_buffer)
                
def load_checkpoint_model_zimage(checkpoint_path, t, vae_path=None, te_path=None):

    def detect_state_dict_dtype(state_dict):
        for _, v in state_dict.items():
            if torch.is_tensor(v):
                return v.dtype
        return torch.float32

    zbase = os.path.join(".", "backend", "huggingface", "Tongyi-MAI", "Z-Image-Turbo")
    print("[Z-Image Loader] Loading text encoder config from:", os.path.join(zbase, "text_encoder"))

    te_config = AutoConfig.from_pretrained(os.path.join(zbase, "text_encoder"), trust_remote_code=True, local_files_only=True)

    if not (t.mode == "ADDifT" or t.mode == "Multi-ADDifT") and te_path is not None:
        te_state = load_file(te_path)
        print("TextEncoder StateDict has been read.")
        te_dtype = detect_state_dict_dtype(te_state)
        print(f"[TextEncoder] Detected dtype: {te_dtype}")
        with torch.device("meta"):
            text_encoder = AutoModelForCausalLM.from_config(te_config, trust_remote_code=True, torch_dtype=te_dtype)
        missing, unexpected = text_encoder.load_state_dict(te_state, strict=False, assign=True)
        print("[TextEncoder] missing keys:", missing)
        print("[TextEncoder] unexpected keys:", unexpected)
        if missing:
            print("Materializing missing TextEncoder parameters...")
            materialize_remaining_meta_tensors(text_encoder, dtype=te_dtype)
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(zbase, "tokenizer"), trust_remote_code=True, local_files_only=True)
        print("[Z-Image Loader] Loading transformer config from:", os.path.join(zbase, "transformer"))
        if t.use_8bit:
            quantize_(text_encoder, int8_weight_only())
        text_encoder = text_encoder.to(CPU)
    else:
        text_encoder, tokenizer = None, None

    tr_config = ZImageTransformer2DModel.load_config(os.path.join(zbase, "transformer"))
    tr_state = load_file(checkpoint_path)
    tr_state = remap_zimage_transformer_state_dict(tr_state)
    tr_dtype = detect_state_dict_dtype(tr_state)
    print(f"[Transformer] Detected dtype: {tr_dtype}")
    with torch.device("meta"):
        transformer = ZImageTransformer2DModel.from_config(tr_config, torch_dtype=tr_dtype)
    missing, unexpected = transformer.load_state_dict(tr_state, strict=False, assign=True)   
    print("[Transformer] missing keys:", missing)
    print("[Transformer] unexpected keys:", unexpected)
    if missing:
        print("Materializing missing Transformer parameters...")
        materialize_remaining_meta_tensors(transformer, dtype=tr_dtype)
    if t.use_8bit:
        quantize_(transformer, int8_weight_only())
        t.train_model_precision = transformer.dtype
    if t.read_model_dtype:
        t.train_model_precision = transformer.dtype
    transformer = transformer.to(CPU)

    if isinstance(vae_path, (str, os.PathLike)):
        vae_config = AutoencoderKL.load_config(os.path.join(zbase, "vae"))
        vae_state = load_file(vae_path)
        vae_dtype = detect_state_dict_dtype(vae_state)
        print(f"[VAE] Detected dtype: {vae_dtype}")
        with torch.device("meta"):
            vae = AutoencoderKL.from_config(vae_config, torch_dtype=vae_dtype)
        missing, unexpected = vae.load_state_dict(vae_state, strict=False, assign=True)
        vae = vae.to(CPU)
    else:
        vae = vae_path

    pipe = ZImagePipeline(transformer=transformer,text_encoder=text_encoder,tokenizer=tokenizer,vae=vae,scheduler=None)

    text_model = TextModel(tn3 = pipe.tokenizer, te3 = pipe.text_encoder, sdver = t.sdver)
    transformer = pipe.transformer
    vae = pipe.vae
    
    del pipe
    flush()
    return text_model, transformer, vae

def remap_zimage_transformer_state_dict(sd: dict) -> dict:
    new_sd = {}

    for k, v in sd.items():
        new_k = None
        if k.startswith("x_embedder."):
            suffix = k[len("x_embedder."):]
            new_k = f"all_x_embedder.2-1.{suffix}"

        elif k.startswith("final_layer."):
            suffix = k[len("final_layer."):]
            new_k = f"all_final_layer.2-1.{suffix}"

        elif ".attention.q_norm." in k:
            new_k = k.replace(".attention.q_norm.", ".attention.norm_q.")

        elif ".attention.k_norm." in k:
            new_k = k.replace(".attention.k_norm.", ".attention.norm_k.")

        elif ".attention.out." in k:
            new_k = k.replace(".attention.out.", ".attention.to_out.0.")

        elif k.endswith(".attention.qkv.weight"):
            base = k.replace(".attention.qkv.weight", "")

            q_w, k_w, v_w = torch.chunk(v, 3, dim=0)

            new_sd[f"{base}.attention.to_q.weight"] = q_w
            new_sd[f"{base}.attention.to_k.weight"] = k_w
            new_sd[f"{base}.attention.to_v.weight"] = v_w
            continue 

        elif k.endswith(".attention.qkv.bias"):
            base = k.replace(".attention.qkv.bias", "")
            q_b, k_b, v_b = torch.chunk(v, 3, dim=0)
            new_sd[f"{base}.attention.to_q.bias"] = q_b
            new_sd[f"{base}.attention.to_k.bias"] = k_b
            new_sd[f"{base}.attention.to_v.bias"] = v_b
            continue

        else:
            new_k = k

        new_sd[new_k] = v

    return new_sd


class TextModel(nn.Module):
    def __init__(self, tn1 = None, te1 = None, tn2 = None, te2 = None, tn3 = None, te3 = None, sdver = 0, clip_skip=-1):
        super().__init__()
        self.tokenizers = [tn1, tn2, tn3]
        self.text_encoders = nn.ModuleList([te1, te2, te3])
        self.clip_skip = clip_skip if clip_skip is not None else -1
        self.textual_inversion = False
        self.sdver = sdver

    def tokenize(self, texts):
        tokens = []
        for tokenizer, text_encoder in zip(self.tokenizers, self.text_encoders):
            if tokenizer is None:
                tokens.append(None)
                continue
            token = tokenizer(
                texts, 
                max_length=tokenizer.model_max_length, 
                padding="max_length",
                truncation=True,
                return_tensors='pt'
            ).input_ids.to(text_encoder.device)
            tokens.append(token)
        return tokens

    #sdver: 0:sd1 ,1:sd2, 2:sdxl, 3:sd3, 4:flux 5:zimage
    def forward(self, tokens):
        if 2 > self.sdver:
            return self.encode_sd1_2(tokens)
        elif self.sdver == 2:
            return self.encode_sdxl(tokens)
        elif self.sdver == 3:
            return self.encode_sd3(tokens)
        elif self.sdver == 4:
            return self.encode_flux(tokens)
        elif self.sdver == 5:
            return self.encode_zimage(tokens)
        
    def encode_sd1_2(self, tokens):
        encoder_hidden_states = self.text_encoders[0](tokens[0], output_hidden_states=True).hidden_states[self.clip_skip]
        encoder_hidden_states = self.text_encoders[0].text_model.final_layer_norm(encoder_hidden_states)
        return encoder_hidden_states, None
    
    def encode_sdxl(self, tokens):
        encoder_hidden_states = self.text_encoders[0](tokens[0], output_hidden_states=True).hidden_states[self.clip_skip]
        encoder_output_2 = self.text_encoders[1](tokens[1], output_hidden_states=True)
        last_hidden_state = encoder_output_2.last_hidden_state

        # calculate pooled_output
        eos_token_index = torch.where(tokens[1] == self.tokenizers[1].eos_token_id)[1].to(device=last_hidden_state.device)
        pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),eos_token_index]
        pooled_output = self.text_encoders[1].text_projection(pooled_output)

        encoder_hidden_states_2 = encoder_output_2.hidden_states[self.clip_skip]

        # (b, n, 768) + (b, n, 1280) -> (b, n, 2048)
        encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_2], dim=2)

        # pooled_output is zero vector for empty text            
        for i, token in enumerate(tokens[1]):
            if token[1].item() == self.tokenizers[1].eos_token_id:
                pooled_output[i] = 0

        return encoder_hidden_states, pooled_output
    
    def encode_sd3(self, tokens):
        encoder_output = self.text_encoders[0](tokens[0], output_hidden_states=True)
        last_hidden_state = encoder_output.last_hidden_state
        # calculate pooled_output
        eos_token_index = torch.where(tokens[0] == self.tokenizers[0].eos_token_id)[1][0].to(device=last_hidden_state.device)
        pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), eos_token_index]
        pooled_output = self.text_encoders[0].text_projection(pooled_output)

        encoder_hidden_states = encoder_output.hidden_states[self.clip_skip]

        encoder_output_2 = self.text_encoders[1](tokens[1], output_hidden_states=True)
        last_hidden_state = encoder_output_2.last_hidden_state
        # calculate pooled_output
        eos_token_index = torch.where(tokens[1] == self.tokenizers[1].eos_token_id)[1].to(device=last_hidden_state.device)
        pooled_output_2 = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),eos_token_index]
        pooled_output_2 = self.text_encoders[1].text_projection(pooled_output_2)

        encoder_hidden_states_2 = encoder_output_2.hidden_states[self.clip_skip]

        encoder_hidden_states_3 = self.text_encoders[2](tokens[2], output_hidden_states=False)[0]

        # (b, n, 768) + (b, n, 1280) -> (b, n, 2048)
        encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_2], dim=2)

        # pad
        encoder_hidden_states = torch.cat([encoder_hidden_states, torch.zeros_like(encoder_hidden_states)], dim=2)
        encoder_hidden_states = torch.cat([encoder_hidden_states, encoder_hidden_states_3], dim=1) # t5

        pooled_output = torch.cat([pooled_output, pooled_output_2], dim=1)

        return encoder_hidden_states, pooled_output

    def encode_flux(self, tokens):
        pooled_output = self.text_encoders[0](tokens[0], output_hidden_states=False).pooler_output
        encoder_hidden_states = self.text_encoders[2](tokens[2], output_hidden_states=False)[0]

        return encoder_hidden_states, pooled_output

    def encode_zimage(self, prompt):
        max_sequence_length = 1024
        prompt = [prompt]

        for i, prompt_item in enumerate(prompt):
            messages = [{"role": "user", "content": prompt_item},]
            prompt_item = self.tokenizers[2].apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=True)
            prompt[i] = prompt_item

        text_inputs = self.tokenizers[2](prompt,padding="max_length",max_length=max_sequence_length,truncation=True,return_tensors="pt")

        text_input_ids = text_inputs.input_ids.to(CUDA)
        prompt_masks = text_inputs.attention_mask.to(CUDA).bool()
        prompt_embeds = self.text_encoders[2](input_ids=text_input_ids,attention_mask=prompt_masks,output_hidden_states=True).hidden_states[-2]
        embeddings_list = []
        for i in range(len(prompt_embeds)):
            embeddings_list.append(prompt_embeds[i][prompt_masks[i]])
        return embeddings_list[0], None

    def encode_text(self, texts):
        if self.sdver == 5:
            encoder_hidden_states, pooled_output = self.encode_zimage(texts)
        else:
            tokens = self.tokenize(texts)
            encoder_hidden_states, pooled_output = self.forward(tokens)
        return encoder_hidden_states, pooled_output

    def gradient_checkpointing_enable(self, enable=True):
        if enable:
            for te in self.text_encoders:
                if te is not None:
                    for param in te.parameters():
                        param.requires_grad = True
                    te.gradient_checkpointing_enable()
        else:
            for te in self.text_encoders:
                if te is not None:
                    te.gradient_checkpointing_disable()

    def to(self, device=None, dtype=None):
        super().to(device=device)
        for i, te in enumerate(self.text_encoders):
            if te is not None:
                self.text_encoders[i] = te.to(device=device, dtype=dtype)
        return self
    
    def apop(self):
        for i, te in enumerate(self.text_encoders):
            if te is not None:
                self.text_encoders[i] = self.text_encoders[i].to("cpu")
            self.text_encoders[i] = None 

    def quantize(self):
        for i, te in enumerate(self.text_encoders):
            if te is not None:
                quantize_(self.text_encoders[i], int8_weight_only())
                
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def make_accelerator(t):
    accelerator = Accelerator(
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        mixed_precision=parse_precision(t.train_model_precision, mode = False)
    )
    print(parse_precision(t.train_model_precision, mode = False))

    return accelerator

def parse_precision(precision, mode = True):
    print(precision, mode)
    if mode:
        if precision == "fp32" or precision == "float32":
            return torch.float32
        elif precision == "fp16" or precision == "float16" or precision == "fp8":
            return torch.float16
        elif precision == "bf16" or precision == "bfloat16":
            return torch.bfloat16
        elif precision == "model":
            return torch.float16
        elif precision == "int8":
            return torch.float16
    else:
        if precision == torch.float16 or precision == "fp8":
            return 'fp16'
        elif precision == torch.bfloat16:
            return 'bf16'

    raise ValueError(f"Invalid precision type: {precision}")
    
def load_lr_scheduler(t, optimizer):
    if t.train_optimizer == "adafactor":
        return AdafactorSchedule(optimizer)
    
    args = t.train_lr_scheduler_settings
    print(f"LR Scheduler args: {args}")
    
    # アニーリング系のスケジューラを追加
    if t.train_lr_scheduler == "cosine_annealing":
        return CosineAnnealingLR(
            optimizer,
            T_max=t.train_iterations,
            **args
        )
    elif t.train_lr_scheduler == "cosine_annealing_with_restarts":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.pop("T_0") if "T_0" in args else 10,
            **args
        )
    elif t.train_lr_scheduler == "exponential":
        return ExponentialLR(
            optimizer,
            gamma=args.pop("gamma") if "gamma" in args else 0.9,
            **args
        )
    elif t.train_lr_scheduler == "step":
        return StepLR(
            optimizer,
            step_size=args.pop("step_size") if "step_size" in args else 10,
            **args
        )
    elif t.train_lr_scheduler == "multi_step":
        return MultiStepLR(
            optimizer,
            milestones=args.pop("milestones") if "milestones" in args else [30, 60, 90],
            **args
        )
    elif t.train_lr_scheduler == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            **args
        )
    elif t.train_lr_scheduler == "cyclic":
        return CyclicLR(
            optimizer,
            base_lr=args.pop("base_lr") if "base_lr" in args else  1e-5,
            max_lr=args.pop("max_lr") if "max_lr" in args else  1e-3,
            mode='triangular',
            **args
        )
    elif t.train_lr_scheduler == "one_cycle":
        return OneCycleLR(
            optimizer,
            max_lr=args.pop("max_lr") if "max_lr" in args else  1e-3,
            total_steps=t.train_iterations,
            **args
        )
    
    return get_scheduler(
        name=t.train_lr_scheduler,
        optimizer=optimizer,
        step_rules=t.train_lr_step_rules,
        num_warmup_steps=t.train_lr_warmup_steps if t.train_lr_warmup_steps > 0 else 0,
        num_training_steps=t.train_iterations,
        num_cycles=t.train_lr_scheduler_num_cycles if t.train_lr_scheduler_num_cycles > 0 else 1,
        power=t.train_lr_scheduler_power if t.train_lr_scheduler_power > 0 else 1.0,
        **t.train_lr_scheduler_settings
    )

def load_torch_file(ckpt, safe_load=False, device=None):
    if not safe_load:
        try:
            from modules import checkpoint_pickle
        except:
            safe_load = True
    if device is None:
        device = torch.device("cpu")
    if ckpt.lower().endswith(".safetensors"):
        sd = safetensors.torch.load_file(ckpt, device=device.type)
    else:
        if safe_load:
            if not 'weights_only' in torch.load.__code__.co_varnames:
                print("Warning torch.load doesn't support weights_only on this pytorch version, loading unsafely.")
                safe_load = False
        if safe_load:
            pl_sd = torch.load(ckpt, map_location=device, weights_only=True)
        else:
            pl_sd = torch.load(ckpt, map_location=device, pickle_module=checkpoint_pickle)
        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd
    return sd   


VAE_CONFIG_SD1 ={
  "_class_name": "AutoencoderKL",
  "_diffusers_version": "0.6.0",
  "act_fn": "silu",
  "block_out_channels": [128,256,512,512],
  "down_block_types": ["DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D"],
  "in_channels": 3,
  "latent_channels": 4,
  "layers_per_block": 2,
  "norm_num_groups": 32,
  "out_channels": 3,
  "sample_size": 256,
  "up_block_types": ["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
}

VAE_CONFIG_SD2 = {
  "_class_name": "AutoencoderKL",
  "_diffusers_version": "0.8.0",
  "_name_or_path": "hf-models/stable-diffusion-v2-768x768/vae",
  "act_fn": "silu",
  "block_out_channels": [128,256,512,512],
  "down_block_types": ["DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D"],
  "in_channels": 3,
  "latent_channels": 4,
  "layers_per_block": 2,
  "norm_num_groups": 32,
  "out_channels": 3,
  "sample_size": 768,
  "up_block_types": ["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
}

VAE_CONFIG_SDXL={
  "_class_name": "AutoencoderKL",
  "_diffusers_version": "0.20.0.dev0",
  "_name_or_path": "",
  "act_fn": "silu",
  "block_out_channels": [128,256,512,512],
  "down_block_types": ["DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D"],
  "force_upcast": True,
  "in_channels": 3,
  "latent_channels": 4,
  "layers_per_block": 2,
  "norm_num_groups": 32,
  "out_channels": 3,
  "sample_size": 1024,
  "scaling_factor": 0.13025,
  "up_block_types": ["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
}

VAE_CONFIG_SD3 = {
  "_class_name": "AutoencoderKL",
  "_diffusers_version": "0.29.0.dev0",
  "act_fn": "silu",
  "block_out_channels": [128,256,512,512],
  "down_block_types": ["DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D"],
  "force_upcast": True,
  "in_channels": 3,
  "latent_channels": 16,
  "latents_mean": None,
  "latents_std": None,
  "layers_per_block": 2,
  "norm_num_groups": 32,
  "out_channels": 3,
  "sample_size": 1024,
  "scaling_factor": 1.5305,
  "shift_factor": 0.0609,
  "up_block_types": ["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
  "use_post_quant_conv": False,
  "use_quant_conv": False
}

VAE_CONFIG_FLUX = {
  "_class_name": "AutoencoderKL",
  "_diffusers_version": "0.30.0.dev0",
  "_name_or_path": "../checkpoints/flux-dev",
  "act_fn": "silu",
  "block_out_channels": [128,256,512,512],
  "down_block_types": ["DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D","DownEncoderBlock2D"],
  "force_upcast": True,
  "in_channels": 3,
  "latent_channels": 16,
  "latents_mean": None,
  "latents_std": None,
  "layers_per_block": 2,
  "mid_block_add_attention": True,
  "norm_num_groups": 32,
  "out_channels": 3,
  "sample_size": 1024,
  "scaling_factor": 0.3611,
  "shift_factor": 0.1159,
  "up_block_types": ["UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D","UpDecoderBlock2D"],
  "use_post_quant_conv": False,
  "use_quant_conv": False
}
VAE_CONFIGS = [VAE_CONFIG_SD1, VAE_CONFIG_SD2, VAE_CONFIG_SDXL, VAE_CONFIG_SD3, VAE_CONFIG_FLUX, VAE_CONFIG_FLUX]

from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint

#### Load VAE ####################################################    
def load_VAE(t, path):
    vae_config = VAE_CONFIGS[t.sdver]
    state_dict = load_torch_file(path,safe_load=True)
    state_dict = convert_ldm_vae_checkpoint(state_dict, vae_config)
    vae = AutoencoderKL(**vae_config)
    vae.load_state_dict(state_dict, strict=False)
    vae.eval()
    print("VAE is loaded from", path)
    return vae

#### TextEncoders ################################################
te_list = {}

try:
    te_paths: set[str] = {
        os.path.abspath(os.path.join(paths.models_path, "text_encoder")),
        *shared.cmd_opts.text_encoder_dirs,
    }

    def find_files_with_extensions(base_path, extensions):
        found_files = {}
        for root, _, files in os.walk(base_path):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    full_path = os.path.join(root, file)
                    found_files[file] = full_path
        return found_files

    for te_paths in te_paths:
        te_paths = find_files_with_extensions(te_paths, EXTENTIONS)
        te_list.update(te_paths)
except Exception as e: 
    print(e)
    
def read_arbitrary_config(name):
    config_path = os.path.join(path_root, f"{name}.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.json file found in the directory: {path_root}")

    with open(config_path, "rt", encoding="utf-8") as file:
        config_data = json.load(file)

    return config_data

def flush():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()