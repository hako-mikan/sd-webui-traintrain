import os
import csv
import random
import time
import numpy
import gc
import json
from PIL import Image
import traceback
import torch
from tqdm import tqdm
from modules import sd_models, sd_vae, shared, prompt_parser, scripts, lowvram
from trainer.lora import LoRANetwork, LycorisNetwork
from trainer import trainer, dataset
from diffusers.optimization import get_scheduler
from pprint import pprint
from accelerate.utils import set_seed
from diffusers.models import AutoencoderKL
from transformers.optimization import AdafactorSchedule


MAX_DENOISING_STEPS = 1000
ML = "LoRA"
MD = "Difference"

try:
    from ldm_patched.modules import model_management
    forge = True
except:
    forge = False

jsonspath = trainer.jsonspath
logspath = trainer.logspath
presetspath = trainer.presetspath

stoptimer = 0

CUDA = torch.device("cuda:0")

queue_list = []
current_name = None

#dfalse, mode, model, vae, *train_settings_1, *train_settings_2, *prompts, *in_images

def get_name_index(wanted):
    for i, name in enumerate(trainer.all_configs):
        if name[0] == wanted:
            return i

def queue(*args):
    global queue_list
    name_index = get_name_index("save_lora_name") + 4
    dup = args[name_index] == current_name
    for queue in queue_list:
        if queue[name_index] == args[name_index]:
            dup = True
    if dup:
        return ("Duplicated LoRA name! Could not add to queue.")

    queue_list.append(args)
    return "Added to Queue"

def get_del_queue_list(del_name = None):
    global queue_list
    name_index = get_name_index("save_lora_name")
    out = []
    del_index = None

    for i, q in enumerate(queue_list):
        data = [*q[1:-2]]
        name = data[name_index + 3]
        data = [name] + data
        if del_name and name == del_name:
            del_index = i
        else:
            out.append(data)
    if del_index:
        del queue_list[del_index]
    return out

def setcurrentname(args):
    name_index = get_name_index("save_lora_name") + 4
    global current_name
    current_name = args[name_index]

def train(*args):
    if not args[0]:
        setcurrentname(args)
    result = train_main(*args)
    while len(queue_list) > 0:
        settings = queue_list.pop(0)
        result +="\n" + train_main(*settings)
    return result

def train_main(jsononly, mode, modelname, vaename, *args):
    t = trainer.Trainer(jsononly, modelname, vaename, mode, args)

    if jsononly:
        return "Preset saved"

    if t.isfile:
        return "File exist!"

    print(" Start Training!")

    currentinfo = shared.sd_model.sd_checkpoint_info
    checkpoint_info = sd_models.get_closet_checkpoint_match(modelname)
    
    lowvram.module_in_gpu = None #web-uiのバグ対策
    sd_models.load_model(checkpoint_info)

    t.isxl = shared.sd_model.is_sdxl
    t.isv2 = shared.sd_model.is_sd2

    t.vae_scale_factor = 0.13025 if t.isxl else 0.18215

    model_ver = "sdxl_base_v1-0" if t.isxl else None
    model_ver = "sd_v2" if t.isv2 else model_ver
    t.model_version = "sd_v1" if model_ver is None else model_ver

    checkpoint_filename = shared.sd_model.sd_checkpoint_info.filename

    if t.mode != ML:
        t.orig_cond, t.orig_vector  = text2cond(t, t.prompts[0])
        t.targ_cond, t.targ_vector  = text2cond(t, t.prompts[1])
        t.un_cond, t.un_vector = text2cond(t, t.prompts[2])

    print("Preparing the Model...")

    if forge:
        sd_models.model_data.sd_model = None
        sd_models.model_data.loaded_sd_models = []
        model_management.unload_all_models()
        model_management.soft_empty_cache()
        gc.collect()
    else:
        sd_models.unload_model_weights()

    lowvram.module_in_gpu = None #web-uiのバグ対策

    vae_path = sd_vae.vae_dict.get(vaename, None)
    vae = AutoencoderKL.from_single_file(vae_path) if vae_path is not None else None

    if t.isxl: 
        text_model, unet, vae = trainer.load_checkpoint_model_xl(checkpoint_filename, t, vae = vae)
    else:
        text_model, unet, vae = trainer.load_checkpoint_model(checkpoint_filename, t, vae = vae)
    
    unet.to(CUDA, dtype=t.train_model_precision)
    try:
        unet.enable_xformers_memory_efficient_attention()
        print("Enabling Xformers")
    except:
        print("Failed to enable Xformers")

    unet.requires_grad_(False)
    unet.eval()

    text_model.to(device = CUDA, dtype = t.train_model_precision)

    if t.use_gradient_checkpointing:
        unet.train()
        unet.enable_gradient_checkpointing()
        text_model.text_encoder.text_model.embeddings.requires_grad_(True)  # 先頭のモジュールが勾配有効である必要があるらしい
        if text_model.text_encoder_2 is not None:
            text_model.text_encoder_2.text_model.embeddings.requires_grad_(True)
        text_model.train() #trainがTrueである必要があるらしい
        text_model.gradient_checkpointing_enable()

    t.unet = unet
    t.text_model = text_model

    vae.to(CUDA, dtype=t.train_model_precision)
    t.vae = vae

    t.text2cond = text2cond
    t.image2latent = image2latent

    t.noise_scheduler = trainer.load_noise_scheduler("ddpm", t.model_v_pred)

    t.a = trainer.make_accelerator(t)
    
    t.unet = t.a.prepare(t.unet)
    t.text_model.text_encoder = t.a.prepare(t.text_model.text_encoder)
    if text_model.text_encoder_2 is not None:
        t.text_model.text_encoder_2 = t.a.prepare(t.text_model.text_encoder_2)

    if 0 > t.train_seed: t.train_seed = random.randint(0, 2**32)
    set_seed(t.train_seed)
    makesavelist(t)
    del vae, text_model, unet

    try:
        if t.mode == ML:
            result = train_lora(t)
        elif t.mode == "iLECO":
            result = train_leco(t)
        elif t.mode == "Difference":
            result = train_diff(t)
        else:
            result = "Test mode"

        print("Done.")

    except Exception as e:
        print(traceback.format_exc())
        result =  f"Error: {e}"

    del t
    flush()

    try:
        sd_models.load_model(currentinfo)
    except:
        lowvram.module_in_gpu = None #web-uiのバグ対策

    sd_models.model_data.loaded_sd_models = [] #web-uiのバグ対策

    return result

def train_lora(t):
    global stoptimer
    stoptimer = 0

    t.a.print("Preparing image latents and text-conditional...")
    dataloaders = dataset.make_dataloaders(t)
    t.dataloader = dataset.ContinualRandomDataLoader(dataloaders)
    t.dataloader = (t.a.prepare(t.dataloader))

    t.a.print("Train LoRA Start")
    
    network, optimizer, lr_scheduler = create_network(t)

    if not t.dataloader.data:
        return "No data!"

    loss_ema = None
    loss_velocity = None

    del t.vae
    if "BASE" not in t.network_blocks:
        del t.text_model
    
    flush()

    pbar = tqdm(range(t.train_iterations))
    while t.train_iterations >= pbar.n:
        for batch in t.dataloader:
            for i in range(t.train_repeat):
                latents = batch["latent"].to(CUDA, dtype=t.train_lora_precision)
                conds1 = batch["cond1"]
                conds2 = batch["cond2"] if "cond2" in batch else None

                noise = torch.randn_like(latents)

                batch_size = latents.shape[0]

                timesteps = torch.randint(t.train_min_timesteps, t.train_max_timesteps, ((1 if t.train_fixed_timsteps_in_batch else batch_size),),device=CUDA) 
                timesteps = torch.cat([timesteps.long()] * (batch_size if t.train_fixed_timsteps_in_batch else 1))

                noisy_latents = t.noise_scheduler.add_noise(latents, noise, timesteps)

                with network, t.a.autocast():
                    if isinstance(conds1[0], str):
                        conds1, conds2 = t.text_model.encode_text(conds1)

                    added_cond_kwargs = get_added_cond_kwargs(t, conds2, batch_size, size = [*latents.shape[2:4]])

                    conds1.to(CUDA, dtype=t.train_lora_precision) 
                    noise_pred = t.unet(noisy_latents, timesteps, conds1, added_cond_kwargs = added_cond_kwargs).sample

                if t.image_use_transparent_background_ajust and "mask" in batch:
                    noise_pred = noise_pred * batch["mask"].to(CUDA) + noise * (1 - batch["mask"].to(CUDA))

                loss, loss_ema, loss_velocity = process_loss(t, noise_pred, noise, timesteps, loss_ema, loss_velocity)

                c_lrs = [f"{x:.2e}" for x in lr_scheduler.get_last_lr()]
                pbar.set_description(f"Loss EMA * 1000: {loss_ema * 1000:.4f}, Current LR: "+", ".join(c_lrs)+ f", Epoch: {t.dataloader.epoch}")   
                pbar.update(1)

                if t.logging_save_csv:
                    savecsv(pbar.n, loss_ema, [x.cpu().item() if isinstance(x, torch.Tensor)  else x for x in lr_scheduler.get_last_lr()],t.csvpath)

                t.a.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                del noise_pred

                flush()
                
                result = finisher(network, t, pbar.n)
                if result is not None:
                    return result

            if pbar.n >=t.train_iterations:
                break

    return savecount(network, t, 0)

def train_leco(t):
    global stoptimer
    stoptimer = 0

    del t.vae, t.text_model
    flush()

    network, optimizer, lr_scheduler = create_network(t)

    t.orig_cond = torch.cat([t.orig_cond] * t.train_batch_size)
    t.targ_cond = torch.cat([t.targ_cond] * t.train_batch_size)

    if t.orig_vector is not None:
        t.orig_vector = torch.cat([t.orig_vector] * t.train_batch_size)
        t.targ_vector = torch.cat([t.targ_vector] * t.train_batch_size)

    height, width = t.image_size

    latents = torch.randn((t.train_batch_size, 4, height // 8, width // 8), device=CUDA,dtype = t.train_model_precision)

    loss_ema = None
    loss_velocity = None

    pbar = tqdm(range(t.train_iterations))
    while t.train_iterations >= pbar.n:
        with torch.no_grad(), t.a.autocast():                
            timesteps = torch.randint(t.train_min_timesteps, t.train_max_timesteps, (t.train_batch_size,),device=CUDA)
            timesteps = timesteps.long()
            added_cond_kwargs = get_added_cond_kwargs(t, t.targ_vector, t.train_batch_size)
            targ_pred = t.unet(latents, timesteps, t.targ_cond, added_cond_kwargs = added_cond_kwargs).sample
        
        added_cond_kwargs = get_added_cond_kwargs(t, t.orig_vector, t.train_batch_size)

        with network, t.a.autocast():
            orig_pred = t.unet(latents, timesteps, t.orig_cond, added_cond_kwargs = added_cond_kwargs).sample

        loss, loss_ema, loss_velocity = process_loss(t, orig_pred, targ_pred, timesteps, loss_ema, loss_velocity)

        c_lrs = [f"{x:.2e}" for x in lr_scheduler.get_last_lr()]

        pbar.set_description(f"Loss EMA * 1000: {loss_ema * 1000:.4f}, Current LR: "+", ".join(c_lrs))   
        pbar.update(1)
        if t.logging_save_csv:
            savecsv(pbar.n, loss_ema, [x.cpu().item() if isinstance(x, torch.Tensor)  else x for x in lr_scheduler.get_last_lr()],t.csvpath)

        t.a.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        flush()

        result = finisher(network, t, pbar.n)
        if result is not None:
            return result 

    return savecount(network, t, 0)


def train_diff(t):
    global stoptimer
    stoptimer = 0

    t.orig_latent = image2latent(t,t.images[0]).to(t.train_model_precision).repeat_interleave(t.train_batch_size,0)
    t.targ_latent = image2latent(t,t.images[1]).to(t.train_model_precision)

    del t.vae, t.text_model
    flush()

    print("Copy Machine Start")
    t.image_size = [*t.orig_latent.shape[2:4]]

    orig_network, result = make_diff_lora(t, True) 
    if "Stopped" in result:
        return "Stopped"

    orig_network.eval()
    orig_network.requires_grad_(False)

    
    if t.diff_1st_pass_only:
        return result

    del t.a
    print("Target LoRA Start")
    t.setpass(1)

    t.a = trainer.make_accelerator(t)
    if 0 > t.train_seed: t.train_seed = random.randint(0, 2**32)
    set_seed(t.train_seed)
    makesavelist(t)

    t.targ_latent = t.targ_latent.repeat_interleave(t.train_batch_size,0)
    t.image_size = [*t.targ_latent.shape[2:4]]
    t.diff_load_1st_pass = ""
    with orig_network:
        _, result = make_diff_lora(t, False)
    return result

def make_diff_lora(t, copy):
    image_latent = t.orig_latent if copy else t.targ_latent
    batch_size = image_latent.shape[0]
    network, optimizer, lr_scheduler = create_network(t)
    added_cond_kwargs = get_added_cond_kwargs(t, torch.cat([t.targ_vector] * batch_size), batch_size) if t.targ_vector is not None else None 

    if t.diff_load_1st_pass and copy:
        return network, ""

    loss_ema = None
    loss_velocity = None

    pbar = tqdm(range(t.train_iterations))
    while t.train_iterations >= pbar.n:
        optimizer.zero_grad()
        noise = torch.randn_like(image_latent)

        timesteps = torch.randint(t.train_min_timesteps, t.train_max_timesteps, ((1 if t.train_fixed_timsteps_in_batch else batch_size),),device=CUDA) 
        timesteps = torch.cat([timesteps.long()] * (batch_size if t.train_fixed_timsteps_in_batch else 1))

        noisy_latents = t.noise_scheduler.add_noise(image_latent, noise, timesteps)

        with network, t.a.autocast():
            noise_pred = t.unet(noisy_latents, timesteps, torch.cat([t.orig_cond] * batch_size), added_cond_kwargs = added_cond_kwargs).sample

        loss, loss_ema, loss_velocity = process_loss(t, noise_pred, noise, timesteps, loss_ema, loss_velocity)

        c_lrs = [f"{x:.2e}" for x in lr_scheduler.get_last_lr()]

        pbar.set_description(f"Loss EMA * 1000: {loss_ema * 1000:.4f}, Loss Velosity: {loss_velocity * 1000:.4f}, Current LR: "+", ".join(c_lrs))   
        pbar.update(1)
        if t.logging_save_csv:
            savecsv(pbar.n, loss_ema, [x.cpu().item() if isinstance(x, torch.Tensor)  else x for x in lr_scheduler.get_last_lr()],t.csvpath, copy = copy)

        t.a.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        flush()
        
        result = finisher(network, t, pbar.n, copy = copy)
        if result is not None:
            del optimizer, lr_scheduler
            return network, result

    result = savecount(network, t, 0, copy = copy)

    return network, result


#### Prepare LoRA, Optimizer, lr_scheduler, Save###############################################
def flush():
    torch.cuda.empty_cache()
    gc.collect()

def create_network(t):
    network = load_network(t)
    t.optimizer_module = trainer.get_optimizer(t.train_optimizer)
    optimizer = t.optimizer_module(network.prepare_optimizer_params(),lr=t.train_learning_rate if t.train_optimizer != "adafactor" else None)
    lr_scheduler = load_lr_scheduler(t, optimizer)

    t.db(vars(lr_scheduler), pp = True)

    network, optimizer, lr_scheduler = t.a.prepare(network, optimizer, lr_scheduler)

    return network, optimizer, lr_scheduler

def load_network(t):
    types = trainer.all_configs[get_name_index("network_type")][2]
    if t.network_type in types[:2]:
        return LoRANetwork(t).to(CUDA, dtype=t.train_lora_precision)
    else:
        return LycorisNetwork(t).to(CUDA, dtype=t.train_lora_precision)

def load_lr_scheduler(t, optimizer):
    if t.train_optimizer == "adafactor":
        return AdafactorSchedule(optimizer)
    else:
        return get_scheduler(
            name = t.train_lr_scheduler,
            optimizer = optimizer,
            step_rules = t.train_lr_step_rules,
            num_warmup_steps = t.train_lr_warmup_steps if t.train_lr_warmup_steps > 0 else 0,
            num_training_steps = t.train_iterations,
            num_cycles = t.train_lr_scheduler_num_cycles if t.train_lr_scheduler_num_cycles > 0 else 1,
            power = t.train_lr_scheduler_power if t.train_lr_scheduler_power > 0 else 1.0
        )

def stop_time(save):
    global stoptimer
    stoptimer = 2 if save else 1

def finisher(network, t, i, copy = False):
    if t.save_list and i >= t.save_list[0]:
        savecount(network, t, t.save_list.pop(0), copy)

    if stoptimer > 0:
        if stoptimer > 1:
            result = ". " + savecount(network, t, i, copy)
        else:
            result = ""
        return "Stopped" + result

def savecount(network, t, i, copy = False):
    if t.metadata == {}:
       metadator(t)
    if copy and not t.diff_save_1st_pass:
        return "Not save copy"
    add = "_copy" if copy else ""
    add = f"{add}_{i}steps" if i > 0 else add
    filename = os.path.join(t.save_dir, f"{t.save_lora_name}{add}.safetensors")
    print(f" Saving to {filename}")
    metaname = f"{t.save_lora_name}{add}"
    network.save_weights(filename, t, metaname)
    return f"Successfully created to {filename}"

def makesavelist(t):
    if t.save_per_steps > 0:
        t.save_list = [x * t.save_per_steps for x in range(1, t.train_iterations // t.save_per_steps + 1)]
        if t.train_iterations in t.save_list:
            t.save_list.remove(t.train_iterations)
    else:
        t.save_list = []

def process_loss(t, original, target, timesteps, loss_ema, loss_velocity):
    loss = torch.nn.functional.mse_loss(original.float(), target.float(), reduction="none")
    loss = loss.mean([1, 2, 3])

    if t.train_snr_gamma > 0:
        loss = apply_snr_weight(loss, timesteps, t.noise_scheduler, t.train_snr_gamma)

    loss = loss.mean()

    if loss_ema is None:
        loss_ema = loss.item()
        loss_velocity = 0
    else:
        loss_velocity = loss_velocity * 0.9 + (loss_ema - (loss_ema * 0.9 + loss.item() * 0.1)) * 0.1
        loss_ema = loss_ema * 0.9 + loss.item() * 0.1
    
    return loss, loss_ema, loss_velocity

#### Anti-Overfitting functions ####################################################
def apply_snr_weight(loss, timesteps, noise_scheduler, gamma):
    snr = torch.stack([noise_scheduler.all_snr[t] for t in timesteps])
    gamma_over_snr = torch.div(torch.ones_like(snr) * gamma, snr)
    snr_weight = torch.minimum(gamma_over_snr, torch.ones_like(gamma_over_snr)).float()  # from paper
    loss = loss * snr_weight
    return loss

#### Encode Latent, Embeddings ####################################################
def image2latent(t,image):
    if isinstance(image, str):
        with Image.open(image) as img:
            image = img
    image = numpy.array(image)
    image = image.astype(numpy.float32) / 255.0
    image = numpy.moveaxis(image, 2, 0)
    image = torch.from_numpy(image).unsqueeze(0)
    image = image * 2 - 1
    image = image.to(CUDA,dtype=t.train_model_precision)
    with torch.no_grad():
        latent = t.vae.encode(image).latent_dist.sample() * t.vae_scale_factor
    return latent

def text2cond(t, prompt):
    input = SdConditioning([prompt], width=t.image_size[0], height=t.image_size[1])
    cond = prompt_parser.get_learned_conditioning(shared.sd_model,input,1)
    if t.isxl:
        return [cond[0][0].cond["crossattn"].unsqueeze(0).to(CUDA, dtype=t.train_model_precision),
                (cond[0][0].cond["vector"][:1280].unsqueeze(0).to(CUDA, dtype=t.train_model_precision))]
    else:
        return (cond[0][0].cond.unsqueeze(0).to(CUDA, dtype=t.train_model_precision)), None

class SdConditioning(list):
    def __init__(self, prompts, is_negative_prompt=False, width=None, height=None, copy_from=None):
        super().__init__()
        self.extend(prompts)

        if copy_from is None:
            copy_from = prompts

        self.is_negative_prompt = is_negative_prompt or getattr(copy_from, 'is_negative_prompt', False)
        self.width = width or getattr(copy_from, 'width', None)
        self.height = height or getattr(copy_from, 'height', None)

def get_added_cond_kwargs(t, projection, batch_size, size = None):
    size = size if size is not None else t.image_size
    size_condition = list(size + [0, 0] + size)
    size_condition = torch.tensor([size_condition], dtype=t.train_model_precision, device=CUDA).repeat(batch_size, 1)
    if projection is not None:
        return {"text_embeds": projection, "time_ids": size_condition}
    else:
        return None

#### Debug, Logging ####################################################
def check_requires_grad(model: torch.nn.Module):
    for name, module in list(model.named_modules())[:5]:
        if len(list(module.parameters())) > 0:
            print(f"Module: {name}")
            for name, param in list(module.named_parameters())[:2]:
                print(f"    Parameter: {name}, Requires Grad: {param.requires_grad}")

def check_training_mode(model: torch.nn.Module):
    for name, module in list(model.named_modules())[:5]:
        print(f"Module: {name}, Training Mode: {module.training}")

def savecsv(step, loss, lr, csvpath, copy=False):
    header = ["Step", "Loss"] + ["Learning Rate " + str(i+1) for i in range(len(lr))]

    if copy:
        csvpath = csvpath.replace(".csv", "_copy.csv")
    
    directory = os.path.dirname(csvpath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    file_exists = os.path.isfile(csvpath)

    with open(csvpath, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([step, loss] + lr)

#### Metadata ####################################################
def metadator(t):
    t.metadata = {
        "ss_session_id": random.randint(0, 2**32),  # random integer indicating which group of epochs the model came from
        "ss_training_started_at": time.time(),  # unix timestamp
        "ss_output_name": t.save_lora_name,
        "ss_learning_rate": t.train_learning_rate,
        "ss_max_train_steps": t.train_iterations,
        "ss_lr_warmup_steps": t.train_lr_warmup_steps,
        "ss_lr_scheduler": t.train_lr_scheduler,
        "ss_network_module": "network.lora",
        "ss_network_dim": t.network_rank,  # None means default because another network than LoRA may have another default dim
        "ss_network_alpha": t.network_alpha,  # some networks may not have alpha
        "ss_mixed_precision": t.train_lora_precision,
        "ss_lr_step_rules":t.train_lr_step_rules,
        "ss_lr_warmup_steps":t.train_lr_warmup_steps,
        "ss_lr_scheduler_num_cycles": t.train_lr_scheduler_num_cycles,
        "ss_lr_scheduler_power": t.train_lr_scheduler_power,
        "ss_v2": bool(t.isv2),
        "ss_base_model_version": t.model_version,
        "ss_seed": t.train_seed,
        "ss_optimizer": t.train_optimizer,
        "ss_min_snr_gamma": t.train_snr_gamma,
        "ss_tag_frequency": json.dumps({1:t.count_dict})
    }
