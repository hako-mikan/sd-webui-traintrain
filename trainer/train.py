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
from torch.nn import ModuleList
from tqdm import tqdm
import safetensors.torch

from pprint import pprint
from accelerate.utils import set_seed

try:
    from modules import sd_models, sd_vae, shared, prompt_parser
    standalone = False
except:
    from modules import checkpoint_pickle
    standalone = True
    forge = False
        
if standalone:
    from traintrain.trainer.lora import LoRANetwork, LycorisNetwork
    from traintrain.trainer import trainer, dataset
else:
    from trainer.lora import LoRANetwork, LycorisNetwork
    from trainer import trainer, dataset
       
    try:
        from modules.sd_models import forge_model_reload, model_data
        from modules_forge.main_entry import forge_unet_storage_dtype_options
        from backend.memory_management import free_memory
        forge = True
    except:
        forge = False
        
    try:
        from modules import lowvram
        neo = False
    except:
        neo = True
        
MODEL_LIST = ["SD1", "SD2", "SDXL", "SD3", "FLUX","ZIMAGE","REFINER", "UNKNOWN"]
MAX_DENOISING_STEPS = 1000
ML = "LoRA"
MD = "Difference"

jsonspath = trainer.jsonspath
logspath = trainer.logspath
presetspath = trainer.presetspath

stoptimer = 0

CUDA = torch.device("cuda:0")
CPU = torch.device("cpu")

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
    flush()
    return result

def train_main(jsononly_or_paths, mode, modelname, vaename, tename, *args):
    t = trainer.Trainer(jsononly_or_paths, modelname, vaename, tename, mode, args)

    if type(jsononly_or_paths) is bool and jsononly_or_paths == True:
        return "Preset saved"

    if t.isfile:
        return "File exist!"

    if modelname == "":
        return "No Model Selected."

    print(" Start Training!")

    if standalone:
        checkpoint_filename = modelname
        if not os.path.exists(modelname):
            checkpoint_filename = os.path.join(t.models_dir, modelname) if hasattr(t, "models_dir") else modelname
        state_dict = trainer.load_torch_file(checkpoint_filename)
        model_version = detect_model_version(state_dict)
        del state_dict
        flush()
        t.sd_typer(ver=model_version)
        vae = None
        vae_path = None if vaename in ["", "None"] else vaename
        if vae_path is not None and not os.path.exists(vaename):
            if hasattr(t, "vae_dir"):
                vae_path = os.path.join(t.vae_dir, vae_path)
        text_encoder = None
        te_path = None if vaename in ["", "None"] else tename
        if te_path is not None and not os.path.exists(tename):
            if hasattr(t, "te_dir"):
                te_path = os.path.join(t.vae_dir, te_path)

    else:
        currentinfo = shared.sd_model.sd_checkpoint_info if hasattr(shared.sd_model, "sd_checkpoint_info") else None

        checkpoint_info = sd_models.get_closet_checkpoint_match(modelname)

        if not neo:lowvram.module_in_gpu = None #web-uiのバグ対策
        
        if forge:
            unet_storage_dtype, _ = forge_unet_storage_dtype_options.get(shared.opts.forge_unet_storage_dtype, (None, False))
            forge_model_params = dict(
                checkpoint_info=checkpoint_info,
                additional_modules=shared.opts.forge_additional_modules,
                unet_storage_dtype=unet_storage_dtype
            )
            model_data.forge_loading_parameters = forge_model_params
            forge_model_reload()
            vae = None if neo else shared.sd_model.forge_objects.vae.first_stage_model 
        else:
            sd_models.load_model(checkpoint_info)
            vae = None
            
        te_path = trainer.te_list[tename] if tename != "None" else None
        t.sd_typer()

        checkpoint_filename = shared.sd_model.sd_checkpoint_info.filename

        t.orig_cond, t.orig_vector  = text2cond(t, t.prompts[0])
        t.targ_cond, t.targ_vector  = text2cond(t, t.prompts[1])
        t.un_cond, t.un_vector = text2cond(t, t.prompts[2])

        print("Preparing the Model...")

        if forge:
            sd_models.model_data.sd_model = None
            sd_models.model_data.loaded_sd_models = []
            free_memory(0,CUDA, free_all = True)
            gc.collect()
        else:
            sd_models.unload_model_weights()

        if not neo:lowvram.module_in_gpu = None #web-uiのバグ対策

        vae_path = sd_vae.vae_dict.get(vaename, None)
        
    if not vae:
        vae = trainer.load_VAE(t, vae_path) if vae_path is not None else None

    if t.is_sdxl: 
        text_model, unet, vae = trainer.load_checkpoint_model_xl(checkpoint_filename, t, vae = vae)
    elif t.is_zimage:
        text_model, unet, vae = trainer.load_checkpoint_model_zimage(checkpoint_filename, t, vae_path = vae,te_path = te_path)
    else:
        text_model, unet, vae = trainer.load_checkpoint_model(checkpoint_filename, t, vae = vae)
    
    try:
        unet.enable_xformers_memory_efficient_attention()
        print("Enabling Xformers")
    except:
        print("Failed to enable Xformers")

    unet.requires_grad_(False)
    unet.eval()

    text_model = text_model.to(device = CUDA, dtype = t.train_model_precision)
    text_model.requires_grad_(False)
    text_model.eval()

    if t.use_gradient_checkpointing:
        unet.train()
        unet.enable_gradient_checkpointing()
        if not t.is_zimage:
            text_model.train()
            text_model.gradient_checkpointing_enable()

    t.unet = unet
    t.text_model = text_model
    t.vae = vae
    
    t.text2cond = text2cond
    t.image2latent = image2latent

    t.noise_scheduler = trainer.load_noise_scheduler("ddpm", t.model_v_pred)

    t.a = trainer.make_accelerator(t)

    t.unet = t.a.prepare(t.unet, device_placement=[False])
    t.text_model.text_encoders = ModuleList([t.a.prepare(te) if te is not None else None for te in t.text_model.text_encoders])

    if 0 > t.train_seed: t.train_seed = random.randint(0, 2**32)
    set_seed(t.train_seed)
    makesavelist(t)
    
    if standalone:
        with torch.no_grad(), t.a.autocast():
            t.orig_cond, t.orig_vector  = t.text_model.encode_text(t.prompts[0])
            t.targ_cond, t.targ_vector  = t.text_model.encode_text(t.prompts[1])
            t.un_cond, t.un_vector = t.text_model.encode_text(t.prompts[2])
    
    del vae, text_model, unet

    try:
        if t.mode == ML:
            result = train_lora(t)
        elif t.mode == "iLECO":
            result = train_leco(t)
        elif t.mode == "Difference":
            result = train_diff(t)
        elif t.mode ==  "ADDifT" or t.mode == "Multi-ADDifT":
            result = train_diff2(t)
        else:
            result = "Test mode"

        print("Done.")

    except Exception as e:
        print(traceback.format_exc())
        result =  f"Error: {e}"
    
    t.unet.to("cpu")   
    t.unet = None
    del t
    flush()
    
    if standalone:
        return result
    
    try:
        if forge:
            forge_model_params["checkpoint_info"] = currentinfo if currentinfo else checkpoint_info
            model_data.forge_loading_parameters = forge_model_params
            model_data.forge_hash = None
            forge_model_reload()
        else:
            sd_models.load_model(currentinfo)
    except:
        if not neo:lowvram.module_in_gpu = None #web-uiのバグ対策

    if not forge: sd_models.model_data.loaded_sd_models = [] #web-uiのバグ対策

    return result
def train_lora(t):
    global stoptimer
    stoptimer = 0
    
    t.vae = t.vae.to(CUDA, dtype=t.train_VAE_precision)
    t.a.print("Preparing image latents and text-conditional...")
    dataloaders = dataset.make_dataloaders(t)
    t.dataloader = dataset.ContinualRandomDataLoader(dataloaders)
    t.dataloader = (t.a.prepare(t.dataloader))
    
    del t.vae
    if "BASE" not in t.network_blocks:
        t.text_model.apop()
        del t.text_model
    flush()
    print("t.train_model_precision",t.train_model_precision)
    t.unet = t.unet.to(CUDA, dtype=t.train_model_precision)
    
    t.a.print("Train LoRA Start")
    
    network, optimizer, lr_scheduler = create_network(t)

    if not t.dataloader.data:
        return "No data!"

    loss_ema = None
    loss_velocity = None

    pbar = tqdm(range(t.train_iterations))
    while t.train_iterations >= pbar.n:
        for batch in t.dataloader:
            for i in range(t.train_repeat):
                latents = batch["latent"].to(CUDA, dtype=t.train_lora_precision)
                conds1 = batch["cond1"]
                conds2 = batch["cond2"] if "cond2" in batch else None

                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]

                if t.is_zimage:
                    timesteps, noisy_latents, target = apply_flow_matching_noise(latents, noise, 3.0)
                else:
                    timesteps = torch.randint(t.train_min_timesteps, t.train_max_timesteps, ((1 if t.train_fixed_timsteps_in_batch else batch_size),), device=CUDA) 
                    timesteps = torch.cat([timesteps.long()] * (batch_size if t.train_fixed_timsteps_in_batch else 1))
                    noisy_latents = t.noise_scheduler.add_noise(latents, noise, timesteps)
                    target = noise 

                with network, t.a.autocast():
                    if isinstance(conds1[0], str):
                        conds1, conds2 = t.text_model.encode_text(conds1)
                    
                    conds1 = conds1.to(CUDA, dtype=t.train_lora_precision) if type(conds1) is torch.Tensor else [c.to(CUDA, dtype=t.train_lora_precision) for c in conds1]
                    if conds2 is not None:
                        conds2 = conds2.to(CUDA, dtype=t.train_lora_precision)

                    added_cond_kwargs = get_added_cond_kwargs(t, conds2, batch_size, size = [*latents.shape[2:4]])
 
                    if t.is_zimage:
                        noisy_latents_in = noisy_latents.unsqueeze(2)
                        noisy_latents_list = list(noisy_latents_in.unbind(0))
                        noise_pred_out, _ = t.unet(noisy_latents_list, timesteps, conds1)
                        
                        noise_pred = -torch.stack(noise_pred_out, dim=0).squeeze(dim=2)
                    else:
                        noise_pred = t.unet(noisy_latents, timesteps, conds1, added_cond_kwargs = added_cond_kwargs).sample

                if t.model_v_pred:
                    target = t.noise_scheduler.get_velocity(latents, noise, timesteps)

                loss, loss_ema, loss_velocity = process_loss(t, noise_pred, target, timesteps, loss_ema, loss_velocity)

                c_lrs = [f"{x:.2e}" for x in lr_scheduler.get_last_lr()]
                pbar.set_description(f"Loss EMA * 1000: {loss_ema * 1000:.4f}, Current LR: "+", ".join(c_lrs)+ f", Epoch: {t.dataloader.epoch}")   
                pbar.update(1)

                if t.logging_save_csv:
                    savecsv(t, pbar.n, loss_ema, [x.cpu().item() if isinstance(x, torch.Tensor)  else x for x in lr_scheduler.get_last_lr()],t.csvpath)

                t.a.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                del noise_pred, noise

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

    del t.vae
    if "BASE" not in t.network_blocks:
        del t.text_model
    flush()
    
    t.unet = t.unet.to(CUDA, dtype=t.train_model_precision)
   
    network, optimizer, lr_scheduler = create_network(t)

    t.orig_cond = torch.cat([t.orig_cond] * t.train_batch_size)
    t.targ_cond = torch.cat([t.targ_cond] * t.train_batch_size)

    if t.orig_vector is not None:
        t.orig_vector = torch.cat([t.orig_vector] * t.train_batch_size)
        t.targ_vector = torch.cat([t.targ_vector] * t.train_batch_size) 

    height, width = t.image_size

    latents = torch.randn((t.train_batch_size, 16 if t.is_zimage else 4, height // 8, width // 8), device=CUDA,dtype = t.train_model_precision)
    if t.is_zimage:
        latents = latents.unsqueeze(2)
        latents = list(latents.unbind(0))
        
    loss_ema = None
    loss_velocity = None
    
    torch.squeeze

    pbar = tqdm(range(t.train_iterations))
    while t.train_iterations >= pbar.n:
        with torch.no_grad(), t.a.autocast():       
            if t.is_zimage:
                timesteps, noisy_latents, target = apply_flow_matching_noise(latents, None, 3.0)         
            else:
                timesteps = torch.randint(t.train_min_timesteps, t.train_max_timesteps, (t.train_batch_size,),device=CUDA)
                timesteps = timesteps.long()
            added_cond_kwargs = get_added_cond_kwargs(t, t.targ_vector, t.train_batch_size)
            if t.is_zimage:
                targ_pred, _ = t.unet(latents, timesteps, t.targ_cond)
                targ_pred = -torch.stack(targ_pred,dim=0).squeeze(dim=2)
            else:
                targ_pred = t.unet(latents, timesteps, t.targ_cond, added_cond_kwargs = added_cond_kwargs).sample
        
        added_cond_kwargs = get_added_cond_kwargs(t, t.orig_vector, t.train_batch_size)

        with network, t.a.autocast():
            if t.is_zimage:
                orig_pred, _ = t.unet(latents, timesteps, t.orig_cond)
                orig_pred = -torch.stack(orig_pred,dim=0).squeeze(dim=2)
            else:
                orig_pred = t.unet(latents, timesteps, t.orig_cond, added_cond_kwargs = added_cond_kwargs).sample
        
        loss, loss_ema, loss_velocity = process_loss(t, orig_pred, targ_pred, timesteps, loss_ema, loss_velocity)

        c_lrs = [f"{x:.2e}" for x in lr_scheduler.get_last_lr()]

        pbar.set_description(f"Loss EMA * 1000: {loss_ema * 1000:.4f}, Current LR: "+", ".join(c_lrs))   
        pbar.update(1)
        if t.logging_save_csv:
            savecsv(t, pbar.n, loss_ema, [x.cpu().item() if isinstance(x, torch.Tensor)  else x for x in lr_scheduler.get_last_lr()],t.csvpath)

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
    
    t.vae.to(CUDA)
    t.orig_latent = image2latent(t,t.images[0]).to(t.train_model_precision).repeat_interleave(t.train_batch_size,0)
    t.targ_latent = image2latent(t,t.images[1]).to(t.train_model_precision)

    del t.vae
    if "BASE" not in t.network_blocks:
        del t.text_model
    flush()
    
    t.unet = t.unet.to(CUDA, dtype=t.train_model_precision)
    
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
        if t.is_zimage:
            timesteps, noisy_latents, target = apply_flow_matching_noise(image_latent, noise, 3.0)         
        else:
            timesteps = torch.randint(t.train_min_timesteps, t.train_max_timesteps, ((1 if t.train_fixed_timsteps_in_batch else batch_size),),device=CUDA) 
            timesteps = torch.cat([timesteps.long()] * (batch_size if t.train_fixed_timsteps_in_batch else 1))
            noisy_latents = t.noise_scheduler.add_noise(image_latent, noise, timesteps)

        with network, t.a.autocast():
            if t.is_zimage:
                image_latent = image_latent.unsqueeze(2)
                image_latent = list(image_latent.unbind(0))
                noise_pred = t.unet(noisy_latents, timesteps, torch.cat([t.orig_cond] * batch_size)).sample
                noise_pred = -torch.stack(noise_pred,dim=0).squeeze(dim=2)
            else:
                noise_pred = t.unet(noisy_latents, timesteps, torch.cat([t.orig_cond] * batch_size), added_cond_kwargs = added_cond_kwargs).sample

        if not t.is_zimage:
            target = noise
        if t.model_v_pred:
            target = t.noise_scheduler.get_velocity(image_latent, noise, timesteps)

        loss, loss_ema, loss_velocity = process_loss(t, noise_pred, target, timesteps, loss_ema, loss_velocity, copy)

        c_lrs = [f"{x:.2e}" for x in lr_scheduler.get_last_lr()]

        pbar.set_description(f"Loss EMA * 1000: {loss_ema * 1000:.4f}, Loss Velosity: {loss_velocity * 1000:.4f}, Current LR: "+", ".join(c_lrs))   
        pbar.update(1)
        if t.logging_save_csv:
            savecsv(t, pbar.n, loss_ema, [x.cpu().item() if isinstance(x, torch.Tensor)  else x for x in lr_scheduler.get_last_lr()],t.csvpath, copy = copy)

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


def train_diff2(t):
    global stoptimer
    stoptimer = 0

    t.vae.to(CUDA)
    if t.mode == "ADDifT":
        t.orig_latent = image2latent(t,t.images[0]).to(t.train_model_precision)
        t.targ_latent = image2latent(t,t.images[1]).to(t.train_model_precision)
        data = dataset.LatentsConds(t, [([t.orig_latent, None, t.orig_cond, t.targ_vector],[t.targ_latent,None, t.targ_cond, t.targ_vector])])
        dataloaders = [dataset.DataLoader(data, batch_size=t.train_batch_size, shuffle=True)]
    else:
        t.a.print("Preparing image latents and text-conditional...")
        dataloaders = dataset.make_dataloaders(t)
    t.dataloader = dataset.ContinualRandomDataLoader(dataloaders)
    t.dataloader = (t.a.prepare(t.dataloader))

    t.a.print("Train ADDifT Start")

    if not t.dataloader.data:
        return "No data!"

    del t.vae
    if "BASE" not in t.network_blocks:
        del t.text_model
    flush()
    t.unet = t.unet.to(CUDA, dtype=t.train_model_precision)
  
    network, optimizer, lr_scheduler = create_network(t)

    loss_ema = None
    noise = None
    loss_velocity = None

    pbar = tqdm(range(t.train_iterations))
    epoch = 0
    time_min = t.train_min_timesteps
    time_max = t.train_max_timesteps
    
    while t.train_iterations >= pbar.n + 1:
        for batch in t.dataloader:
            orig_latent = batch["orig_latent"]
            targ_latent = batch["targ_latent"]

            batch_size = orig_latent.shape[0]
 
            orig_conds1 = batch["orig_conds1"] if "orig_conds1" in batch else torch.cat([t.orig_cond] * batch_size)
            orig_conds2 = batch["orig_conds2"] if "orig_conds2" in batch else (torch.cat([t.orig_vector] * batch_size) if isinstance(t.orig_vector, torch.Tensor) else None)
            targ_conds1 = batch["targ_conds1"] if "targ_conds1" in batch else torch.cat([t.targ_cond] * batch_size)
            targ_conds2 = batch["targ_conds2"] if "targ_conds2" in batch else (torch.cat([t.targ_vector] * batch_size) if isinstance(t.targ_vector, torch.Tensor) else None)
            
            orig_added_cond_kwargs = get_added_cond_kwargs(t, orig_conds2, batch_size, size = [*orig_latent.shape[2:4]]) if orig_conds2 is not None else None 
            targ_added_cond_kwargs = get_added_cond_kwargs(t, targ_conds2, batch_size, size = [*targ_latent.shape[2:4]]) if targ_conds2 is not None else None 

            optimizer.zero_grad()
            noise = torch.randn_like(orig_latent) 

            turn = pbar.n % 2 == 0 

            #t.train_min_timesteps = int(500 * (pbar.n * 2 / t.train_iterations)) if 1 > pbar.n * 2 / t.train_iterations else 500
            #t.train_max_timesteps = int(750 + 250 * (1 - (pbar.n * 2/ (t.train_iterations * 2)))) if 1 > pbar.n * 2 / t.train_iterations else 750
            #print(t.train_min_timesteps, t.train_max_timesteps)
            if turn:
                span = (t.train_max_timesteps - t.train_min_timesteps) /10
                index = pbar.n % 10 + 1
                time_min = t.train_min_timesteps + span * index
                time_max = t.train_min_timesteps + span * (index + 1)

                if not (time_min > time_max):
                    time_max = time_min + 1

            timesteps = torch.randint(int(999 if time_min > 999 else time_min), int(time_max if 1000 > time_max else 1000), ((1 if t.train_fixed_timsteps_in_batch else batch_size),),device=CUDA) 
            timesteps = torch.cat([timesteps.long()] * (batch_size if t.train_fixed_timsteps_in_batch else 1))

            if 0 > t.diff_alt_ratio and not turn:
                targ_latent = orig_latent = noise

            if t.is_zimage:
                timesteps_z, orig_noisy_latents, target = apply_flow_matching_noise(orig_latent if turn else targ_latent, noise, 3.0, timesteps = timesteps)    
                timesteps_z, targ_noisy_latents, target = apply_flow_matching_noise(targ_latent if turn else orig_latent, noise, 3.0, timesteps = timesteps)        
            else:
                orig_noisy_latents = t.noise_scheduler.add_noise(orig_latent if turn else targ_latent, noise, timesteps)
                targ_noisy_latents = t.noise_scheduler.add_noise(targ_latent if turn else orig_latent, noise, timesteps)
                
            if t.is_zimage:
                orig_noisy_latents = orig_noisy_latents.unsqueeze(2)
                targ_noisy_latents = targ_noisy_latents.unsqueeze(2)
                orig_noisy_latents = list(orig_noisy_latents.unbind(0))
                targ_noisy_latents = list(targ_noisy_latents.unbind(0))

            with torch.no_grad(), t.a.autocast(): 
                if t.is_zimage:
                    orig_noise_pred, _ = t.unet(orig_noisy_latents, timesteps_z, orig_conds1)
                    orig_noise_pred = -torch.stack(orig_noise_pred,dim=0).squeeze(dim=2)
                    print(orig_noise_pred.shape)
                else:
                    orig_noise_pred = t.unet(orig_noisy_latents, timesteps, orig_conds1, added_cond_kwargs = orig_added_cond_kwargs).sample
            
            network.set_multiplier(0.25 if turn else - 0.25 * abs(t.diff_alt_ratio))

            with t.a.autocast():
                if t.is_zimage:
                    targ_noise_pred, _ = t.unet(targ_noisy_latents, timesteps_z, targ_conds1)
                    targ_noise_pred = -torch.stack(targ_noise_pred,dim=0).squeeze(dim=2)
                else:
                    targ_noise_pred = t.unet(targ_noisy_latents, timesteps, targ_conds1, added_cond_kwargs = targ_added_cond_kwargs).sample
            network.set_multiplier(0)

            if t.diff_use_diff_mask and "mask" in batch:
                targ_noise_pred = targ_noise_pred * batch["mask"].to(CUDA) 
                orig_noise_pred = orig_noise_pred * batch["mask"].to(CUDA)   

            loss, loss_ema, loss_velocity = process_loss(t, targ_noise_pred, orig_noise_pred, timesteps, loss_ema, loss_velocity)

            c_lrs = [f"{x:.2e}" for x in lr_scheduler.get_last_lr()]

            pbar.set_description(f"Loss EMA * 1000: {loss_ema * 1000:.4f}, Loss Velosity: {loss_velocity * 1000:.4f}, Current LR: "+", ".join(c_lrs) + f", Epoch: {epoch}")   
            pbar.update(1)
            if t.logging_save_csv:
                savecsv(t, pbar.n, loss_ema, [x.cpu().item() if isinstance(x, torch.Tensor)  else x for x in lr_scheduler.get_last_lr()],t.csvpath)
            
            optimizer.zero_grad()
            t.a.backward(loss)
            optimizer.step()
            lr_scheduler.step()

            flush()
            
            result = finisher(network, t, pbar.n)
            if result is not None:
                return result

        epoch += 1
        
    result = savecount(network, t, 0)
    return result

#### Prepare LoRA, Optimizer, lr_scheduler, Save###############################################
def flush():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

def create_network(t):
    network = load_network(t)
    optimizer= trainer.get_optimizer(t.train_optimizer, network.prepare_optimizer_params(), t.train_learning_rate, t.train_optimizer_settings, network)

    t.is_schedulefree = t.train_optimizer.endswith("schedulefree".lower())

    if t.is_schedulefree:
        optimizer.train()
    else:
        lr_scheduler = trainer.load_lr_scheduler(t, optimizer)

    print(f"Optimizer : {type(optimizer).__name__}")
    print(f"Optimizer Settings : {t.train_optimizer_settings}")

    network, optimizer, lr_scheduler = t.a.prepare(network, optimizer, None if t.is_schedulefree else lr_scheduler)

    return network, optimizer, DummyScheduler(optimizer) if t.is_schedulefree else lr_scheduler

class DummyScheduler():
    def __init__(self, optimizer):
        self.optimizer = optimizer
        pass

    def get_last_lr(self):
        return [p["scheduled_lr"] for p in self.optimizer.param_groups]

    def step(self):
        pass

def load_network(t):
    types = trainer.all_configs[get_name_index("network_type")][2]
    if t.network_type in types[:2]:
        return LoRANetwork(t).to(CUDA, dtype=t.train_lora_precision)
    else:
        return LycorisNetwork(t).to(CUDA, dtype=t.train_lora_precision)

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
    filename = network.save_weights(filename, t, metaname)
    return f"Successfully created to {filename}"

def makesavelist(t):
    if t.save_per_steps > 0:
        t.save_list = [x * t.save_per_steps for x in range(1, t.train_iterations // t.save_per_steps + 1)]
        if t.train_iterations in t.save_list:
            t.save_list.remove(t.train_iterations)
    else:
        t.save_list = []

def process_loss(t, original, target, timesteps, loss_ema, loss_velocity, copy = False):
    if t.train_loss_function == "MSE":
        loss = torch.nn.functional.mse_loss(original.float(), target.float(), reduction="none")
    elif t.train_loss_function == "L1":
        loss = torch.nn.functional.l1_loss(original.float(), target.float(), reduction="none")
    elif t.train_loss_function == "Smooth-L1":
        loss = torch.nn.functional.smooth_l1_loss(original.float(), target.float(), reduction="none")
    
    loss = loss.mean([1, 2, 3])

    if t.train_snr_gamma > 0 and not t.is_zimage:
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
            image = img.convert("RGB")
    image = numpy.array(image)
    image = image.astype(numpy.float32) / 255.0
    image = numpy.moveaxis(image, 2, 0)
    image = torch.from_numpy(image).unsqueeze(0)
    image = image * 2 - 1
    image = image.to(CUDA,dtype=t.train_VAE_precision)
    with torch.no_grad():
        t.vae.to(t.train_VAE_precision)
        latent = t.vae.encode(image)
        if isinstance(latent, torch.Tensor):
            return ((latent - t.vae_shift_factor) * t.vae_scale_factor)
        else:
            return ((latent.latent_dist.sample() - t.vae_shift_factor) * t.vae_scale_factor)

def text2cond(t, prompt):
    if not standalone:
        input = SdConditioning([prompt], width=t.image_size[0], height=t.image_size[1])
        cond = prompt_parser.get_learned_conditioning(shared.sd_model,input,1)
        if t.is_sdxl:
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

def apply_flow_matching_noise(latents, noise, shift=3.0, timesteps=None):
    """ return timesteps, noisy_latents, target """
    batch_size = latents.shape[0]
    device = latents.device
    dtype = latents.dtype

    if timesteps is None:
        u = torch.randn(batch_size, device=device)
        t_01 = torch.sigmoid(u)
        sigmas = (t_01 * shift) / (1 + (shift - 1) * t_01)
        timesteps = sigmas * 1000.0
    else:
        timesteps = timesteps.to(device)
        sigmas = timesteps.float() / 1000.0
        if sigmas.ndim == 0:
            sigmas = sigmas.expand(batch_size)
    if noise is None:
        return timesteps, None, None

    view_shape = [batch_size] + [1] * (latents.ndim - 1)
    sigmas_view = sigmas.view(view_shape)

    noisy_latents = (1.0 - sigmas_view) * latents + sigmas_view * noise
    noisy_latents = noisy_latents.to(dtype)
    target = noise - latents

    return timesteps, noisy_latents, target

#### Debug, Logging ####################################################
def check_requires_grad(model: torch.nn.Module):
    for name, module in list(model.named_modules())[:5] + list(model.named_modules())[5:]:
        if len(list(module.parameters())) > 0:
            print(f"Module: {name}")
            for name, param in list(module.named_parameters())[:2]:
                print(f"    Parameter: {name}, Requires Grad: {param.requires_grad}")

def check_training_mode(model: torch.nn.Module):
    for name, module in list(model.named_modules())[:5] + list(model.named_modules())[5:]:
        print(f"Module: {name}, Training Mode: {module.training}")

CSVHEADS = ["network_rank", "network_alpha", "train_learning_rate", "train_iterations", "train_lr_scheduler", "model_version", "train_optimizer", "save_lora_name"]

def savecsv(t, step, loss, lr, csvpath, copy=False):
    header = []
    for key in CSVHEADS:
        header.append([key, getattr(t, key, "")])

    header.append(["Step", "Loss"] + ["Learning Rate " + str(i+1) for i in range(len(lr))])

    if copy:
        csvpath = csvpath.replace(".csv", "_copy.csv")
    
    directory = os.path.dirname(csvpath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    file_exists = os.path.isfile(csvpath)

    with open(csvpath, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            for head in header:
                writer.writerow(head)
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
        "ss_v2": bool(t.is_sd2),
        "ss_base_model_version": t.model_version,
        "ss_seed": t.train_seed,
        "ss_optimizer": t.train_optimizer,
        "ss_min_snr_gamma": t.train_snr_gamma,
        "ss_tag_frequency": json.dumps({1:t.count_dict})
    }
    
#### StandAlone ####################################################
def detect_model_version(state_dict):
    flux_test_key = "model.diffusion_model.double_blocks.0.img_attn.norm.key_norm.scale"
    sd3_test_key = "model.diffusion_model.final_layer.adaLN_modulation.1.bias"
    legacy_test_key = "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight"
    zimage_key1 = "context_refiner.0.attention.k_norm.weight"
    zimage_key2 = "model.diffusion_model.context_refiner.0.attention.k_norm.weight"

    if legacy_test_key in state_dict:
        match state_dict[legacy_test_key].shape[1]:
            case 768:
                return 0
            case 1024:
                return 1
            case 1280:
                return 6     # sdxl refiner model
            case 2048:
                return 2
    elif flux_test_key in state_dict:
        return 4
    elif sd3_test_key in state_dict:
        return 3
    elif zimage_key1 in state_dict or zimage_key2 in state_dict:
        return 5
    else:
        return -1
