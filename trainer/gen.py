import random
from modules import shared, processing, images, sd_samplers
from modules.ui import  plaintext_to_html
from modules.shared import opts
from modules.processing import create_infotext,Processed
from modules.generation_parameters_copypaste import create_override_settings_dict

paramsnames = None

def gen_both(orig, targ, negative, *txt2imgparams):
    orig = setup_gen_p(False, orig, negative, *txt2imgparams)
    targ = setup_gen_p(False, targ, negative, *txt2imgparams, s_seed = orig[5])
    return orig[:4] + targ[:4]

def setup_gen_p(train, prompt, negative_prompt, *txt2imgparams, s_seed = None):
    #print(txt2imgparams)
    #print(paramsnames)
    #[None, 'Prompt', 'Negative prompt', 'Styles', 'Sampling steps', 'Sampling method', 'Batch count', 'Batch size', 'CFG Scale', 
    # 'Height', 'Width', 'Hires. fix', 'Denoising strength', 'Upscale by', 'Upscaler', 'Hires steps', 'Resize width to', 'Resize height to', 
    # 'Hires checkpoint', 'Hires sampling method', 'Hires prompt', 'Hires negative prompt', 'Override settings', 'Script', 'Refiner', 
    # 'Checkpoint', 'Switch at', 'Seed', 'Extra', 'Variation seed', 'Variation strength', 'Resize seed from width', 'Resize seed from height', '', 'Active', 'Active', 'X Types', 'X Values', 'Y Types', 'Y Values']  

    def g(wanted,wantedv=None):
        if wanted in paramsnames:return txt2imgparams[paramsnames.index(wanted)]
        elif wantedv and wantedv in paramsnames:return txt2imgparams[paramsnames.index(wantedv)]
        else:return None

    sampler_index = g("Sampling method")
    if type(sampler_index) is str:
        sampler_name = sampler_index
    else:       
        sampler_name = sd_samplers.samplers[sampler_index].name

    hr_sampler_index = g("Hires sampling method")
    if hr_sampler_index is None: hr_sampler_index = 0
    if type(sampler_index) is str:
        hr_sampler_name = hr_sampler_index
    else:       
        hr_sampler_name = "Use same sampler" if hr_sampler_index == 0 else  sd_samplers.samplers[hr_sampler_index+1].name

    if g("Seed") == -1 and s_seed is None:
        s_seed = random.randrange(4294967294)

    p = processing.StableDiffusionProcessingTxt2Img(
        sd_model=shared.sd_model,
        outpath_samples=opts.outdir_samples or opts.outdir_txt2img_samples,
        outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=g("Styles"),
        negative_prompt=negative_prompt,
        seed=g("Seed") if s_seed is None else s_seed,
        subseed=g("Variation seed"),
        subseed_strength=g("Variation strength"),
        seed_resize_from_h=g("Resize seed from height"),
        seed_resize_from_w=g("Resize seed from width"),
        seed_enable_extras=g("Extra"),
        sampler_name=sampler_name,
        batch_size=g("Batch size"),
        n_iter=g("Batch count"),
        steps=g("Sampling steps"),
        cfg_scale=g("CFG Scale"),
        width=g("Width"),
        height=g("Height"),
        restore_faces=g("Restore faces","Face restore"),
        tiling=g("Tiling"),
        enable_hr=g("Hires. fix","Second pass"),
        hr_scale=g("Upscale by"),
        hr_upscaler=g("Upscaler"),
        hr_second_pass_steps=g("Hires steps","Secondary steps"),
        hr_resize_x=g("Resize width to"),
        hr_resize_y=g("Resize height to"),
        override_settings=create_override_settings_dict(g("Override settings")),
        do_not_save_grid=True,
        do_not_save_samples=True,
        do_not_reload_embeddings=True,
    )
    p.hr_checkpoint_name=None if g("Hires checkpoint") == 'Use same checkpoint' else g("Hires checkpoint")
    p.hr_sampler_name=None if hr_sampler_name == 'Use same sampler' else  hr_sampler_name

    p.cached_c = [None,None]
    p.cached_uc = [None,None]

    p.cached_hr_c = [None, None]
    p.cached_hr_uc = [None, None]

    if type(p.prompt) == list:
        p.all_prompts = [shared.prompt_styles.apply_styles_to_prompt(x, p.styles) for x in p.prompt]
    else:
        p.all_prompts = [shared.prompt_styles.apply_styles_to_prompt(p.prompt, p.styles)]

    if type(p.negative_prompt) == list:
        p.all_negative_prompts = [shared.prompt_styles.apply_negative_styles_to_prompt(x, p.styles) for x in p.negative_prompt]
    else:
        p.all_negative_prompts = [shared.prompt_styles.apply_negative_styles_to_prompt(p.negative_prompt, p.styles)]

    if train:
        return p
    
    processed:Processed = processing.process_images(p)

    infotext = create_infotext(p, p.all_prompts, p.all_seeds, p.all_subseeds)
    if infotext.count("Steps: ")>1:
        infotext = infotext[:infotext.rindex("Steps")]

    for i, image in enumerate(processed.images):
        images.save_image(image, opts.outdir_txt2img_samples, "",p.seed, p.prompt,shared.opts.samples_format, p=p,info=infotext)

    return processed.images,infotext,plaintext_to_html(processed.info), plaintext_to_html(processed.comments),p, s_seed