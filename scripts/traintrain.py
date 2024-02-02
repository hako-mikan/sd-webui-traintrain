from cProfile import label
import os
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image, ImageChops
import random
import numpy as np
from modules import scripts, script_callbacks, sd_models, sd_vae
from modules.shared import opts
from modules.ui import create_output_panel, create_refresh_button
from trainer import train, trainer, gen

jsonspath = trainer.jsonspath
logspath = trainer.logspath
presetspath = trainer.presetspath

MODES = ["LoRA", "iLECO", "Difference"]

BLOCKID26=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
BLOCKID17=["BASE","IN01","IN02","IN04","IN05","IN07","IN08","M00","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]
BLOCKID12=["BASE","IN04","IN05","IN07","IN08","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05"]
BLOCKID20=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08"]

PRECISION_TYPES = ["fp32", "bf16", "fp16", "float32", "bfloat16", "float16"]
NETWORK_TYPES = ["lierla", "c3lier","loha"]
NETWORK_DIMS = [str(2**x) for x in range(10)]
NETWORK_ELEMENTS = ["Full", "CrossAttention", "SelfAttention"]
IMAGESTEPS = [str(x*64) for x in range(10)]
OPTIMIZERS = ["adamw", "adamw8bit","adafactor","lion", "prodigy", "dadaptadam","dadaptlion","adam8bit","adam",]
LOSS_REDUCTIONS = ["none", "mean"]

SCHEDULERS = ["linear", "cosine", "cosine_with_restarts" ,"polynomial", "constant", "constant_with_warmup" ,"piecewise_constant"]
#NOISE_SCHEDULERS = ["Euler A", "DDIM", "DDPM", "LMSD"]
NOISE_SCHEDULERS = ["DDIM", "DDPM", "PNDM", "LMS", "Euler", "Euler a", "DPMSolver", "DPMsingle", "Heun", "DPM 2", "DPM 2 a"]
TARGET_MODULES = ["Both", "U-Net", "Text Encoder"]

ALL = [True,True,True,True]
LORA = [True,False,False,False]
ILECO = [False,True,False,False]
NDIFF =  [True,True,False,False]
DIFF = [False,False,True,True]
DIFF1 = [False,False,True,False]
DIFF2 = [False,False,False,True]
NDIFF2 = [True,True,True,False]

#requiered parameters
use_2nd_pass_settings = ["use_2nd_pass_settings", "CH", None, False, bool, DIFF2]
lora_data_directory = ["lora_data_directory","TX",None,"", str, LORA]
lora_trigger_word = ["lora_trigger_word","TX",None,"", str, LORA]
network_type = ["network_type","DD",NETWORK_TYPES,NETWORK_TYPES[0],str,ALL]
network_rank = ["network_rank","DD",NETWORK_DIMS[2:],"16",int,ALL]
network_alpha = ["network_alpha","DD",NETWORK_DIMS,"8",int,ALL]
network_element = ["network_element","DD",NETWORK_ELEMENTS,None,str,ALL]
image_size = ["image_size(height, width)", "TX",None,512,str,NDIFF]
train_iterations = ["train_iterations","TX",None,1000,int,ALL]
train_batch_size = ["train_batch_size", "TX",None,2,int,ALL]
train_learning_rate = ["train_learning_rate","TX",None,"1e-4",float,ALL]
train_optimizer =["train_optimizer","DD",OPTIMIZERS,"adamw",str,ALL]
train_lr_scheduler =["train_lr_scheduler","DD",SCHEDULERS, "cosine",str,ALL]
save_lora_name =  ["save_lora_name", "TX",None,"",str,NDIFF2]
use_gradient_checkpointing = ["use_gradient_checkpointing","CH",None,False,bool,ALL]

#option parameters
network_conv_rank = ["network_conv_rank","DD",["0"] + NETWORK_DIMS[2:],"0",int,ALL]
network_conv_alpha = ["network_conv_alpha","DD",["0"] + NETWORK_DIMS,"0",int,ALL]
train_seed = ["train_seed", "TX",None,-1,int, ALL]
train_textencoder_learning_rate = ["train_textencoder_learning_rate","TX",None,"",float,LORA]
train_model_precision = ["train_model_precision","DD",PRECISION_TYPES[:3],"fp16",str,ALL]
train_lora_precision = ["train_lora_precision","DD",PRECISION_TYPES[:3],"fp32",str,ALL]
image_buckets_step = ["image_buckets_step", "DD",IMAGESTEPS,"256",int,LORA]
image_min_length = ["image_min_length", "TX",None,512,int,LORA]
image_max_ratio = ["image_max_ratio", "TX",None,2,float,LORA]
sub_image_num = ["sub_image_num", "TX",None,0,int,LORA]
image_mirroring =  ["image_mirroring", "CH",None,False,bool,LORA]
image_use_filename_as_tag =  ["image_use_filename_as_tag", "CH",None,False,bool,LORA]
image_disable_upscale = ["image_disable_upscale", "CH",None,False,bool,LORA]
save_per_steps = ["save_per_steps", "TX",None,0,int,ALL]
save_precision = ["save_precision","DD",PRECISION_TYPES[:3],"fp16",str,ALL]
save_overwrite = ["save_overwrite", "CH",None,False,bool,ALL]
save_as_json = ["save_as_json", "CH",None,False,bool,NDIFF2]

diff_save_1st_pass = ["diff_save_1st_pass", "CH",None,False,bool,DIFF1]
diff_1st_pass_only = ["diff_1st_pass_only", "CH",None,False,bool,DIFF1]
diff_load_1st_pass = ["diff_load_1st_pass","TX", None, "", str, DIFF1]
train_lr_step_rules = ["train_lr_step_rules","TX",None,"",str,ALL]
train_lr_warmup_steps = ["train_lr_warmup_steps","TX",None,0,int,ALL]
train_lr_scheduler_num_cycles = ["train_lr_scheduler_num_cycles","TX",None,1,int,ALL]
train_lr_scheduler_power = ["train_lr_scheduler_power","TX",None, 1.0, float,ALL]
train_snr_gamma = ["train_snr_gamma","TX",None,5,float,ALL]
train_fixed_timsteps_in_batch = ["train_fixed_timsteps_in_batch","CH",None,False,bool,ALL]
image_use_transparent_background_ajust = ["image_use_transparent_background_ajust","CH",None,False,bool,ALL]

logging_verbose = ["logging_verbose","CH",None,False,bool,NDIFF2]
logging_save_csv = ["logging_save_csv","CH",False,"",bool,NDIFF2]
model_v_pred = ["model_v_pred", "CH",None,False,bool,ALL]

network_blocks = ["network_blocks(BASE = TextEncoder)","CB",BLOCKID26,BLOCKID26,list,ALL]

#unuased parameters
logging_use_wandb = ["logging_use_wandb","CH",None,False,bool]
train_repeat = ["train_repeat","TX",None, 1.0, int,ALL]
train_use_bucket = ["train_use_bucket", "CH",None,False,bool]
train_optimizer_args = ["train_optimizer_args","TX",None,"",str]
logging_dir = ["logging_dir","TX",None,"",str,NDIFF2]
gradient_accumulation_steps = ["gradient_accumulation_steps","TX",None,"1",str,ALL]
gen_noise_scheduler = ["gen_noise_scheduler", "DD",NOISE_SCHEDULERS,NOISE_SCHEDULERS[6],str,NDIFF2]
lora_train_targets = ["lora_train_targets","RD",TARGET_MODULES,TARGET_MODULES[0], str, LORA]
logging_use_tensorboard = ["logging_use_tensorboard","CH",False,"",bool,NDIFF2]
train_min_timesteps = ["train_min_timesteps", "TX",None,0,int,ALL]
train_max_timesteps = ["train_max_timesteps", "TX",None,1000,int,ALL]

r_column1 = [network_type,network_rank,network_alpha,lora_data_directory,lora_trigger_word]
r_column2 = [image_size ,train_iterations,train_batch_size ,train_learning_rate]
r_column3 = [train_optimizer,train_lr_scheduler, save_lora_name,use_gradient_checkpointing]
row1 = [network_blocks]

o_column1 = [network_conv_rank,network_conv_alpha,network_element,image_buckets_step,
                     image_min_length,image_max_ratio,sub_image_num,image_mirroring,
                     image_use_filename_as_tag,image_disable_upscale,image_use_transparent_background_ajust,train_fixed_timsteps_in_batch]
o_column2 = [train_textencoder_learning_rate,train_seed,train_lr_step_rules, train_lr_warmup_steps, train_lr_scheduler_num_cycles,train_lr_scheduler_power, 
                     train_snr_gamma, save_per_steps]
o_column3 = [train_model_precision, train_lora_precision,save_precision,diff_load_1st_pass, diff_save_1st_pass,diff_1st_pass_only,
                    logging_save_csv,logging_verbose,save_overwrite, save_as_json,model_v_pred]

trainer.all_configs = r_column1 + r_column2 + r_column3 + row1 + o_column1 + o_column2 + o_column3 + [use_2nd_pass_settings]

def makeui(sets, pas = 0):
    output = []
    add_id = "2_" if pas > 0 else "1_"
    for name, uitype, choices, value, _, visible in sets:
        visible = visible[pas]
        with gr.Row():
            if uitype == "DD":
                output.append(gr.Dropdown(label=name.replace("_"," "), choices=choices, value=value if value else choices[0] , elem_id="tt_" + name, visible = visible))
            if uitype == "TX":
                output.append(gr.Textbox(label=name.replace("_"," "),value = value, elem_id="tt_" +add_id + name, visible = visible))
            if uitype == "CH":
                output.append(gr.Checkbox(label=name.replace("_"," "),value = value, elem_id="tt_" + name, visible = visible))
            if uitype == "CB":
                output.append(gr.CheckboxGroup(label=name.replace("_"," "),choices=choices, value = value, elem_id="tt_" + name, type="value", visible = visible))
            if uitype == "RD":
                output.append(gr.Radio(label=name.replace("_"," "),choices=[x + " " for x in choices] if pas > 0 else choices, value = value, elem_id="tt_" + name,visible = visible))
    return output

txt2img_gen_button = None
img2img_gen_button = None
paramsnames = []
txt2img_params = []
img2img_params = []

button_o_gen = None
button_t_gen = None
button_b_gen = None

prompts = None
imagegal_orig = None
imagegal_targ = None

class ToolButton(gr.Button, gr.components.FormComponent):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, **kwargs):
        super().__init__(variant="tool",
                         elem_classes=kwargs.pop('elem_classes', []),
                         **kwargs)

    def get_block_name(self):
        return "button"

def on_ui_tabs():
    global imagegal_orig, imagegal_targ, prompts, result
    global button_o_gen, button_t_gen, button_b_gen

    def load_preset(name):
        json_files = [f.replace(".json","") for f in os.listdir(presetspath) if f.endswith('.json')]
        if name is None:
            return json_files
        else:
            return trainer.import_json(name, preset = True)

    folder_symbol = '\U0001f4c2'   
    load_symbol = '\u2199\ufe0f'   # ↙
    save_symbol = '\U0001f4be'     # 💾
    refresh_symbol = '\U0001f504'  # 🔄

    with gr.Blocks() as ui:
        with gr.Tab("Train"):
            result = gr.Textbox(label="Message")
            with gr.Row():
                start= gr.Button(value="Start Training",elem_classes=["compact_button"],variant='primary')
                stop= gr.Button(value="Stop",elem_classes=["compact_button"],variant='primary')
                stop_save= gr.Button(value="Stop and Save",elem_classes=["compact_button"],variant='primary')
            with gr.Row():
                with gr.Column():
                    queue = gr.Button(value="Add to Queue", elem_classes=["compact_button"],variant='primary')
                with gr.Column():
                    with gr.Row():
                        presets = gr.Dropdown(choices=load_preset(None), show_label=False, elem_id="tt_preset")
                        loadpreset = ToolButton(value=load_symbol)
                        savepreset = ToolButton(value=save_symbol)
                        refleshpreset = ToolButton(value=refresh_symbol)
                with gr.Column():
                    with gr.Row():
                        sets_file = gr.Textbox(show_label=False)
                        openfolder = ToolButton(value=folder_symbol)
                        loadjson = ToolButton(value=load_symbol)
            with gr.Row(equal_height=True):
                with gr.Column():
                    mode = gr.Radio(label="Mode", choices= MODES, value = "LoRA")
                with gr.Column():
                    with gr.Row(equal_height=True):
                        model = gr.Dropdown(sd_models.checkpoint_tiles(),elem_id="model_converter_model_name",label="Model",interactive=True)
                        create_refresh_button(model, sd_models.list_models,lambda: {"choices": sd_models.checkpoint_tiles()},"refresh_checkpoint_Z")
                with gr.Column():
                    with gr.Row(equal_height=True):
                        vae = gr.Dropdown(choices=["None"] + list(sd_vae.vae_dict), value="None", label="VAE", elem_id="modelmerger_bake_in_vae")
                        create_refresh_button(vae, sd_vae.refresh_vae_list, lambda: {"choices": ["None"] + list(sd_vae.vae_dict)}, "modelmerger_refresh_bake_in_vae")

            dummy = gr.Checkbox(visible=False, value = False)

            gr.HTML(value="Required Parameters(+prompt in iLECO, +images in Difference)")
            with gr.Row():
                with gr.Column(variant="compact"):
                    col1_r1 = makeui(r_column1)
                with gr.Column(variant="compact"):
                    col2_r1 = makeui(r_column2)
                with gr.Column(variant="compact"):
                    col3_r1 = makeui(r_column3)
            with gr.Row():
                blocks_grs_1 = makeui(row1)

            gr.HTML(value="Option Parameters")
            with gr.Row():
                with gr.Column(variant="compact"):
                    col1_o1 = makeui(o_column1)
                with gr.Column(variant="compact"):
                    col2_o1 = makeui(o_column2)
                with gr.Column(variant="compact"):
                    col3_o1 = makeui(o_column3)

                train_settings_1 = col1_r1 + col2_r1 + col3_r1 + blocks_grs_1 + col1_o1 + col2_o1 + col3_o1 + [dummy]

            with gr.Accordion("2nd pass", open= False, visible = False) as diff_2nd:
                with gr.Row():
                    use_2nd = makeui([use_2nd_pass_settings], 3)
                    copy = gr.Button(value= "Copy settings from 1st pass")
                gr.HTML(value="Required Parameters")
                with gr.Row():  
                    with gr.Column(variant="compact"):
                        col1_r2 = makeui(r_column1, 3)
                    with gr.Column(variant="compact"):
                        col2_r2 = makeui(r_column2, 3)
                    with gr.Column(variant="compact"):
                        col3_r2 = makeui(r_column3, 3)
                with gr.Row():
                    blocks_grs_2 = makeui(row1)

                gr.HTML(value="Option Parameters")
                with gr.Row():
                    with gr.Column(variant="compact"):
                        col1_o2 = makeui(o_column1, 3)
                    with gr.Column(variant="compact"):
                        col2_o2 = makeui(o_column2, 3)
                    with gr.Column(variant="compact"):
                        col3_o2= makeui(o_column3, 3)

                        
                train_settings_2 = col1_r2 + col2_r2 + col3_r2 + blocks_grs_2 + col1_o2 + col2_o2 + col3_o2 + use_2nd

            with gr.Group(visible=False) as g_leco:
                with gr.Row():
                    orig_prompt = gr.TextArea(label="Original Prompt",lines=3)
                with gr.Row():
                    targ_prompt = gr.TextArea(label="Target Prompt",lines=3)
                with gr.Row():
                    neg_prompt = gr.TextArea(label="Negative Prompt(not userd in training)",lines=3)
                with gr.Row():
                    button_o_gen = gr.Button(value="Generate Original",elem_classes=["compact_button"],variant='primary')
                    button_t_gen = gr.Button(value="Generate Target",elem_classes=["compact_button"],variant='primary')
                    button_b_gen = gr.Button(value="Generate All",elem_classes=["compact_button"],variant='primary')
                with gr.Row():
                    with gr.Column():
                        o_g =  create_output_panel("txt2img", opts.outdir_txt2img_samples)
                        imagegal_orig = [x for x in o_g] if isinstance(o_g, tuple) else [o_g.gallery, o_g.generation_info, o_g.infotext, o_g.html_log]
                    with gr.Column():
                        t_g =  create_output_panel("txt2img", opts.outdir_txt2img_samples)
                        imagegal_targ = [x for x in t_g] if isinstance(t_g, tuple) else [t_g.gallery, t_g.generation_info, t_g.infotext, t_g.html_log]

            with gr.Group(visible=False) as g_diff:
                with gr.Row():
                    with gr.Column():
                        orig_image = gr.Image(label="Original Image", interactive=True)
                    with gr.Column():
                        targ_image = gr.Image(label="Target Image", interactive=True)

        with gr.Tab("Queue"):
            with gr.Row():
                reload_queue= gr.Button(value="Reload Queue",elem_classes=["compact_button"],variant='primary')
                delete_queue= gr.Button(value="Delete Queue",elem_classes=["compact_button"],variant='primary')
                delete_name= gr.Textbox(label="Name of Queue to delete")
            with gr.Row():
                queue_list = gr.DataFrame(headers=["Name", "Mode", "Model", "VAE"] + [x[0] for x in trainer.all_configs] * 2 + ["Original prompt", "Target prompt"] )

        with gr.Tab("Plot"):
            with gr.Row():
                reload_plot= gr.Button(value="Reloat Plot",elem_classes=["compact_button"],variant='primary')
                plot_file = gr.Textbox(label="Name of logfile, blank for last or current training")
            with gr.Row():
                plot = gr.Plot()

        with gr.Tab("Image"):
            gr.HTML(value="Rotate random angle and scaling")
            image_result = gr.Textbox(label="Message")
            with gr.Row():
                with gr.Column(variant="compact"):
                    angle_bg= gr.Button(value="From Directory",elem_classes=["compact_button"],variant='primary')
                with gr.Column(variant="compact"):
                    angle_bg_i= gr.Button(value="From File",elem_classes=["compact_button"],variant='primary')
                with gr.Column(variant="compact"):
                    fix_side = gr.Radio(label="fix side", value= "none", choices =["none", "right", "left", "top", "bottom"] )
            with gr.Row():
                with gr.Column(variant="compact"):  
                    image_dir = gr.Textbox(label="Image directory")
                with gr.Column(variant="compact"):
                    output_name = gr.Textbox(label="Output name")
                with gr.Column(variant="compact"):
                    save_dir = gr.Textbox(label="Output directory")                   
            with gr.Row():
                num_of_images = gr.Slider(label="number of images", maximum=1000, minimum=0, step=1, value=5)
                max_tilting_angle = gr.Slider(label="max_tilting_angle", maximum=180, minimum=0, step=1, value=180)
                min_scale = gr.Slider(label="minimun downscale ratio", maximum=1, minimum=0, step=0.01, value=0.4)
            with gr.Row():
                change_angle = gr.Checkbox(label="change angle", value= False)
                change_scale = gr.Checkbox(label="change scale", value= False)

            input_image = gr.Image(label="Input Image", interactive=True, type="pil", image_mode="RGBA")

        dtrue = gr.Checkbox(value = True, visible= False)
        dfalse = gr.Checkbox(value = False, visible= False)

        prompts = [orig_prompt, targ_prompt, neg_prompt]
        in_images = [orig_image, targ_image]

        def savepreset_f(*args):
            train.train(*args)
            return gr.update(choices=load_preset(None))

        angle_bg.click(change_angle_bg,[dtrue, image_dir, save_dir, input_image,output_name, num_of_images ,change_angle,max_tilting_angle, change_scale, min_scale, fix_side], [image_result])
        angle_bg_i.click(change_angle_bg,[dfalse, image_dir, save_dir, input_image,output_name, num_of_images ,change_angle,max_tilting_angle, change_scale, min_scale, fix_side], [image_result])

        start.click(train.train, [dfalse, mode, model, vae, *train_settings_1, *train_settings_2, *prompts, *in_images],[result])
        queue.click(train.queue, [dfalse, mode, model, vae, *train_settings_1, *train_settings_2, *prompts, *in_images],[result])
        savepreset.click(savepreset_f, [dtrue, mode, model, vae, *train_settings_1, *train_settings_2, *prompts, *in_images], [presets])
        refleshpreset.click(lambda : gr.update(choices=load_preset(None)), outputs = [presets])

        reload_queue.click(train.get_del_queue_list, outputs= queue_list)
        delete_queue.click(train.get_del_queue_list, inputs = [delete_name], outputs= queue_list)

        stop.click(train.stop_time,[dfalse],[result])
        stop_save.click(train.stop_time,[dtrue],[result])

        def change_the_mode(mode):
            mode = MODES.index(mode)
            out = [x[5][mode] for x in trainer.all_configs]
            if mode == 1: #LECO
                out.extend([False, True, False])
            elif mode > 1: #Difference
                out.extend([True, False, True])
            else:
                out.extend([False, False, False])
            return [gr.update(visible = x) for x in out]

        def change_the_block(type, select):
            blocks = BLOCKID17 if type == NETWORK_TYPES[0] else BLOCKID26
            return gr.update(choices = blocks, value = [x for x in select if x in blocks])

        def openfolder_f():
            os.startfile(jsonspath)

        loadjson.click(trainer.import_json,[sets_file], [mode, model, vae] +  train_settings_1 +  train_settings_2 + prompts[:2])
        loadpreset.click(load_preset,[presets], [mode, model, vae] +  train_settings_1 +  train_settings_2 + prompts[:2])
        mode.change(change_the_mode,[mode],[*train_settings_1, diff_2nd, g_leco ,g_diff])
        openfolder.click(openfolder_f)
        copy.click(lambda *x: x, train_settings_1[1:], train_settings_2[1:])

        reload_plot.click(plot_csv, [plot_file],[plot])

    return (ui, "TrainTrain", "TrainTrain"),

def plot_csv(csv_path):
    def get_csv(csv_path):
        csv_path = csv_path if ".csv" in csv_path else csv_path + ".csv"
        if csv_path:
            for root, dirs, files in os.walk(logspath):
                if csv_path in files:
                    return os.path.join(root, csv_path)

        # 指定されたファイル名が見つからない場合、または csv_path が空の場合
        # ディレクトリ内で最新の CSV ファイルを探す
        latest_csv = None
        latest_time = 0

        for root, dirs, files in os.walk(logspath):
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    file_time = os.path.getmtime(file_path)

                    if file_time > latest_time:
                        latest_csv = file_path
                        latest_time = file_time

        return latest_csv

    df = pd.read_csv(get_csv(csv_path))
    x = df.columns[0]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 主要な y 軸 (2 列目)
    color = 'tab:red'
    ax1.set_xlabel(x)
    ax1.set_ylabel(df.columns[1], color=color)
    ax1.plot(df[x], df[df.columns[1]], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # 追加の y 軸 (3 列目以降)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Learning Rates', color=color)  # 他の列のラベル
    for column in df.columns[2:]:
        ax2.plot(df[x], df[column], label=column)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Training Result")
    fig.tight_layout()
    plt.legend()
    plt.grid(True)

    return plt.gcf()


# ここに必要な追加の関数を定義します。
def downscale_image(image, min_scale, fix_side=None):
    import random
    from PIL import Image

    scale = random.uniform(min_scale, 1)
    original_size = image.size
    new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
    downscaled_image = image.resize(new_size, Image.ANTIALIAS)
    new_image = Image.new("RGBA", original_size, (0, 0, 0, 0))

    # 配置位置を決定する
    if fix_side == "right":
        x_position = original_size[0] - new_size[0]
        y_position = random.randint(0, original_size[1] - new_size[1])
    elif fix_side == "top":
        x_position = random.randint(0, original_size[0] - new_size[0])
        y_position = 0
    elif fix_side == "left":
        x_position = 0
        y_position = random.randint(0, original_size[1] - new_size[1])
    elif fix_side == "bottom":
        x_position = random.randint(0, original_size[0] - new_size[0])
        y_position = original_size[1] - new_size[1]
    else:
        # fix_sideがNoneまたは無効な値の場合、ランダムな位置に配置
        x_position = random.randint(0, original_size[0] - new_size[0])
        y_position = random.randint(0, original_size[1] - new_size[1])

    new_image.paste(downscaled_image, (x_position, y_position))
    return new_image


MARGIN = 5

def marginer(bbox, image):
    return (
        max(bbox[0] - MARGIN, 0),  # 左
        max(bbox[1] - MARGIN, 0),  # 上
        min(bbox[2] + MARGIN, image.width),  # 右
        min(bbox[3] + MARGIN, image.height)  # 下
    )


def change_angle_bg(from_dir, image_dir, save_dir, input_image, output_name, num_of_images ,
                                change_angle, max_tilting_angle, change_scale, min_scale, fix_side):

    if from_dir:
        image_files = [file for file in os.listdir(image_dir) if file.endswith((".png", ".jpg", ".jpeg"))]
    else:
        image_files = [input_image]

    for file in image_files:
        if isinstance(file, str):
            modified_folder_path = os.path.join(image_dir, "modified")
            os.makedirs(modified_folder_path, exist_ok=True)

            path = os.path.join(image_dir, file)
            name, extention = os.path.splitext(file)
            with Image.open(path) as img:
                img = img.convert("RGBA")
        else:
            modified_folder_path = save_dir
            os.makedirs(modified_folder_path, exist_ok=True)

            img = file
            name = output_name
            extention = "png"

        for i in range(num_of_images):
            modified_img = img

            #画像を回転
            if change_angle:
                angle = random.uniform(-max_tilting_angle, max_tilting_angle)
                modified_img = modified_img.rotate(angle, expand=True)

            if change_scale:
                modified_img = downscale_image(modified_img, min_scale, fix_side)

            # 変更した画像を保存
            save_path = os.path.join(modified_folder_path, f"{name}_id_{i}.{extention}")
            modified_img.save(save_path)

    return f"Images saved in {modified_folder_path}"


BLOCKID=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11","Not Merge"]
BLOCKIDXL=['BASE', 'IN0', 'IN1', 'IN2', 'IN3', 'IN4', 'IN5', 'IN6', 'IN7', 'IN8', 'M', 'OUT0', 'OUT1', 'OUT2', 'OUT3', 'OUT4', 'OUT5', 'OUT6', 'OUT7', 'OUT8', 'VAE']
BLOCKIDXLL=['BASE', 'IN00', 'IN01', 'IN02', 'IN03', 'IN04', 'IN05', 'IN06', 'IN07', 'IN08', 'M00', 'OUT00', 'OUT01', 'OUT02', 'OUT03', 'OUT04', 'OUT05', 'OUT06', 'OUT07', 'OUT08', 'VAE']
ISXLBLOCK=[True, True, True, True, True, True, True, True, True, True, False, False, False, True, True, True, True, True, True, True, True, True, True, False, False, False]

def getjsonlist():
    if not os.path.isdir(jsonspath):
        return []
    json_files = [f for f in os.listdir(jsonspath) if f.endswith('.json')]
    json_files = [f.replace(".json", "") for f in json_files]
    return json_files

class GenParamGetter(scripts.Script):
    events_assigned = False
    def title(self):
        return "TrainTrain Generation Parameter Getter"
    
    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def get_wanted_params(params,wanted):
        output = []
        for target in wanted:
            if target is None:
                output.append(params[0])
                continue
            for param in params:
                if hasattr(param,"label"):
                    if param.label == target:
                        output.append(param)
        return output

    def after_component(self, component: gr.components.Component, **_kwargs):
        """Find generate button"""
        if component.elem_id == "txt2img_generate":
            GenParamGetter.txt2img_gen_button = component
        elif  component.elem_id == "img2img_generate":
            GenParamGetter.img2img_gen_button = component

    def get_components_by_ids(root: gr.Blocks, ids: list[int]):
        components: list[gr.Blocks] = []

        if root._id in ids:
            components.append(root)
            ids = [_id for _id in ids if _id != root._id]
        
        if hasattr(root,"children"):
            for block in root.children:
                components.extend(GenParamGetter.get_components_by_ids(block, ids))
        return components
    
    def compare_components_with_ids(components: list[gr.Blocks], ids: list[int]):
        return len(components) == len(ids) and all(component._id == _id for component, _id in zip(components, ids))

    def get_params_components(demo: gr.Blocks, app):
        global paramsnames, txt2img_params, img2img_params
        for _id, _is_txt2img in zip([GenParamGetter.txt2img_gen_button._id, GenParamGetter.img2img_gen_button._id], [True, False]):
            dependencies: list[dict] = [x for x in demo.dependencies if x["trigger"] == "click" and _id in x["targets"]]
            dependency: dict = None
            cnet_dependency: dict = None
            UiControlNetUnit = None
            for d in dependencies:
                if len(d["outputs"]) == 1:
                    outputs = GenParamGetter.get_components_by_ids(demo, d["outputs"])
                    output = outputs[0]
                    if (
                        isinstance(output, gr.State)
                        and type(output.value).__name__ == "UiControlNetUnit"
                    ):
                        cnet_dependency = d
                        UiControlNetUnit = type(output.value)

                elif len(d["outputs"]) == 4:
                    dependency = d

            params = [params for params in demo.fns if GenParamGetter.compare_components_with_ids(params.inputs, dependency["inputs"])]

            from pprint import pprint

            if _is_txt2img:
                gen.paramsnames = [x.label if hasattr(x,"label") else "None" for x in params[0].inputs]

            if _is_txt2img:
                txt2img_params = params[0].inputs 
            else:
                img2img_params = params[0].inputs

            #pprint(paramsnames)
        if not GenParamGetter.events_assigned:
            with demo:
                button_o_gen.click(
                    fn=gen.setup_gen_p,
                    inputs=[gr.Checkbox(value=False, visible=False), prompts[0], prompts[2], *txt2img_params],
                    outputs=imagegal_orig,
                )

                button_t_gen.click(
                    fn=gen.setup_gen_p,
                    inputs=[gr.Checkbox(value=False, visible=False), prompts[1], prompts[2], *txt2img_params],
                    outputs=imagegal_targ,
                )

                button_b_gen.click(
                    fn=gen.gen_both,
                    inputs=[*prompts, *txt2img_params],
                    outputs=imagegal_orig + imagegal_targ
                )

            GenParamGetter.events_assigned = True

if __package__ == "traintrain":
    script_callbacks.on_ui_tabs(on_ui_tabs)
    script_callbacks.on_app_started(GenParamGetter.get_params_components)
