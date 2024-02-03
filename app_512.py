import gradio as gr
import os
import sys
import argparse
import random
import time
from omegaconf import OmegaConf
import torch
import torchvision
from pytorch_lightning import seed_everything
from huggingface_hub import hf_hub_download
from einops import repeat
import torchvision.transforms as transforms
from utils.utils import instantiate_from_config
sys.path.insert(0, "scripts/evaluation")
from funcs import (
    batch_ddim_sampling,
    load_model_checkpoint,
    get_latent_z,
    save_videos
)

def download_model():
    REPO_ID = 'Doubiiu/DynamiCrafter_512'
    filename_list = ['model.ckpt']
    if not os.path.exists('./checkpoints/dynamicrafter_512_v1/'):
        os.makedirs('./checkpoints/dynamicrafter_512_v1/')
    for filename in filename_list:
        local_file = os.path.join('./checkpoints/dynamicrafter_512_v1/', filename)
        if not os.path.exists(local_file):
            hf_hub_download(repo_id=REPO_ID, filename=filename, local_dir='./checkpoints/dynamicrafter_512_v1/', force_download=True)
    

def infer(image, prompt, steps=50, cfg_scale=7.5, eta=1.0, fs=3, seed=123):
    resolution = (320, 512)
    download_model()
    ckpt_path='checkpoints/dynamicrafter_512_v1/model.ckpt'
    config_file='configs/inference_512_v1.0.yaml'
    config = OmegaConf.load(config_file)
    model_config = config.pop("model", OmegaConf.create())
    model_config['params']['unet_config']['params']['use_checkpoint']=False   
    model = instantiate_from_config(model_config)
    assert os.path.exists(ckpt_path), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, ckpt_path)
    model.eval()
    model = model.cuda()
    save_fps = 8

    seed_everything(seed)
    transform = transforms.Compose([
        transforms.Resize(min(resolution)),
        transforms.CenterCrop(resolution),
        ])
    torch.cuda.empty_cache()
    print('start:', prompt, time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
    start = time.time()
    if steps > 60:
        steps = 60 

    batch_size=1
    channels = model.model.diffusion_model.out_channels
    frames = model.temporal_length
    h, w = resolution[0] // 8, resolution[1] // 8
    noise_shape = [batch_size, channels, frames, h, w]

    # text cond
    text_emb = model.get_learned_conditioning([prompt])

    # img cond
    img_tensor = torch.from_numpy(image).permute(2, 0, 1).float().to(model.device)
    img_tensor = (img_tensor / 255. - 0.5) * 2

    image_tensor_resized = transform(img_tensor) #3,256,256
    videos = image_tensor_resized.unsqueeze(0) # bchw
    
    z = get_latent_z(model, videos.unsqueeze(2)) #bc,1,hw
    
    img_tensor_repeat = repeat(z, 'b c t h w -> b c (repeat t) h w', repeat=frames)

    cond_images = model.embedder(img_tensor.unsqueeze(0)) ## blc
    img_emb = model.image_proj_model(cond_images)

    imtext_cond = torch.cat([text_emb, img_emb], dim=1)

    fs = torch.tensor([fs], dtype=torch.long, device=model.device)
    cond = {"c_crossattn": [imtext_cond], "fs": fs, "c_concat": [img_tensor_repeat]}
    
    ## inference
    batch_samples = batch_ddim_sampling(model, cond, noise_shape, n_samples=1, ddim_steps=steps, ddim_eta=eta, cfg_scale=cfg_scale)
    ## b,samples,c,t,h,w

    video_path = './output.mp4'
    save_videos(batch_samples, './', filenames=['output'], fps=save_fps)
    model = model.cpu()
    return video_path


i2v_examples = [
    ['prompts/512/bloom01.png', 'time-lapse of a blooming flower with leaves and a stem', 50, 7.5, 1.0, 24, 123],
    ['prompts/512/campfire.png', 'a bonfire is lit in the middle of a field', 50, 7.5, 1.0, 24, 123],
    ['prompts/512/isometric.png', 'rotating view, small house', 50, 7.5, 1.0, 24, 123],
    ['prompts/512/girl08.png', 'a woman looking out in the rain', 50, 7.5, 1.0, 24, 1234],
    ['prompts/512/ship02.png', 'a sailboat sailing in rough seas with a dramatic sunset', 50, 7.5, 1.0, 24, 123],
    ['prompts/512/zreal_penguin.png', 'a group of penguins walking on a beach', 50, 7.5, 1.0, 20, 123],
]




css = """#input_img {max-width: 512px !important} #output_vid {max-width: 512px; max-height: 320px}"""

with gr.Blocks(analytics_enabled=False, css=css) as dynamicrafter_iface:
    gr.Markdown("<div align='center'> <h1> DynamiCrafter: Animating Open-domain Images with Video Diffusion Priors </span> </h1> \
                    <h2 style='font-weight: 450; font-size: 1rem; margin: 0rem'>\
                    <a href='https://doubiiu.github.io/'>Jinbo Xing</a>, \
                    <a href='https://menghanxia.github.io/'>Menghan Xia</a>, <a href='https://yzhang2016.github.io/'>Yong Zhang</a>, \
                    <a href=''>Haoxin Chen</a>, <a href=''> Wangbo Yu</a>,\
                    <a href='https://github.com/hyliu'>Hanyuan Liu</a>, <a href='https://xinntao.github.io/'>Xintao Wang</a>,\
                    <a href='https://www.cse.cuhk.edu.hk/~ttwong/myself.html'>Tien-Tsin Wong</a>,\
                    <a href='https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=zh-CN'>Ying Shan</a>\
                </h2> \
                    <a style='font-size:18px;color: #000000' href='https://arxiv.org/abs/2310.12190'> [ArXiv] </a>\
                    <a style='font-size:18px;color: #000000' href='https://doubiiu.github.io/projects/DynamiCrafter/'> [Project Page] </a> \
                    <a style='font-size:18px;color: #000000' href='https://github.com/Doubiiu/DynamiCrafter'> [Github] </a> </div>")
    
    with gr.Tab(label='ImageAnimation_320x512'):
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        i2v_input_image = gr.Image(label="Input Image",elem_id="input_img")
                    with gr.Row():
                        i2v_input_text = gr.Text(label='Prompts')
                    with gr.Row():
                        i2v_seed = gr.Slider(label='Random Seed', minimum=0, maximum=10000, step=1, value=123)
                        i2v_eta = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, label='ETA', value=1.0, elem_id="i2v_eta")
                        i2v_cfg_scale = gr.Slider(minimum=1.0, maximum=15.0, step=0.5, label='CFG Scale', value=7.5, elem_id="i2v_cfg_scale")
                    with gr.Row():
                        i2v_steps = gr.Slider(minimum=1, maximum=60, step=1, elem_id="i2v_steps", label="Sampling steps", value=50)
                        i2v_motion = gr.Slider(minimum=15, maximum=30, step=1, elem_id="i2v_motion", label="FPS", value=24)
                    i2v_end_btn = gr.Button("Generate")
                # with gr.Tab(label='Result'):
                with gr.Row():
                    i2v_output_video = gr.Video(label="Generated Video",elem_id="output_vid",autoplay=True,show_share_button=True)

            gr.Examples(examples=i2v_examples,
                        inputs=[i2v_input_image, i2v_input_text, i2v_steps, i2v_cfg_scale, i2v_eta, i2v_motion, i2v_seed],
                        outputs=[i2v_output_video],
                        fn = infer,
            )
        i2v_end_btn.click(inputs=[i2v_input_image, i2v_input_text, i2v_steps, i2v_cfg_scale, i2v_eta, i2v_motion, i2v_seed],
                        outputs=[i2v_output_video],
                        fn = infer
        )

dynamicrafter_iface.queue(max_size=12).launch(show_api=True)