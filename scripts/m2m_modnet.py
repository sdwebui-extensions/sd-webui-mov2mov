import os

import cv2
import gradio as gr
from PIL import Image

import modules.scripts as scripts
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import modules.paths as ph
from scripts.modnet.modnet import MODNet
from modules.shared import cmd_opts
import hashlib

"""
from modnet-entry("https://github.com/RimoChan/modnet-entry")
"""
modnet_models_path = f'{cmd_opts.data_dir}/models/mov2mov-ModNet'
if os.environ.get('SERVICE_NAME', '') != '' and cmd_opts.just_ui:
    modnet_models_path = f'{os.path.dirname(cmd_opts.data_dir)}/models/mov2mov-ModNet'
os.makedirs(modnet_models_path, exist_ok=True)

modnet_photographic_path = f"{modnet_models_path}/modnet_photographic_portrait_matting.ckpt"
modnet_webcam_path = f"{modnet_models_path}/modnet_webcam_portrait_matting.ckpt"

modnet_photographic_url = "https://huggingface.co/DavG25/modnet-pretrained-models/resolve/main/models/modnet_photographic_portrait_matting.ckpt"
modnet_photographic_checksum = "db1f7ec96b370abebbd506e360ce9819380cb45bb99930f1e955dfcbe9e4035708a6190ebf292e0b8b740cefdc879bad9146ba8158baac0a702aadf5311a8cd6"

modnet_webcam_url = "https://huggingface.co/DavG25/modnet-pretrained-models/resolve/main/models/modnet_webcam_portrait_matting.ckpt"
modnet_webcam_checksum = "19ec6baa9934f834739d496c6f5d119a4e6fe09f67d19342025d3e9199b694814bc89d4e866db1d11da5e1740eba72a2ff4116420c5c212bf886d330d526f603"

# Compare checksum to make sure downloaded file is not modified
def checksum(filename, hash_factory=hashlib.blake2b, chunk_num_blocks=128):
    h = hash_factory()
    with open(filename,'rb') as f: 
        while chunk := f.read(chunk_num_blocks*h.block_size): 
            h.update(chunk)
    return h.hexdigest()

# Download modnet_photographic_portrait_matting.ckpt
if not os.path.exists(modnet_photographic_path):
    print('Downloading model for mov2mov ModNet, this is a one time operation')
    from basicsr.utils.download_util import load_file_from_url
    load_file_from_url(modnet_photographic_url, model_dir=modnet_models_path)

    if checksum(modnet_photographic_path) != modnet_photographic_checksum:
        os.remove(modnet_photographic_path)
        print(f"\nWarning: unable to automatically downloading ModNet model for mov2mov (checksum mismatch), please manually download from {modnet_photographic_url} and place in the folder {modnet_models_path}\n")
    else:
        print('Model download for mov2mov ModNet completed\n')

# Download modnet_webcam_portrait_matting.ckpt
if not os.path.exists(modnet_webcam_path):
    print('Downloading model for mov2mov ModNet, this is a one time operation')
    from basicsr.utils.download_util import load_file_from_url
    load_file_from_url(modnet_webcam_url, model_dir=modnet_models_path)

    if checksum(modnet_webcam_path) != modnet_webcam_checksum:
        os.remove(modnet_webcam_path)
        print(f"\nWarning: unable to automatically downloading ModNet model for mov2mov (checksum mismatch), please manually download from {modnet_webcam_url} and place in the folder {modnet_models_path}\n")
    else:
        print('Model download for mov2mov ModNet completed\n')


modnet_models = ['none'] + [model for model in os.listdir(modnet_models_path) if model.endswith('.ckpt')]


def get_model(ckpt_name):
    ckpt_path = os.path.join(modnet_models_path, ckpt_name)
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)
    if torch.cuda.is_available():
        modnet = modnet.cuda()
        weights = torch.load(ckpt_path)
    else:
        weights = torch.load(ckpt_path, map_location=torch.device('cpu'))
    modnet.load_state_dict(weights)
    modnet.eval()
    return modnet


def create_modnet():
    ctrls = ()
    with gr.Group():
        with gr.Accordion("ModNet for mov2mov", open=True):
            background_image = gr.Image(label='Background', type='numpy', elem_id='modnet_background_image')
            background_movie = gr.Video(label='Background', elem_id='modnet_background_movie')
            enable = gr.Checkbox(label='Enable', value=False, )
            ctrls += (background_image, background_movie, enable)
            with gr.Row():
                mode = gr.Radio(label='Mode', choices=[
                    'Image', 'Movie'
                ], type='index', value='Image')
                guidance = gr.Radio(label='Guidance', choices=[
                    'Start', 'End'
                ], type='index', value='Start')
                ctrls += (mode, guidance)

            movie_frames = gr.Slider(minimum=10,
                                     maximum=60,
                                     step=1,
                                     label='Video frames',
                                     elem_id='modnet_movie_frames',
                                     value=30)
            ctrls += (movie_frames,)
            with gr.Row():
                models = gr.Dropdown(label='Model', choices=list(modnet_models), value='none')
                ctrls += (models,)

            with gr.Row():
                resize_mode = gr.Radio(label="Resize mode",
                                       choices=["Just resize", "Crop and resize", "Resize and fill",
                                                ], type="index", value="Just resize")
                ctrls += (resize_mode,)

    mode.change(fn=None, inputs=[mode], outputs=[], _js=f'switchModnetMode')
    return ctrls


def infer(modnet, im, ref_size=1024):
    im_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # unify image channels to 3
    if len(im.shape) == 2:
        im = im[:, :, None]
    if im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.shape[2] == 4:
        im = im[:, :, 0:3]

    # convert image to PyTorch tensor
    im = Image.fromarray(im)
    im = im_transform(im)

    # add mini-batch dim
    im = im[None, :, :, :]

    # resize image for input
    im_b, im_c, im_h, im_w = im.shape
    if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
        if im_w >= im_h:
            im_rh = ref_size
            im_rw = int(im_w / im_h * ref_size)
        elif im_w < im_h:
            im_rw = ref_size
            im_rh = int(im_h / im_w * ref_size)
    else:
        im_rh = im_h
        im_rw = im_w

    im_rw = im_rw - im_rw % 32
    im_rh = im_rh - im_rh % 32
    im = F.interpolate(im, size=(im_rh, im_rw), mode='area')

    # inference
    _, _, matte = modnet(im.cuda() if torch.cuda.is_available() else im, True)

    # resize and save matte
    matte = F.interpolate(matte, size=(im_h, im_w), mode='area')
    matte = matte[0][0].data.cpu().numpy()
    return matte


def infer2(modnet, img):
    image = np.asarray(img)
    h, w, _ = image.shape
    alpha = infer(modnet, image, max(h, w))
    alpha_bool = (~alpha.astype(np.bool)).astype('int')

    alpha_uint8 = (alpha * 255).astype('uint8')
    new_image = np.concatenate((image, alpha_uint8[:, :, None]), axis=2)
    return Image.fromarray(new_image, 'RGBA'), Image.fromarray(alpha_uint8, mode='L')
