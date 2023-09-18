import os.path
import re
import time

import cv2
from PIL import Image, ImageOps

import modules.images
from modules import shared, processing
from modules.generation_parameters_copypaste import create_override_settings_dict
from modules.processing import StableDiffusionProcessingImg2Img, process_images, Processed
from modules.shared import opts, state
import modules.scripts as scripts
from scripts.m2m_util import get_mov_all_images, images_to_video
from scripts.m2m_config import mov2mov_outpath_samples, mov2mov_output_dir
from modules.ui import plaintext_to_html
from scripts.m2m_modnet import get_model, infer, infer2
import requests
import json
from modules.api.api import decode_base64_to_image
import tempfile
from modules.shared import encode_image_to_base64
from enum import Enum
import numpy as np
try:
    from modules.sd_samplers import visible_sampler_names
except:
    from modules.sd_samplers import all_samplers
    def visible_sampler_names():
        return all_samplers

import modules.sd_models as sd_models
import modules.sd_vae as sd_vae


def process_mov2mov(p, mov_file, movie_frames, max_frames, resize_mode, w, h, generate_mov_mode,
                    # extract_characters,
                    # merge_background,
                    # modnet_model,
                    modnet_enable, modnet_background_image, modnet_background_movie, modnet_model, modnet_resize_mode,
                    modnet_merge_background_mode, modnet_movie_frames,

                    args):
    processing.fix_seed(p)

    # ModNet checks
    if (modnet_enable):
        # Warn the user if ModNet is enabled without selecting a model, then disable it to prevent errors
        if modnet_model.lower() == 'none':
          print('\nError: ModNet for mov2mov is enabled without selecting a model, please select a model for ModNet\n')
          return
        # Warn the user if ModNet image mode is enabled without selecting an image
        if modnet_merge_background_mode == 3 and modnet_background_image is None:
          print('\nError: ModNet for mov2mov is enabled in Image mode without a valid image, please select a valid image for ModNet\n')
          return
        # Warn the user if ModNet video mode is enabled without selecting a video
        if modnet_merge_background_mode == 4 and modnet_background_movie is None:
          print('\nError: ModNet for mov2mov is enabled in Video mode without a valid video, please select a valid video for ModNet\n')
          return

    # 判断是不是多prompt
    re_prompts = re.findall(r'\*([0-9]+):(.*?)\|\|', p.prompt, re.DOTALL)
    re_negative_prompts = re.findall(r'\*([0-9]+):(.*?)\|\|', p.negative_prompt, re.DOTALL)
    prompts = {}
    negative_prompts = {}
    for ppt in re_prompts: prompts[int(ppt[0])] = ppt[1]
    for ppt in re_negative_prompts: negative_prompts[int(ppt[0])] = ppt[1]

    images = get_mov_all_images(mov_file, movie_frames)
    if not images:
        print('Failed to parse the video, please check')
        return
    if modnet_enable and modnet_merge_background_mode == 4:
        background_images = get_mov_all_images(modnet_background_movie, modnet_movie_frames)
        if not background_images:
            print('Failed to parse the background video, please check')
            return

    print(f'The video conversion is completed, images:{len(images)}')
    if max_frames == -1 or max_frames > len(images):
        max_frames = len(images)

    max_frames = int(max_frames)

    p.do_not_save_grid = True
    state.job_count = max_frames  # * p.n_iter
    generate_images = []
    for i, image in enumerate(images):
        if i >= max_frames:
            break

        if i + 1 in prompts.keys():
            p.prompt = prompts[i + 1]
            print(f'change prompt:{p.prompt}')

        if i + 1 in negative_prompts.keys():
            p.negative_prompt = negative_prompts[i + 1]
            print(f'change negative_prompts:{p.negative_prompt}')

        state.job = f"{i + 1} out of {max_frames}"
        if state.skipped:
            state.skipped = False

        if state.interrupted:
            break

        modnet_network = None
        # 存一张底图
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 'RGB')
        img = ImageOps.exif_transpose(img)

        b_img = img.copy()
        if modnet_enable and modnet_model:
            # 提取出人物
            print(f'loading modnet model: {modnet_model}')
            modnet_network = get_model(modnet_model)
            print(f'Loading modnet model completed')
            b_img, _ = infer2(modnet_network, img)

        p.init_images = [b_img] * p.batch_size
        proc = scripts.scripts_img2img.run(p, *args)
        if proc is None:
            print(f'current progress: {i + 1}/{max_frames}')
            processed = process_images(p)
            # Get generated image
            try:
                gen_image = processed.images[0]
            except IndexError as e:
                # Replace generated image with black image if something went wrong
                print(f'Replaced frame {i + 1} with a black frame because of an image generation error: {e}')
                gen_image = Image.new('RGB', (w, h), 'black')
            

            # 合成图像
            if modnet_enable and modnet_merge_background_mode != 0:
                if modnet_merge_background_mode == 1:
                    backup = img
                elif modnet_merge_background_mode == 2:
                    backup = Image.new('RGB', (w, h), 'green')
                elif modnet_merge_background_mode == 3:
                    backup = Image.fromarray(modnet_background_image, 'RGB')

                elif modnet_merge_background_mode == 4:
                    # mov
                    if i >= len(background_images):
                        backup = Image.new('RGB', (w, h), 'green')
                    else:
                        backup = Image.fromarray(cv2.cvtColor(background_images[i], cv2.COLOR_BGR2RGB), 'RGB')

                backup = modules.images.resize_image(modnet_resize_mode, backup, w, h)
                b_img = modules.images.resize_image(resize_mode, img, w, h)
                # 合成
                _, mask = infer2(modnet_network, b_img)

                # Resize image before composite to avoid 'ValueError: images do not match'
                gen_image = modules.images.resize_image(resize_mode, gen_image, w, h)

                # Run composite
                gen_image = Image.composite(gen_image, backup, mask)

            # if extract_characters and merge_background:
            #     backup = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 'RGB')
            #     backup = modules.images.resize_image(resize_mode, backup, w, h)
            #     _, mask = infer2(modnet_network, backup)
            #     gen_image = Image.composite(gen_image, backup, mask)

            generate_images.append(gen_image)

    if not os.path.exists(shared.opts.data.get("mov2mov_output_dir", mov2mov_output_dir)):
        os.makedirs(shared.opts.data.get("mov2mov_output_dir", mov2mov_output_dir), exist_ok=True)

    if generate_mov_mode == 0:
        r_f = '.mp4'
        codec = 'libx264'
    elif generate_mov_mode == 1:
        r_f = '.mp4'
        codec = 'libx264'
    elif generate_mov_mode == 2:
        r_f = '.avi'
        codec = 'libxvid'

    out_path=os.path.join(shared.opts.data.get("mov2mov_output_dir", mov2mov_output_dir), str(int(time.time())) + r_f)

    video = images_to_video(generate_images, movie_frames, w, h, codec, out_path)

    if video is not None:
        print(f'Video generation completed, file saved in: {video}')

    return video


def mov2mov(id_task: str,
            prompt,
            negative_prompt,
            mov_file,
            steps,
            sampler,
            restore_faces,
            tiling,
            # extract_characters,
            # merge_background,
            # modnet_model,
            modnet_enable, modnet_background_image, modnet_background_movie, modnet_model, modnet_resize_mode,
            modnet_merge_background_mode, modnet_movie_frames,

            # fixed_seed,
            generate_mov_mode,
            noise_multiplier,
            # color_correction,
            cfg_scale,
            image_cfg_scale,
            denoising_strength,
            movie_frames,
            max_frames,
            # seed,
            # subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, seed_enable_extras,
            height,
            width,
            resize_mode,
            override_settings_text, *args):
    # Stop if no input video was provided
    if not mov_file:
        print('Error: mov2mov generation was started without a valid video, please select a valid input video for mov2mov')
        return
    if isinstance(sampler, int):
        sampler = visible_sampler_names()[sampler].name

    if shared.cmd_opts.just_ui:
        override_settings_text.append(f'sd_model_checkpoint: {shared.sd_model.sd_checkpoint_info.model_name}')
        override_settings_text.append(f'sd_vae: {shared.sd_model.sd_checkpoint_info.model_name}')
        override_settings_text.append(f'outpath_samples: {shared.opts.data.get("mov2mov_outpath_samples", mov2mov_outpath_samples)}')
        override_settings_text.append(f'outpath_grids: {opts.outdir_grids or opts.outdir_img2img_grids}')
        if not mov_file.startswith(shared.cmd_opts.data_dir):
            os.makedirs(f'{shared.cmd_opts.data_dir}/outputs/mov2mov/videos', exist_ok=True)
            os.system(f"cp {mov_file} {shared.cmd_opts.data_dir}/outputs/mov2mov/videos/{mov_file.split('/')[-1]}")
            mov_file = f"{shared.cmd_opts.data_dir}/outputs/mov2mov/videos/{mov_file.split('/')[-1]}"
        req_dict = {"args":[id_task, prompt, negative_prompt, mov_file, steps, sampler, restore_faces, tiling,
            modnet_enable, modnet_background_image, modnet_background_movie, modnet_model, modnet_resize_mode,
            modnet_merge_background_mode, modnet_movie_frames, generate_mov_mode, noise_multiplier, cfg_scale,
            image_cfg_scale, denoising_strength, movie_frames, max_frames, height, width, resize_mode, override_settings_text]}
        alwayson_args = {}
        for alwayson_scripts in modules.scripts.scripts_img2img.alwayson_scripts:
            if alwayson_scripts.name is None:
                continue
            alwayson_args[alwayson_scripts.name] = {}
            alwayson_args[alwayson_scripts.name]['args'] = []
            for _arg in args[alwayson_scripts.args_from:alwayson_scripts.args_to]:
                if alwayson_scripts.name=='controlnet':
                    if isinstance(_arg, dict):
                        cn_dict = _arg
                    else:
                        cn_dict = vars(_arg)
                    for cn_key, cn_val in cn_dict.items():
                        if isinstance(cn_val, Enum):
                            cn_val = cn_val.value
                        elif (cn_key == 'image') and (isinstance(cn_val, dict)):
                            if cn_val.get('image', None) is not None:
                                cn_val['image'] = encode_image_to_base64(cn_val['image'])
                            if cn_val.get('mask', None) is not None:
                                cn_val['mask'] = encode_image_to_base64(cn_val['mask'])
                        elif (cn_key == 'image') and (isinstance(cn_val, (np.ndarray, Image.Image, str))):
                            cn_val = encode_image_to_base64(cn_val)
                        elif cn_key=='model' and cn_val:
                            cn_val = cn_val.split(' ')[0]
                        cn_dict[cn_key] = cn_val
                    alwayson_args[alwayson_scripts.name]['args'].append(cn_dict)
                elif isinstance(_arg, (int, float, str, bool)):
                    alwayson_args[alwayson_scripts.name]['args'].append(_arg)
                elif isinstance(_arg, Enum):
                    alwayson_args[alwayson_scripts.name]['args'].append(_arg.value)
                elif isinstance(_arg, Image.Image):
                    alwayson_args[alwayson_scripts.name]['args'].append(encode_image_to_base64(_arg))
                elif isinstance(_arg, tempfile._TemporaryFileWrapper):
                    filename = _arg.name
                    if filename.endswith(('.png', '.jpg')):
                        alwayson_args[alwayson_scripts.name]['args'].append(encode_image_to_base64(Image.open(filename)))
                    else:
                        print(f'mov2mov {alwayson_scripts.name} {filename} is not a image, will be droped')
                        alwayson_args[alwayson_scripts.name]['args'].append(filename)
                elif isinstance(_arg, list):
                    _arg_list = []
                    for tmp_arg in _arg:
                        if isinstance(tmp_arg, tempfile._TemporaryFileWrapper):
                            filename = tmp_arg.name
                            if filename.endswith(('.png', '.jpg')):
                                _arg_list.append(encode_image_to_base64(Image.open(filename)))
                            else:
                                _arg_list.append(filename)
                        elif isinstance(tmp_arg, (int, float, str, bool)):
                            _arg_list.append(tmp_arg)
                        elif isinstance(tmp_arg, Image.Image):
                            _arg_list.append(encode_image_to_base64(tmp_arg))
                        else:
                            try:
                                json.dumps(tmp_arg)
                                _arg_list.append(tmp_arg)
                            except Exception as e:
                                print(f'mov2mov {alwayson_scripts.title().lower()} {_arg} some args are not in [int, float, str, bool]')
                    alwayson_args[alwayson_scripts.title().lower()]['args'].append(_arg_list)
                else:
                    try:
                        json.dumps(_arg)
                        alwayson_args[alwayson_scripts.name]['args'].append(_arg)
                    except Exception as e:
                        print(f'mov2mov {alwayson_scripts.name} {_arg} some args are not in [int, float, str, bool], {e}')
        req_dict['alwayson_args'] = alwayson_args
        result = requests.post('/'.join([shared.cmd_opts.server_path, 'mov2mov/process']), json=req_dict)
        if result.status_code == 200:
            result = json.loads(result.text)
            return [decode_base64_to_image(img_b64) for img_b64 in result['images']], result['generate_video_path'], \
                result['generation_info_js'], result['info'], result['comments']
        else:
            raise f"run id_task {id_task} failed with {result.text}"
    shared.state.id_task = id_task
    override_settings = create_override_settings_dict(override_settings_text)
    tmp_dict = {}
    for txt_pair in override_settings_text:
        for k, v in txt_pair.split(":", maxsplit=1):
            tmp_dict[k] = v.strip()
    for key in ['outpath_samples', 'outpath_grids', 'sd_model_checkpoint', 'sd_vae']:
        if key in tmp_dict:
            override_settings[key] = tmp_dict[key]
    assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'
    mask_blur = 4
    inpainting_fill = 1
    inpaint_full_res = False
    inpaint_full_res_padding = 32
    inpainting_mask_invert = 0
    outpath_samples = override_settings.pop('outpath_samples', shared.opts.data.get("mov2mov_outpath_samples", mov2mov_outpath_samples))
    outpath_grids = override_settings.pop('outpath_grids', opts.outdir_grids or opts.outdir_img2img_grids)
    p = StableDiffusionProcessingImg2Img(
        sd_model=shared.sd_model,
        outpath_samples=outpath_samples,
        outpath_grids=outpath_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        styles=[],
        # seed=seed,
        # subseed=subseed,
        # subseed_strength=subseed_strength,
        # seed_resize_from_h=seed_resize_from_h,
        # seed_resize_from_w=seed_resize_from_w,
        # seed_enable_extras=seed_enable_extras,
        sampler_name=sampler,
        batch_size=1,
        n_iter=1,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        restore_faces=restore_faces,
        tiling=tiling,
        init_images=[None],
        mask=None,
        mask_blur=mask_blur,
        inpainting_fill=inpainting_fill,
        resize_mode=resize_mode,
        denoising_strength=denoising_strength,
        image_cfg_scale=image_cfg_scale,
        inpaint_full_res=inpaint_full_res,
        inpaint_full_res_padding=inpaint_full_res_padding,
        inpainting_mask_invert=inpainting_mask_invert,
        override_settings=override_settings,
        initial_noise_multiplier=noise_multiplier
    )
    for k, v in p.override_settings.items():
        setattr(opts, k, v)

        if k == 'sd_model_checkpoint':
            sd_models.reload_model_weights()

        if k == 'sd_vae':
            sd_vae.reload_vae_weights()

    p.scripts = scripts.scripts_img2img
    p.script_args = args

    p.extra_generation_params["Mask blur"] = mask_blur

    print(f'\nStart parsing the number of mov frames')

    generate_video = process_mov2mov(p, mov_file, movie_frames, max_frames, resize_mode, width, height,
                                     generate_mov_mode,
                                     # extract_characters, merge_background, modnet_model,
                                     modnet_enable, modnet_background_image, modnet_background_movie, modnet_model,
                                     modnet_resize_mode,
                                     modnet_merge_background_mode, modnet_movie_frames,

                                     args)
    processed = Processed(p, [], p.seed, "")
    p.close()

    shared.total_tqdm.clear()

    generation_info_js = processed.js()

    if opts.do_not_show_images:
        processed.images = []

    return processed.images, generate_video, generation_info_js, plaintext_to_html(processed.info), plaintext_to_html(
        processed.comments)
