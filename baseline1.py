#####################################################################################################################
####################################k########  load_model.py exactly ##############################################
#####################################################################################################################




def prepare_context():
    """
    prepare context for later use
    """
    import torch
    import utils
    from utils import set_logger
    from absl import logging
    import os
    import libs.autoencoder
    import clip
    from libs.clip import FrozenCLIPEmbedder
    from libs.caption_decoder import CaptionDecoder
    from libs.uvit_multi_post_ln_v1 import UViT
    from configs.unidiffuserv1 import get_config
    import builtins
    import ml_collections
    from torch import multiprocessing as mp
    import accelerate
    
    config = get_config()
    mp.set_start_method('spawn')
    assert config.gradient_accumulation_steps == 1, \
        'fix the lr_scheduler bug before using larger gradient_accumulation_steps'
    accelerator = accelerate.Accelerator(gradient_accumulation_steps=config.gradient_accumulation_steps, mixed_precision="fp16")
    device = accelerator.device
    accelerate.utils.set_seed(config.seed, device_specific=True)
    logging.info(f'Process {accelerator.process_index} using device: {device}')

    config.mixed_precision = accelerator.mixed_precision

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        set_logger(log_level='info')
        logging.info(config)
    else:
        set_logger(log_level='error')
        builtins.print = lambda *args: None
    logging.info(f'Run on {accelerator.num_processes} devices')
    
    train_state = utils.initialize_train_state(config, device, uvit_class=UViT)
    origin_sd = torch.load("models/uvit_v1.pth", map_location='cpu')
    
    caption_decoder = CaptionDecoder(device=device, **config.caption_decoder)
    nnet, optimizer = accelerator.prepare(train_state.nnet, train_state.optimizer)
    
    nnet.to(device)
    lr_scheduler = train_state.lr_scheduler
    autoencoder = libs.autoencoder.get_model(**config.autoencoder).to(device)
    
    clip_text_model = FrozenCLIPEmbedder(version=config.clip_text_model, device=device)
    clip_img_model, clip_img_model_preprocess = clip.load(config.clip_img_model, jit=False)
    clip_img_model.to(device).eval().requires_grad_(False)
    
    return {
        "accelerator": accelerator,
        "device": device,
        'config': config,
        "train_state": train_state,
        "origin_sd": origin_sd,
        "caption_decoder": caption_decoder,
        "nnet": nnet,
        "autoencoder": autoencoder,
        "clip_text_model": clip_text_model,
        "clip_img_model": clip_img_model,
        "clip_img_model_preprocess": clip_img_model_preprocess
    }
    


def process_one_json(json_data, image_output_path, context={}):
    """
    Given a json object, process the task the json describes.
    :param json_data: A dictionary containing the input data for the task.
    :param image_output_path: A string representing the output path for the processed images.
    :param context: A dictionary containing the context for the task.
    :return: A dictionary containing the processed images and their corresponding captions.
    """
    # Import necessary modules
    from torch.utils.data import DataLoader
    import utils
    from libs.schedule import stable_diffusion_beta_schedule, Schedule, LSimple_T2I
    from pathlib import Path
    from libs.data import PersonalizedBasev2
    from absl import logging
    import torch
    from sample_fn import sample
    
    # Get context variables
    accelerator = context["accelerator"]
    config = context["config"]
    device = context["device"]

    nnet = context["nnet"]
    autoencoder = context["autoencoder"]
    clip_text_model = context["clip_text_model"]
    clip_img_model = context["clip_img_model"]
    clip_img_model_preprocess = context["clip_img_model_preprocess"]
    
    # 初始化训练步数
    # 重新初始化模型
    # Initialize training step and load model state dictionary
    train_state.step = 0
    nnet.load_state_dict(origin_sd, False)

    """
    处理数据部分
    """
    # Process data
    image_paths = [i["path"] for i in json_data["source_group"]]
    train_dataset = PersonalizedBasev2(image_paths, resolution=512, class_word=json_data["class_word"])
    train_dataset_loader = DataLoader(train_dataset,
                                      batch_size=config.batch_size,
                                      num_workers=config.num_workers,
                                      pin_memory=True,
                                      drop_last=True
                                      )

    train_data_generator = utils.get_data_generator(train_dataset_loader, enable_tqdm=accelerator.is_main_process, desc='train')
    
    _betas = stable_diffusion_beta_schedule()
    schedule = Schedule(_betas)
    logging.info(f'use {schedule}')
    
    # Get model parameters and initialize optimizer and learning rate scheduler
    params = []
    for name, paramter in nnet.named_parameters():
        if 'attn.qkv' in name:
            params.append(paramter)
    optimizer = utils.get_optimizer(params, **config.optimizer)
    lr_scheduler = utils.get_lr_scheduler(optimizer, **config.lr_scheduler)
    
    # Prepare model, optimizer, and learning rate scheduler
    nnet, optimizer, lr_scheduler = accelerator.prepare(nnet, optimizer, lr_scheduler)
    
    # Define training step function
    def train_step():
        metrics = dict()
        img, img4clip, text, data_type = next(train_data_generator)
        img = img.to(device)
        img4clip = img4clip.to(device)
        data_type = data_type.to(device)

        with torch.no_grad():
            z = autoencoder.encode(img)
            clip_img = clip_img_model.encode_image(img4clip).unsqueeze(1)
            text = clip_text_model.encode(text)
            text = caption_decoder.encode_prefix(text)

        loss, loss_img, loss_clip_img, loss_text = LSimple_T2I(img=z, clip_img=clip_img, text=text, data_type=data_type, nnet=nnet, schedule=schedule, device=device)
        accelerator.backward(loss.mean())
        optimizer.step()
        lr_scheduler.step()
        train_state.step += 1
        optimizer.zero_grad()
        metrics['loss'] = accelerator.gather(loss.detach().mean()).mean().item()
        metrics['loss_img'] = accelerator.gather(loss_img.detach().mean()).mean().item()
        metrics['loss_clip_img'] = accelerator.gather(loss_clip_img.detach().mean()).mean().item()
        metrics['scale'] = accelerator.scaler.get_scale()
        metrics['lr'] = train_state.optimizer.param_groups[0]['lr']
        return metrics
    
    # Define loop function for training
    def loop():
        log_step = 0
        while True:
            nnet.train()
            total_step = train_state.step * config.batch_size
            with accelerator.accumulate(nnet):
                metrics = train_step()

            if accelerator.is_main_process and total_step >= log_step:
                nnet.eval()
                total_step = train_state.step * config.batch_size
                logging.info(utils.dct2str(dict(step=total_step, **metrics)))
                log_step += config.log_interval

            accelerator.wait_for_everyone()
            
            if total_step  >= config.max_step:
                break
    loop()
    
    # Set configuration for image sampling
    config.n_samples = 4
    config.n_iter = 1
    images = []
    # Sample images for each caption in the input data
    for caption in json_data["caption_list"]:
        config.prompt = caption
        paths = sample(config, nnet, clip_text_model, autoencoder, caption_decoder, device, json_data["id"], output_path=image_output_path)
        images.append({"prompt": caption, "paths": paths})
        
    # Return dictionary containing processed images and their corresponding captions
    return {
        "id": json_data["id"],
        "images": images
    }