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
    given a json object, process the task the json describes
    """
    from torch.utils.data import DataLoader
    import utils
    from libs.schedule import stable_diffusion_beta_schedule, Schedule, LSimple_T2I
    from pathlib import Path
    from libs.data import PersonalizedBasev2
    from absl import logging
    import torch
    from sample_fn import sample
    from PIL import Image
    from rembg import new_session, remove
    import os
    
    
    # accelerator = context["accelerator"]
    # config = context["config"]
    # device = context["device"]
    # train_state = context["train_state"]
    # origin_sd = context["origin_sd"]
    # caption_decoder = context["caption_decoder"]
    # nnet = context["nnet"]
    # autoencoder = context["autoencoder"]
    # clip_text_model = context["clip_text_model"]
    # clip_img_model = context["clip_img_model"]
    # clip_img_model_preprocess = context["clip_img_model_preprocess"]
    
    # # 初始化训练步数
    # train_state.step = 0
    # # 重新初始化模型
    # nnet.load_state_dict(origin_sd, False)

    """
    处理数据部分
    """
    # process data
    #image_paths = [i["path"] for i in json_data["source_group"]]
    output_dir = '/home/schengwei/Competitionrepo/ot'
    output_path = os.path.join(output_dir, f"{json_data['id']}.jpg")
    model = 'u2netp'
    image_path = json_data["source_group"][0]["path"]
    image = Image.open(image_path)
    output = remove(image, session=new_session(model))
    output = output.convert("RGB") 
    output.save(output_path)

    # config.n_samples = 4
    # config.n_iter = 1
    # images = []
    # for caption in json_data["caption_list"]:
    #     config.prompt = caption
    #     config.output = output
    #     paths = sample(config, nnet, clip_text_model, autoencoder, caption_decoder, device, json_data["id"], output_path=image_output_path)
    #     images.append({"prompt": caption, "paths": paths})
        
    # return {
    #     "id": json_data["id"],
    #     "images": images
    # }