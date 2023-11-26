import ml_collections
import random

def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    config = ml_collections.ConfigDict()
    
    
    config.device = "cuda:7"
    # config.seed = 3214
    # # config.seed = random.randint(500, 2000) # generate a random seed
    config.pred = 'noise_pred'
    config.z_shape = (4, 64, 64)
    config.clip_img_dim = 512
    config.clip_text_dim = 768
    config.text_dim = 64  # reduce dimension
    config.data_type = 1
    config.gradient_accumulation_steps = 1
    config.log_interval = 50
    config.eval_interval = 100
    config.save_interval = 300
    config.max_step = 1500
        
    config.num_workers = 10
    config.batch_size = 6
    config.resolution = 512
    config.closerprompt = "photo of a <new1> girl"
    config.clip_img_model = "other_models/clip"
    # config.clip_text_model = "/data/hdd3/schengwei/models--openai--clip-vit-large-patch14/snapshots/8d052a0f05efbaefbc9e8786ba291cfdf93e5bff"
    config.clip_text_model ="other_models/models--openai--clip-vit-large-patch14"
    # config.uvit = "other_models/first_15000.pth"
    config.uvit = "/home/wuyujia/.cache/final_test/nnet.pth"
    config.only_load_model = True
    

    config.optimizer = d(
        name='adamw',
        lr=2e-5,
        weight_decay=0.03,
        betas=(0.9, 0.9),
        amsgrad=False
    )

    config.lr_scheduler = d(
        name='customized',
        warmup_steps=20
    )

    config.autoencoder = d(
        pretrained_path='models/autoencoder_kl.pth',
    )

    config.caption_decoder = d(
        pretrained_path="models/caption_decoder.pth",
        hidden_dim=config.get_ref('text_dim'),
        tokenizer_path = "./models/gpt2"
    )

    config.nnet = d(
        img_size=64,
        in_chans=4,
        patch_size=2,
        embed_dim=1536,
        depth=30,
        num_heads=24,
        mlp_ratio=4,
        qkv_bias=False,
        pos_drop_rate=0.,
        drop_rate=0.,
        attn_drop_rate=0.,
        mlp_time_embed=False,
        text_dim=config.get_ref('text_dim'),
        num_text_tokens=77,
        clip_img_dim=config.get_ref('clip_img_dim'),
        use_checkpoint=True
    )


    # sample
    config.mode = "t2i"
    config.n_samples = 6 # control the numbers of generating images 
    config.n_iter = 1 # 过多的迭代次数可能导致过拟合或生成的样本过于接近训练数据
    config.sample = d(
        sample_steps=80,
        scale=9, # scale of the text embedding 7 - 10
        t2i_cfg_mode='true_uncond', # 'empty_token' or 'true_uncond'
    )

    return config
