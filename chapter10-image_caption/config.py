# coding:utf8


class Config:
    caption_data_path = 'caption.pth'  # 经过预处理后的人工描述信息
    img_path = '/home/cy/caption_data/'
    # img_path='/mnt/ht/aichallenger/raw/ai_challenger_caption_train_20170902/caption_train_images_20170902/'
    img_feature_path = 'results.pth'  # 所有图片的features,20w*2048的向量
    scale_size = 300
    img_size = 224
    batch_size = 8
    shuffle = True
    num_workers = 4
    rnn_hidden = 256
    embedding_dim = 256
    num_layers = 2
    share_embedding_weights = False

    prefix = 'checkpoints/caption'  # 模型保存前缀

    env = 'caption'
    plot_every = 10
    debug_file = '/tmp/debugc'

    model_ckpt = None  # 模型断点保存路径
    lr = 1e-3
    use_gpu = True
    epoch = 1

    test_img = 'img/example.jpeg'
