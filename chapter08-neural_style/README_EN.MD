## [中文版本](README.MD)

Code for  *PyTorch: Introduction and Practice*, chapter-8 fast neural style.

## Env Setup

- Install [PyTorch](https://pytorch.org/) >= 1.0  
- `pip install -r requirements.txt` 


## Data 

Download [COCO](http://images.cocodataset.org/zips/train2014.zip) or use your own/other datasets。 save the data in `data/coco/`. Try symlink if necessary. make sure images stored in the following structure.
```Bash
data
 └─ coco
    ├── COCO_train2014_000000000009.jpg
    ├── COCO_train2014_000000000025.jpg
    ├── COCO_train2014_000000000030.jpg
```

## Run
If you would like to use `visdom`, `nohup python -m visdom.server &` to start the visdom server.

Command:
```
Usage： python main.py FUNCTION --key=value --key2=value2 ..
```

- Train
```bash
python main.py train --use-gpu --data-root=data --batch-size=2
```

- Test (style transfer)

You may download the pretrained model  `transformer.pth` from [here](https://yun.sfo2.digitaloceanspaces.com/pytorch_book/pytorch_book/transformer.pth). 
```bash
python main.py stylize  --model-path='transformer.pth' \
                 --content-path='amber.jpg'\  #
                 --result-path='output2.png'\  
                 --use-gpu=False
```

Available args:
```python
    # General Args
    use_gpu = True
    model_path = None # pretrain model path (for resume training or test)
    
    # Train Args
    image_size = 256 # image crop_size for training
    batch_size = 8  
    data_root = 'data/' # dataset root：$data_root/coco/a.jpg
    num_workers = 4 # dataloader num of workers
    
    lr = 1e-3
    epoches = 2 # total epoch to train
    content_weight = 1e5 # weight of content_loss  
    style_weight = 1e10 # weight of style_loss

    style_path= 'style.jpg' # style image path
    env = 'neural-style' # visdom env
    plot_every=10 # visualize in visdom for every 10 batch

    debug_file = '/tmp/debugnn' # touch $debug_fie to interrupt and enter ipdb 

    # Test Args
    content_path = 'input.png' # input file to do style transfer [for test]
    result_path = 'output.png' # style transfer result [for test]
   
```

Examples:

![imgs](neural-style-results.png)


To train more styles, try different style images by `--style-path=mystyle.png`

### Code Compatibility
train 
- [x] GPU  
- [ ] CPU 
- [ ] Python2
- [x] Python3

test: 
- [x] GPU
- [x] CPU
- [x] Python2
- [x] Python3
