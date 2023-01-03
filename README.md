# StarsFace-WGAN-GP

## Introduction: Using WGAN-GP to generate stars' faces

- WGAN-GP 的网络结构
- ![image-20230103172406017](https://cdn.jsdelivr.net/gh/crush598/image@main/AI202301031800281.png)

- 原始人脸图像数据
- ![image-20230103172542134](https://cdn.jsdelivr.net/gh/crush598/image@main/AI202301031800364.png)
- 模型生成的人脸图像

- ![WGAN_TEST_one](https://cdn.jsdelivr.net/gh/crush598/image@main/AI202301031808451.png)

## Requirements

- python : 3.8
- [pytorch](https://pytorch.org/get-started/previous-versions/)

  ```bash
  # CUDA 11.1.1
  conda install pytorch==1.10.0 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
  ```

- other dependencies

  ```bash
  pip install -r requirements.txt
  ```

## Dataset

- [stars_face](https://github.com/a312863063/seeprettyface-generator-star)

  整个[数据集](https://pan.baidu.com/s/1g5ASVZcRoYvClxqsQpShXQ?pwd=XVAL?from=init)包含 10000 张分辨率为 1024x1024 的高质量明星人脸图像数据。


下载zip并且解压到 `./dataset` 文件夹下, `dataset/stars`

## Use

### Train

- run WGAN model as

  ```bash
  python train.py --config-file configs/WGAN.yaml
  ```

### Generate images

- use WGAN model as

  ```bash
  python generate.py --config-file configs/WGAN.yaml -g checkpoints/WGAN/WGAN_G_epoch_39999.pth.pth
  ```

- use WGANP model as

  ```bash
  python generate.py --config-file configs/WGANP.yaml -g checkpoints/WGANP/WGANP_G_epoch_39999.pth.pth
  ```

默认情况下,它会在 `./images` 下生成一个 8x8 网格的动漫图片

其他:

- `-g`: generator 的缩写,一个参数是模型权重的路径名

  **这里的模型是指生成器G的模型**

- `-s`: separate 的缩写,没有参数

  可以使用`-s`将将一张大图拆分成每一张小图

## Pretrained model

- software

  ```txt
  OS: CentOS 7.5 Linux X86_64
  Python: 3.8 (anaconda)
  PyTorch: 1.10.0
  ```

- hardware

  ```txt
  CPU: 4核 12GB
  GPU: 显存 20GB
  ```

如果不想自己训练（大约36~48h）,可以下载预训练好的模型G并将其移动到`./checkpoints`下,然后按照上文`Generate images`生成图片, **注意修改后面的生成器G的模型路径**

| 模型    | 数据集 | 判别器                                                       | 生成器                                                       |
| ------- | ------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| WGAN-GP | Stars  | [下载](https://pan.baidu.com/s/1R93S14hY_zbyl7pAbhxF9A?pwd=h7f2) | [下载](https://pan.baidu.com/s/1rcOV79ZosW5JLYppEE3YyQ?pwd=ajdy) |

## Result

- generated fake images

  > 其实不是所有生成的图片都好看,我手动选择了一些我喜欢的图片,使用模型 `WGAN-GP + stars`

  ![image-20230103174654530](https://cdn.jsdelivr.net/gh/crush598/image@main/AI202301031800781.png)

- walking latent space

- ![walking_latent_space](https://cdn.jsdelivr.net/gh/crush598/image@main/AI202301031801649.gif)

- GAN training process

  > 对于相同的噪声输入,不同生成图像的过程

- ![WGAN_TEST_one_process](https://cdn.jsdelivr.net/gh/crush598/image@main/AI202301031802463.gif)
- 训练过程中损失函数的变化情况
- ![loss](https://cdn.jsdelivr.net/gh/crush598/image@main/AI202301031803858.png)

## Conclusion

这是我第一次尝试GAN,久闻大名但从未尝试学习它。恰逢神经网络与深度学习期末课程设计，课程设计的主题是做一些与深度学习相关的事情。所以这是学习 GAN 的好机会！于是看了一些GAN的论文，真的很有意思。

[GAN学习指南：从原理入门到制作生成Demo](https://zhuanlan.zhihu.com/p/24767059)启发了尝试生成虚拟的明星人脸图像

实际上我必须承认,我的预训练模型并没有像我期待的那样表现出色,时至今日有很多优秀的模型在生成图像上具有更好的效果,比如diffusion.

这个项目的最初目的只是为了学习一些关于 GAN 的东西，我选择使用 WGAN-GP 是因为它的数学推导很棒，我想尝试动手写一下而不是只是双击运行,那样稍显无趣.

特别感谢 [pytorch-wgan](https://github.com/Zeleni9/pytorch-wgan),绝大部分代码都参考自这个项目

其实一开始我想生成 256x256 的图像，但最后确选择了 64x64 图像并制作该数据集。首先的原因的自己的资源有限，其次问题为随着图像尺寸的增大，模型表现极差。也许我需要一个更好的模型结构。我试过使用residual block ,但效果不佳。

## Relevant reference

project:

- [Github-GAN+DCGAN+WGAN-pytorch](https://github.com/Zeleni9/pytorch-wgan) (easy to use,recommand)
- [Github-GAN*-pytorch](https://github.com/eriklindernoren/PyTorch-GAN) (overall)
- [Github-DCGAN-tensorflow](https://github.com/carpedm20/DCGAN-tensorflow)
- [Github-GAN*-tensorflow](https://github.com/YadiraF/GAN)

knowledge:

- [知乎-GAN学习指南：从原理入门到制作生成Demo](https://zhuanlan.zhihu.com/p/24767059)
- [GAN video by Li mu](https://www.bilibili.com/video/BV1rb4y187vD)
- [KL散度](https://zhuanlan.zhihu.com/p/365400000)
- [GAN](https://www.zhihu.com/search?q=GAN&type=content&sort=upvoted_count)
- [WGAN-DCGAN](https://github.com/martinarjovsky/WassersteinGAN/blob/master/models/dcgan.py)
- [WGAN-GP](https://github.com/EmilienDupont/wgan-gp)
- https://zhuanlan.zhihu.com/p/28407948?ivk_sa=1024320u
- [evaluation-index](https://zhuanlan.zhihu.com/p/432965561)
- https://zhuanlan.zhihu.com/p/25071913
- https://zhuanlan.zhihu.com/p/58260684
- [生成ANIMEfaces](https://arxiv.org/pdf/1708.05509.pdf)
- [checkerboard](https://distill.pub/2016/deconv-checkerboard/)
- https://www.zhihu.com/search?type=content&q=GAN%20%E6%A3%8B%E7%9B%98
- https://zhuanlan.zhihu.com/p/58260684
- https://make.girls.moe/#/
- [latent walk](https://www.zhihu.com/search?type=content&q=latent%20walk)
