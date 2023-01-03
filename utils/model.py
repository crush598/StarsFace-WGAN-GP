# -*- coding: utf-8 -*-
# @Time    : 2023/12/24 20:23
# @Author  : Hush
# @Email   : crush@tju.edu.cn

from model import *

def build_model(cfg):

    total_model = {
        'DCGAN': DCGAN,
        'WGAN': WGAN,
        'WGANP': WGANP,
        'WGAN256': WGAN256,
    }
    model = total_model[cfg.MODEL.NAME](cfg)

    if cfg.MODEL.DEVICE == 'cuda':
        model = model.cuda()
    
    return model