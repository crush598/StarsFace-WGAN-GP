# -*- coding: utf-8 -*-
# @Time    : 2023/12/25 8:21
# @Author  : Hush
# @Email   : crush@tju.edu.cn

from config.config import get_cfg
from utils import *

def main():
    
    cfg = get_cfg()
    cfg = project_preprocess(cfg)
    
    model = build_model(cfg)
    model.load_model()
    model.generate_images()
    print('Generate images successfully!')
    

if __name__ == '__main__':
    main()