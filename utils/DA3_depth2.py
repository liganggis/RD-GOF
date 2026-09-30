#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os, sys
from pathlib import Path
dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
print(f"dir_path {dir_path}")
sys.path.append(dir_path.__str__())

import torch
import torchvision
from scene import Scene
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from PIL import Image
from tqdm import tqdm

def resize_image(dataset : ModelParams, pipeline : PipelineParams):
    
    with torch.no_grad():
        gaussians = GaussianModel(3)
        scene = Scene(dataset, gaussians, shuffle=False)

        views = scene.getTrainCameras()
        
        output_path = os.path.join(dataset.source_path, "images_2")
        for idx, view in enumerate(tqdm(views, desc="Resize progress")):
            gt_image, _ = view.get_image()
            torchvision.utils.save_image(gt_image.clamp(0.0, 1.0), os.path.join(output_path, view.image_name + ".png"))


        


if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument('--mesh_path', type=str,  default='')
    parser.add_argument('--output_dir', type=str,  default='')

    args = get_combined_args(parser)
    
    # render_sets(model.extract(args), pipeline.extract(args))
    resize_image(model.extract(args), pipeline.extract(args))

    # python ./utils/DA3_depth2.py -s ../../data/DTU_da3/scan37/ -r 2 