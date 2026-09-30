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
from scene import Scene
import json
from tqdm import tqdm
from os import makedirs
import torchvision
from argparse import ArgumentParser 
import numpy as np

import cv2

def get_filename_in_folder(path):
    filenames = []
    for filename in os.listdir(path):
        if os.path.isfile(os.path.join(path, filename)):

            filenames.append(os.path.splitext(filename)[0])
    return filenames

if __name__ == "__main__":

    # scenes = ['Barn', 'Caterpillar', 'Courthouse', 'Ignatius', 'Meetingroom', 'Truck']
    # scenes = ["scan24", "scan37", "scan63", "scan65", "scan69", "scan97", "scan105", "scan110"]
    scene_name = "Courthouse"
    # images_folder_name = "scan24"
    # scene_name = "scan65"
    # images_folder_name = "DTU_images"

    gof_path = "../../data/Meshes/TnT_images/GOF/" + scene_name
    pgsr_path = "../../data/Meshes/TnT_images/PGSR/" + scene_name
    gggs_path = "../../data/Meshes/TnT_images/GGGS/" + scene_name
    turbogof_path = "../../data/Meshes/TnT_images/TurboGOF/" + scene_name

    # print(os.path.abspath(os.path.join(gof_path,  "000001_simple.png")))
    output_path = "../../data/Meshes/TnT_images/Temp/" + scene_name
    # output_path = os.path.join(output_path0, scene_name)
    os.makedirs(output_path, exist_ok=True)

    filenames = get_filename_in_folder(os.path.abspath(turbogof_path))
    # print(filenames[0:10])
    for filename in filenames:
        if int(filename.split("_")[0]) >-1:
            img_gof = cv2.imread(os.path.abspath(os.path.join(gof_path, filename + ".png")))
            img_pgsr = cv2.imread(os.path.abspath(os.path.join(pgsr_path, filename + ".png")))
            img_gggs = cv2.imread(os.path.abspath(os.path.join(gggs_path, filename + ".png")))
            img_turbogof = cv2.imread(os.path.abspath(os.path.join(turbogof_path, filename + ".png")))

            images = [img_gof, img_pgsr, img_gggs, img_turbogof]
            heights = [img.shape[0] for img in images]
            widths = [img.shape[1] for img in images]
            target_h = max(heights)
            target_w = max(widths)

            resized_images = [cv2.resize(img, (target_w, target_h)) for img in images]

            img_rows = np.concatenate(resized_images, axis=1)
            cv2.imwrite(os.path.join(output_path, filename + "_row.png"), img_rows)