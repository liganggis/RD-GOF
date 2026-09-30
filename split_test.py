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
from gaussian_renderer import render, reproject_error
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import cv2
import open3d as o3d
from scene.app_model import AppModel
import trimesh, copy
from collections import deque
from utils.loss_utils import l1_loss
from utils.image_utils import psnr
import cv2
from utils.general_utils import update_config

def render_set(model_path, name, iteration, views, scene, gaussians, pipeline, background, kernel_size,
               bounds=None, max_depth=5.0, volume=None, use_depth_filter=False, use_normal_filter= False, depth_threshold=0.9):
    depths_tsdf_fusion = []
    # uids = []

    _, H, W = views[0].original_image.shape
    ix, iy = torch.meshgrid(
                    torch.arange(W), torch.arange(H), indexing='xy')
    pixels = torch.stack([ix, iy], dim=-1).float().cuda()
    cx = int(views[0].Cx)
    cy = int(views[0].Cy)
    nums = len(views)
    

    for _, view in enumerate(tqdm(views, desc="Rendering Depth progress")):
        render_pkg = render(view, gaussians, pipeline, background, kernel_size=kernel_size, app_model=None)
        depth = render_pkg['mid_depth']
        depths_tsdf_fusion.append(depth)
        # uids.append(view.uid)
    # uids = torch.tensor(uids).cuda()

    temp_d_mask = torch.zeros((H, W), dtype=torch.float32).cuda()
    
    

    errors = torch.zeros((nums, nums), dtype=torch.float32).cuda()
    errors2 = torch.zeros((nums, nums), dtype=torch.float32).cuda()

    nearest_id = []

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    iter_start.record()
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_depth = depths_tsdf_fusion[idx]
        bounds = bounds* 1.0

        block_mask = torch.ones((H, W), dtype=torch.int32).cuda() > 0
        if bounds is not None:
            pts = gaussians.get_points_from_depth(view, render_depth)
            unvalid_mask = (pts[...,0] < bounds[0,0]) | (pts[...,0] > bounds[0,1]) |\
                                        (pts[...,1] < bounds[1,0]) | (pts[...,1] > bounds[1,1]) |\
                                        (pts[...,2] < bounds[2,0]) | (pts[...,2] > bounds[2,1])
            unvalid_mask = unvalid_mask.reshape(H,W)
            block_mask[unvalid_mask] = False
        
        d_masks_list = []
        uids_list = []
        for idx_n, view_n in enumerate(views):
            if idx_n == idx:
                # d_masks_list.append(temp_d_mask)
                continue
            # if idx_n <= idx:
            #     continue
            # view_n = views[idx_n]
            nearest_depth = depths_tsdf_fusion[idx_n]
            d_mask = reproject_error(gaussians, view, view_n, render_depth[None], nearest_depth[None], block_mask, 3, pipeline, background, kernel_size=kernel_size)

            
            # d_masks_list.append(d_mask.reshape(H, W))
            # uids_list.append(view_n.uid)
            
            d_mask_sum = torch.sum(d_mask)
            errors[idx][idx_n] = d_mask_sum
            # di = view.camera_center - view_n.camera_center
            errors2[idx][idx_n] = torch.norm(view.camera_center - view_n.camera_center, p=2, dim=0)

        # d_masks = torch.stack(d_masks_list, dim=0)
        # values, indices = torch.topk(torch.sum(d_masks.reshape(nums, -1), dim=-1), 16)
        # nearest_id.append(indices[values > 1000].detach().cpu().tolist())
    

    iter_end.record()
    torch.cuda.synchronize()
    elapsed_time = iter_start.elapsed_time(iter_end) / 1000.0
    print(f"CUDA time: {elapsed_time: .3f} s")
    print('-------------------')
    values, indices = torch.topk(errors, dim=-1, k=12)
    
    # # print(indices[100][values[0] > 1000].detach().cpu().tolist())
    # print(views[115].nearest_id)
    # print(indices[115].detach().cpu().tolist())
    # print(views[303].nearest_id)
    # print(indices[303].detach().cpu().tolist())
    # print(views[304].nearest_id)
    # print(indices[304].detach().cpu().tolist())
    
    for i, view in enumerate(views):
        # temp_values = values[i][:8]
        # if sum(values[i]) < 12000:
            print('--------------{}'.format(i))
            print(views[i].nearest_id)
            print(indices[i].detach().cpu().tolist())
            print(errors2[i][indices[i]])
    

    # [99, 98, 97, 96, 95, 101, 606, 94, 607, 608, 93, 102, 605, 621, 124, 622]


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
                 max_depth : float, voxel_size : float, tsdf: float, num_cluster: int, use_depth_filter : bool, use_normal_filter : bool, depth_threshold: float):
    
    update_config(dataset, None, True, args.cfg_path)
    dataset.preload_grayImg = False
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bounds = None
        js_file = f"{scene.source_path}/bounding.json"
        if os.path.exists(js_file):
            with open(js_file) as file:
                meta = json.load(file)
                if "aabb_range" in meta:
                    bounds = (np.array(meta["aabb_range"]))

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if bounds is not None:
            max_dis = np.max(bounds[:,1]-bounds[:,0])
            voxel_size = max_dis / 2048.0
        print('----------------------------')
        print(bounds)
            
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_size,
        sdf_trunc=tsdf * voxel_size,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)


        if not skip_train:
            # render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), scene, gaussians, pipeline, background, dataset.kernel_size,
            #            bounds=bounds, max_depth=max_depth, volume=volume, use_depth_filter=use_depth_filter, use_normal_filter=use_normal_filter, depth_threshold = depth_threshold)
            # print(f"extract_triangle_mesh")
            print(scene.getTrainCameras()[0].Fx)
            print(scene.getTrainCameras()[0].Fy)


if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument('--cfg_path', type=str,  default='')
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max_depth", default=20.0, type=float)
    parser.add_argument("--voxel_size", default=0.002, type=float)
    parser.add_argument("--tsdf", default=4.0, type=float)
    parser.add_argument("--num_cluster", default=1, type=int)
    parser.add_argument("-UDF", "--use_depth_filter", action="store_true")
    parser.add_argument("-UNF", "--use_normal_filter", action="store_true")
    parser.add_argument('-DT', "--depth_threshold", default=0.9, type=float)

    args = get_combined_args(parser)
    # update_config(model, pipeline, None, args.cfg_path)
    print("Rendering " + args.model_path)
    # python eval_tnt/run.py --dataset-dir ../../data/TNT_GOF/TNT_real/Barn --traj-path ../../data/TNT_GOF/TNT/Barn/Barn_COLMAP_SfM.log --ply-path ./exps/TNT/Barn/mesh/tsdf_fusion.ply
    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.max_depth, args.voxel_size, args.tsdf, args.num_cluster, args.use_depth_filter, args.use_normal_filter, args.depth_threshold)