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

import torch
from scene import Scene
import os
import json
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
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
from utils.general_utils import update_config

def clean_mesh(mesh, min_len=1000):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < min_len
    mesh_0 = copy.deepcopy(mesh)
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    return mesh_0

def post_process_mesh(mesh, cluster_to_keep=1):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    print(np.sort(cluster_n_triangles.copy())[-5:])
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def render_set(model_path, name, iteration, views, scene, gaussians, pipeline, background, kernel_size, 
               app_model=None, max_depth=5.0, volume=None, use_depth_filter=False, depth_threshold=0.95):
    depths_tsdf_fusion = []
    rgbs_tsdf_fusion = []
    _, H, W = views[0].original_image.shape
    ix, iy = torch.meshgrid(
                    torch.arange(W), torch.arange(H), indexing='xy')
    pixels = torch.stack([ix, iy], dim=-1).float().cuda() 

    # gts_path = os.path.join(model_path, name, "gt")
    # # render_path = os.path.join(model_path, name, "renders")
    # render_depth_path = os.path.join(model_path, name, "renders_depth")
    # render_mid_depth_path = os.path.join(model_path, name, "renders_mid_depth")
    # render_normal_path = os.path.join(model_path, name, "renders_normal")
    # depth_normal_path = os.path.join(model_path, name, "depth_normal")
    # depth_mid_normal_path = os.path.join(model_path, name, "depth_mid_normal")

    # makedirs(gts_path, exist_ok=True)
    # # makedirs(render_path, exist_ok=True)
    # makedirs(render_depth_path, exist_ok=True)
    # makedirs(render_normal_path, exist_ok=True)
    # makedirs(render_mid_depth_path, exist_ok=True)
    # makedirs(depth_normal_path, exist_ok=True)
    # makedirs(depth_mid_normal_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        # rendering = render(view, gaussians, pipeline, background, kernel_size=kernel_size, require_depth=True)["render"]
        # _, H, W = rendering[:3, :, :].shape

        # gt_image, _ = view.get_image()

        render_pkg = render(view, gaussians, pipeline, background, kernel_size=kernel_size, app_model=None)
        
        depth = render_pkg['alpha_depth'].squeeze()
        depth_tsdf = depth.clone()
        
        rgb = torch.clamp(render_pkg['render'].cpu(), 0, 1)*255.0
        
        depths_tsdf_fusion.append(depth_tsdf.squeeze())
        rgbs_tsdf_fusion.append(rgb.squeeze().cpu())

        torch.cuda.empty_cache()
        
    if volume is not None:
        depths_tsdf_fusion = torch.stack(depths_tsdf_fusion, dim=0)
        for idx, view in enumerate(tqdm(views, desc="TSDF Fusion progress")):
            ref_depth = depths_tsdf_fusion[idx].cuda()
            color = rgbs_tsdf_fusion[idx].cuda()

            W, H = view.image_width, view.image_height

            if view.mask is not None:
                ref_depth[view.mask.squeeze()] = 0
            else:
                ref_depth[ref_depth>max_depth] = 0
            
            if use_depth_filter and len(view.nearest_id) > 1:
                nearest_world_view_transforms = scene.world_view_transforms[view.nearest_id]
                depths_nearest = depths_tsdf_fusion[view.nearest_id]
                
                temp = torch.zeros_like(ref_depth)
                for i, world_view_transform in enumerate(nearest_world_view_transforms):
                    pixel_noise, d_mask = gaussians.compute_depth_loss_for_mesh_extract(view, world_view_transform, ref_depth[None, ...], depths_nearest[i][None, ...], pixels, 1)
                    # temp += d_mask.reshape(H, W)
                    temp += pixel_noise.reshape(H, W) > depth_threshold
                depth_mask = temp > 0
                ref_depth[~depth_mask] = 0
                
            
            pose = np.identity(4)
            pose[:3,:3] = view.R.transpose(-1,-2)
            pose[:3, 3] = view.T
            ndc2pix = torch.tensor([
                [W / 2, 0, 0, (W - 1) / 2],
                [0, H / 2, 0, (H - 1) / 2],
                [0, 0, 0, 1]]).float().cuda().T
            # ndc2pix = torch.tensor([
            #     [W / 2, 0, 0, (W - 0) / 2],
            #     [0, H / 2, 0, (H - 0) / 2],
            #     [0, 0, 0, 1]]).float().cuda().T
            intrins =  (view.projection_matrix @ ndc2pix)[:3,:3].T
            intrinsic=o3d.camera.PinholeCameraIntrinsic(
                width=view.image_width,
                height=view.image_height,
                cx = intrins[0,2].item(),
                cy = intrins[1,2].item(), 
                fx = intrins[0,0].item(), 
                fy = intrins[1,1].item()
            )
            
            # extrinsic=np.asarray((view.world_view_transform.T).cpu().numpy())
            extrinsic = np.identity(4)
            extrinsic[:3,:3] = view.R.transpose(-1,-2)
            extrinsic[:3, 3] = view.T

            o3d_color = o3d.geometry.Image(np.asarray(color.permute(1,2,0).cpu().numpy(), order="C", dtype=np.uint8))
            o3d_depth = o3d.geometry.Image(np.asarray((ref_depth*1000).cpu().numpy().astype(np.uint16), order="C"))

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_color, o3d_depth, depth_scale=1000.0, depth_trunc=max_depth, convert_rgb_to_intensity=False)
            volume.integrate(
                rgbd,
                intrinsic,
                extrinsic)
            # volume.integrate(
            #     rgbd,
            #     o3d.camera.PinholeCameraIntrinsic(W, H, view.focal_x, view.focal_y, (W - 1) / 2, (H - 1) / 2),
            #     extrinsic)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool,
                 max_depth : float, voxel_size : float, tsdf: float, num_cluster: int, use_depth_filter : bool, depth_threshold: float):
    
    update_config(dataset, None, True, args.cfg_path)
    dataset.preload_grayImg = False
    with torch.no_grad():
        
        # gaussians = GaussianModel(dataset.sh_degree)
        # scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)


        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=tsdf * voxel_size,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        if not skip_train:
            gaussians = GaussianModel(dataset.sh_degree)
            scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=True)
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), scene, gaussians, pipeline, background, dataset.kernel_size, 
                       max_depth=max_depth, volume=volume, use_depth_filter=use_depth_filter, depth_threshold=depth_threshold)
            torch.cuda.empty_cache()
            print(f"extract_triangle_mesh")
            mesh = volume.extract_triangle_mesh()

            path = os.path.join(dataset.model_path, "tsdf_u/")
            os.makedirs(path, exist_ok=True)
            
            mesh = post_process_mesh(mesh, num_cluster)
            if use_depth_filter:
                o3d.io.write_triangle_mesh(os.path.join(path, "tsdf_depth_filter.ply"), mesh, 
                                        write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
            else:
                o3d.io.write_triangle_mesh(os.path.join(path, "tsdf.ply"), mesh, 
                                        write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)

        # if not skip_test:
        #     render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), scene, gaussians, pipeline, background)

if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument('--cfg_path', type=str,  default='')
    parser.add_argument("--iterations", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max_depth", default=5.0, type=float)
    parser.add_argument("--voxel_size", default=0.002, type=float)
    parser.add_argument("--tsdf", default=4.0, type=float)
    parser.add_argument('-DT', "--depth_threshold", default=0.95, type=float)
    parser.add_argument("--num_cluster", default=1, type=int)
    parser.add_argument("-UDF", "--use_depth_filter", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    print(f"multi_view_num {model.multi_view_num}")
    render_sets(model.extract(args), args.iterations, pipeline.extract(args), args.skip_train, args.skip_test, args.max_depth, args.voxel_size, args.tsdf, args.num_cluster, args.use_depth_filter, args.depth_threshold)