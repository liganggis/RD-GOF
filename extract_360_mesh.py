import torch
from scene import Scene
import os
from os import makedirs
from gaussian_renderer import render, integrate, preprocess_points_pre
import random
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import trimesh
from tetranerf.utils.extension import cpp
from utils.tetmesh import marching_tetrahedra
from skimage.morphology import binary_dilation, disk
import torch.nn.functional as F
import open3d as o3d
from utils.general_utils import update_config

def focus_point_fn(poses: np.ndarray) -> np.ndarray:
  """Calculate nearest point to all focal axes in poses."""
  directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]
  m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])
  mt_m = np.transpose(m, [0, 2, 1]) @ m
  focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
  return focus_pt

@torch.no_grad()
def estimate_bounding_sphere(viewpoint_stack):
    """
    Estimate the bounding sphere given camera pose
    """
    torch.cuda.empty_cache()
    c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in viewpoint_stack])
    poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
    center = (focus_point_fn(poses))
    radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
    center = torch.from_numpy(center).float().cuda()
    print(f"The estimated bounding radius is {radius:.2f}")
    return radius, center

def circle_to_bbox(center, radii):
    mins = center - radii
    maxs = center + radii

    return torch.cat((mins.unsqueeze(-1), maxs.unsqueeze(-1)), dim = -1).reshape((3,2))

@torch.no_grad()
def get_point_from_depth(view, depth, mask):
    RT = view.world_view_transform[:3,:3].transpose(-1,-2)
    T = view.world_view_transform[3,:3]
    W, H = view.image_width, view.image_height
    intrins = view.get_k().cuda()
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float() + 0.0, torch.arange(H, device='cuda').float() + 0.0, indexing='xy')
    pts = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    rays_d = pts @ intrins.inverse().T
    # rays_d = torch.nn.functional.normalize(rays_d, dim=-1)
    pts = depth.reshape(-1, 1) * rays_d
    return (pts[mask.reshape(-1)] - T) @ RT

@torch.no_grad()
def evaluage_alpha(points, views, gaussians, pipeline, background, kernel_size, depth_threshold, bias):
    final_alpha = []
    chunk_size = 10000000
    # chunk_nums = points.shape[0] // chunk_size + 1
    for point_chunk in torch.chunk(points, points.shape[0] // chunk_size + 1):
        final_alpha_chunk = torch.ones(point_chunk.shape[0], dtype=torch.float32, device="cuda")
        for view in tqdm(views, desc="Rendering progress"):
            mask_outside = final_alpha_chunk >= 0.49

            ret = integrate(point_chunk[mask_outside], view, gaussians, pipeline, background, kernel_size=kernel_size, depth_threshold=depth_threshold, bias=bias)
            final_alpha_chunk[mask_outside] = torch.min(final_alpha_chunk[mask_outside], ret["alpha_integrated"])
            # ret = integrate(point_chunk, view, gaussians, pipeline, background, kernel_size=kernel_size, depth_threshold=depth_threshold, bias=bias)
            # final_alpha_chunk = torch.min(final_alpha_chunk, ret["alpha_integrated"])

        final_sdf_chunk = 1 - final_alpha_chunk
        final_alpha.append(final_sdf_chunk)
    return torch.cat(final_alpha)

def getInterpolationPoints(p0, p1, ratio):
    return (p1 - p0) * ratio + p0

@torch.no_grad()
def marching_tetrahedra_with_binary_search(model_path, name, iteration, views, gaussians, pipeline, background, kernel_size, 
                                           length_threshold, add_internal_point, depth_threshold, bias, scale_ratio, filter_mesh : bool, texture_mesh : bool, near : float, far : float, bounds : None, bound_scale_ratio : float):
    render_path = os.path.join(model_path, "fusion_v1")

    makedirs(render_path, exist_ok=True)

    # generate tetra points here
    points = gaussians.get_tetra_points(views, bounds, bound_scale_ratio, scale_ratio, add_internal_point, near, far)
    # colud = trimesh.points.PointCloud(points.cpu().numpy())
    # colud.export(os.path.join(render_path, "points.ply"))
    
    # load cell if exists
    if os.path.exists(os.path.join(render_path, "cells.pt")):
        print("load existing cells")
        cells = torch.load(os.path.join(render_path, "cells.pt"))
    else:
        # create cell and save cells
        print("create cells and save")
        cells = cpp.triangulate(points)
    # we should filter the cell if it is larger than the gaussians
    torch.save(cells, os.path.join(render_path, "cells.pt"))

    # evaluate alpha
    alpha = evaluage_alpha(points, views, gaussians, pipeline, background, kernel_size, depth_threshold, bias)

    vertices = points.cuda()[None]
    tets = cells.cuda().long()

    print(vertices.shape, tets.shape, alpha.shape)
    def alpha_to_sdf(alpha):    
        sdf = alpha - 0.5
        sdf = sdf[None]
        return sdf
    
    sdf = alpha_to_sdf(alpha)

    
    torch.cuda.empty_cache()
    verts_list, faces_list, _ = marching_tetrahedra(vertices, tets, sdf)
    torch.cuda.empty_cache()
    
    end_points, end_sdf = verts_list[0]

    faces=faces_list[0].cpu().numpy()
    points = (end_points[:, 0, :] + end_points[:, 1, :]) / 2.
        
    left_points = end_points[:, 0, :].squeeze()
    right_points = end_points[:, 1, :].squeeze()
    left_sdf = end_sdf[:, 0, :].squeeze()
    right_sdf = end_sdf[:, 1, :].squeeze()
    distance = torch.linalg.norm(left_points - right_points, ord=2, dim=-1)

    indexes = torch.arange(left_points.shape[0]).cuda()

    length_distance_threshold = length_threshold
    mask = distance > length_distance_threshold
    new_left_points = left_points[mask]
    new_right_points = right_points[mask]
    new_left_sdf = left_sdf[mask]
    new_right_sdf = right_sdf[mask]
    new_indexes = indexes[mask]

    # print(distance.shape)
    # print(torch.sum(distance > 0.005))
    # print(torch.sum(distance > 0.0025))
    # print(torch.sum(distance > 0.001))

    final_left_points = []
    final_right_points = []
    final_left_sdf = []
    final_indexes = []
    final_left_points.append(left_points[~mask])
    final_right_points.append(right_points[~mask])
    final_left_sdf.append(left_sdf[~mask])
    final_indexes.append(indexes[~mask])

    points = (left_points + right_points) / 2

    loop_iter = 0
    while torch.sum(mask) > 0:
        print("loop iteration in step {}".format(loop_iter))
        loop_iter += 1
        if loop_iter > 20:
            print("Exceeding the max steps!")
            print(new_left_points.shape)

            final_left_points.append(new_left_points)
            final_right_points.append(new_right_points)
            final_left_sdf.append(new_left_sdf[~mask])
            final_indexes.append(new_indexes)
            break
        print('--------------------- {}'.format(new_left_points.shape))

        if torch.sum(mask) < 150000: # split 8 parts -1,0,1,2,3,4,5,6,-1
            parts = 8
            pts = parts - 1

            mid_points_list = []
            for i in range(pts):
                mid_points_list.append(getInterpolationPoints(new_left_points, new_right_points, (i + 1) / parts))
            mid_points = torch.cat(mid_points_list, dim=0)

            alpha = evaluage_alpha(mid_points, views, gaussians, pipeline, background, kernel_size, depth_threshold, bias)
            mid_sdf = alpha_to_sdf(alpha).squeeze()
            mid_sdf = torch.chunk(mid_sdf, chunks=pts, dim=0)
            mid_points = torch.chunk(mid_points, chunks=pts, dim=0)

            temp_mid_sdf = mid_sdf[3]
            temp_mid_points = mid_points[3]
            
            ind_low = ((temp_mid_sdf < 0) & (new_left_sdf < 0)) | ((temp_mid_sdf > 0) & (new_left_sdf > 0))
            new_left_sdf[ind_low] = temp_mid_sdf[ind_low]
            new_right_sdf[~ind_low] = temp_mid_sdf[~ind_low]
            new_left_points[ind_low] = temp_mid_points[ind_low]
            new_right_points[~ind_low] = temp_mid_points[~ind_low]
            
            temp_mid_sdf[ind_low] = mid_sdf[5][ind_low]
            temp_mid_sdf[~ind_low] = mid_sdf[1][~ind_low]
            temp_mid_points[ind_low] = mid_points[5][ind_low]
            temp_mid_points[~ind_low] = mid_points[1][~ind_low]
            
            ind_low1 = ((temp_mid_sdf < 0) & (new_left_sdf < 0)) | ((temp_mid_sdf > 0) & (new_left_sdf > 0))
            new_left_sdf[ind_low1] = temp_mid_sdf[ind_low1]
            new_right_sdf[~ind_low1] = temp_mid_sdf[~ind_low1]
            new_left_points[ind_low1] = temp_mid_points[ind_low1]
            new_right_points[~ind_low1] = temp_mid_points[~ind_low1]

            temp_mid_sdf[ind_low & ind_low1] = mid_sdf[6][ind_low & ind_low1]
            temp_mid_sdf[ind_low & ~ind_low1] = mid_sdf[4][ind_low & ~ind_low1]
            temp_mid_sdf[~ind_low & ind_low1] = mid_sdf[2][~ind_low & ind_low1]
            temp_mid_sdf[~ind_low & ~ind_low1] = mid_sdf[0][~ind_low & ~ind_low1]

            ind_low2 = ((temp_mid_sdf < 0) & (new_left_sdf < 0)) | ((temp_mid_sdf > 0) & (new_left_sdf > 0))
            new_left_sdf[ind_low2] = temp_mid_sdf[ind_low2]
            new_right_sdf[~ind_low2] = temp_mid_sdf[~ind_low2]
            new_left_points[ind_low2] = temp_mid_points[ind_low2]
            new_right_points[~ind_low2] = temp_mid_points[~ind_low2]

            distance = torch.linalg.norm(new_left_points - new_right_points, ord=2, dim=-1)
            mask = distance > length_distance_threshold

            final_left_points.append(new_left_points[~mask])
            final_right_points.append(new_right_points[~mask])
            final_left_sdf.append(new_left_sdf[~mask])
            final_indexes.append(new_indexes[~mask])
            
            new_left_points = new_left_points[mask]
            new_right_points = new_right_points[mask]
            new_left_sdf = new_left_sdf[mask]
            new_right_sdf = new_right_sdf[mask]
            new_indexes = new_indexes[mask]

        elif torch.sum(mask) < 300000: # split 4 parts 0,1,2
            parts = 4
            pts = parts - 1

            mid_points_list = []
            for i in range(pts):
                mid_points_list.append(getInterpolationPoints(new_left_points, new_right_points, (i + 1) / parts))
            mid_points = torch.cat(mid_points_list, dim=0)

            alpha = evaluage_alpha(mid_points, views, gaussians, pipeline, background, kernel_size, depth_threshold, bias)
            mid_sdf = alpha_to_sdf(alpha).squeeze()
            mid_sdf = torch.chunk(mid_sdf, chunks=pts, dim=0)
            mid_points = torch.chunk(mid_points, chunks=pts, dim=0)

            temp_mid_sdf = mid_sdf[1]
            temp_mid_points = mid_points[1]

            ind_low = ((temp_mid_sdf < 0) & (new_left_sdf < 0)) | ((temp_mid_sdf > 0) & (new_left_sdf > 0))
            new_left_sdf[ind_low] = temp_mid_sdf[ind_low]
            new_right_sdf[~ind_low] = temp_mid_sdf[~ind_low]
            new_left_points[ind_low] = temp_mid_points[ind_low]
            new_right_points[~ind_low] = temp_mid_points[~ind_low]

            temp_mid_sdf[ind_low] = mid_sdf[2][ind_low]
            temp_mid_sdf[~ind_low] = mid_sdf[0][~ind_low]
            temp_mid_points[ind_low] = mid_points[2][ind_low]
            temp_mid_points[~ind_low] = mid_points[0][~ind_low]

            ind_low = ((temp_mid_sdf < 0) & (new_left_sdf < 0)) | ((temp_mid_sdf > 0) & (new_left_sdf > 0))
            new_left_sdf[ind_low] = temp_mid_sdf[ind_low]
            new_right_sdf[~ind_low] = temp_mid_sdf[~ind_low]
            new_left_points[ind_low] = temp_mid_points[ind_low]
            new_right_points[~ind_low] = temp_mid_points[~ind_low]

            distance = torch.linalg.norm(new_left_points - new_right_points, ord=2, dim=-1)
            mask = distance > length_distance_threshold

            final_left_points.append(new_left_points[~mask])
            final_right_points.append(new_right_points[~mask])
            final_left_sdf.append(new_left_sdf[~mask])
            final_indexes.append(new_indexes[~mask])
            
            new_left_points = new_left_points[mask]
            new_right_points = new_right_points[mask]
            new_left_sdf = new_left_sdf[mask]
            new_right_sdf = new_right_sdf[mask]
            new_indexes = new_indexes[mask]

        # elif torch.sum(mask) < 70000: # split 16 parts

        else:
            mid_points = (new_left_points + new_right_points) / 2
            alpha = evaluage_alpha(mid_points, views, gaussians, pipeline, background, kernel_size, depth_threshold, bias)
            mid_sdf = alpha_to_sdf(alpha).squeeze()
            ind_low = ((mid_sdf < 0) & (new_left_sdf < 0)) | ((mid_sdf > 0) & (new_left_sdf > 0))

            new_left_sdf[ind_low] = mid_sdf[ind_low]
            new_right_sdf[~ind_low] = mid_sdf[~ind_low]
            new_left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
            new_right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]

            distance = torch.linalg.norm(new_left_points - new_right_points, ord=2, dim=-1)
            mask = distance > length_distance_threshold

            final_left_points.append(new_left_points[~mask])
            final_right_points.append(new_right_points[~mask])
            final_left_sdf.append(new_left_sdf[~mask])
            final_indexes.append(new_indexes[~mask])

            new_left_points = new_left_points[mask]
            new_right_points = new_right_points[mask]
            new_left_sdf = new_left_sdf[mask]
            new_right_sdf = new_right_sdf[mask]
            new_indexes = new_indexes[mask]

    if final_left_points:
        final_left_points = torch.cat(final_left_points, dim=0)
        final_right_points = torch.cat(final_right_points, dim=0)
        final_left_sdf = torch.cat(final_left_sdf, dim=0)
        indexes = torch.cat(final_indexes, dim=0)

        mask_sdf_left = final_left_sdf >= 0
        temp_left_points = torch.empty_like(final_left_points)
        temp_right_points = torch.empty_like(final_right_points)
        temp_left_points[mask_sdf_left] = final_left_points[mask_sdf_left]
        temp_left_points[~mask_sdf_left] = final_right_points[~mask_sdf_left]
        temp_right_points[~mask_sdf_left] = final_left_points[~mask_sdf_left]
        temp_right_points[mask_sdf_left] = final_right_points[mask_sdf_left]

        # left_points = torch.empty_like(final_left_points)
        # right_points = torch.empty_like(final_right_points)
        left_points[indexes] = temp_left_points
        right_points[indexes] = temp_right_points

        points = (left_points + right_points) / 2
    else:
        print("The final points is empty!")

    print("We have finished the all steps!")
        
    mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), faces=faces, process=False)
    # mesh.export(os.path.join(render_path, f"mesh_{step}.ply"))
    mesh.export(os.path.join(render_path, "mesh_0.ply"))


    # mesh = trimesh.Trimesh(vertices=left_points.cpu().numpy(), faces=faces, process=False)
    # mesh.export(os.path.join(render_path, "mesh_0l.ply"))
    # mesh = trimesh.Trimesh(vertices=right_points.cpu().numpy(), faces=faces, process=False)
    # mesh.export(os.path.join(render_path, "mesh_0r.ply"))

def extract_mesh(dataset : ModelParams, iterations : int, pipeline : PipelineParams, depth_threshold : float, bounds_scale: float, filter_mesh : bool, texture_mesh : bool, near : float, far : float):
    update_config(dataset, None, True, args.cfg_path)
    dataset.preload_grayImg = False
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iterations, shuffle=False)
        
        # gaussians.load_ply(os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size

        radius, center = estimate_bounding_sphere(scene.getTrainCameras())
        # print(radius.shape)
        # print(center.shape)
        
        bounds = circle_to_bbox(center, radius)
        # bounds_scale = 10
        bounds = bounds * bounds_scale
        
        cams = scene.getTrainCameras()
        # gaussians.compute_3D_filter(cams)
        marching_tetrahedra_with_binary_search(dataset.model_path, "test", iterations, cams, gaussians, pipeline, 
                                               background, kernel_size, dataset.length_threshold, dataset.add_internal_point, depth_threshold,
                                                 dataset.bias, dataset.scale_ratio, filter_mesh, texture_mesh, near, far, bounds, dataset.bound_scale_ratio)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument('--cfg_path', type=str,  default='')
    parser.add_argument("--iterations", default=30000, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--filter_mesh", action="store_true")
    parser.add_argument("--texture_mesh", action="store_true")
    parser.add_argument("--near", default=0.02, type=float)
    parser.add_argument("--far", default=1e6, type=float)
    parser.add_argument('-DT', "--depth_threshold", default=3.5, type=float)
    parser.add_argument('-BS', "--bounds_scale", default=3, type=float)
    # parser.add_argument('-B', "--bias", default=0, type=float)
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
    extract_mesh(model.extract(args), args.iterations, pipeline.extract(args), args.depth_threshold, args.bounds_scale, args.filter_mesh, args.texture_mesh, args.near, args.far)