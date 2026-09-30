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
    n_cluster = max(n_cluster, 50) # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0

def cull_mesh(cameras, mesh):
    
    vertices = mesh.vertices
    
    # project and filter
    vertices = torch.from_numpy(vertices).cuda()
    vertices = torch.cat((vertices, torch.ones_like(vertices[:, :1])), dim=-1)
    vertices = vertices.permute(1, 0)
    vertices = vertices.float()
    
    sampled_masks = []
    
    for camera in cameras:
        c2w = (camera.world_view_transform.T).inverse()
        w2c = torch.inverse(c2w).cuda()
        mask = camera.gt_alpha_mask
        
        intrinsic = torch.eye(4)
        intrinsic[0, 0] = camera.focal_x
        intrinsic[1, 1] = camera.focal_y
        intrinsic[0, 2] = camera.image_width / 2.
        intrinsic[1, 2] = camera.image_height / 2.
        intrinsic = intrinsic.cuda()

        W, H = camera.image_width, camera.image_height
        
        with torch.no_grad():
            # transform and project
            cam_points = intrinsic @ w2c @ vertices
            pix_coords = cam_points[:2, :] / (cam_points[2, :].unsqueeze(0) + 1e-6)
            pix_coords = pix_coords.permute(1, 0)
            pix_coords[..., 0] /= W - 1
            pix_coords[..., 1] /= H - 1
            pix_coords = (pix_coords - 0.5) * 2
            valid = ((pix_coords > -1. ) & (pix_coords < 1.)).all(dim=-1).float()
            
            # dialate mask similar to unisurf
            maski = mask[0, :, :].cpu().numpy().astype(np.float32) / 256.
            maski = torch.from_numpy(binary_dilation(maski, disk(6))).float()[None, None].cuda()
            
            sampled_mask = F.grid_sample(maski, pix_coords[None, None], mode='nearest', padding_mode='zeros', align_corners=True)[0, -1, 0]

            sampled_mask = sampled_mask + (1. - valid)

            sampled_masks.append(sampled_mask)
        
    sampled_masks = torch.stack(sampled_masks, -1)

    # filter
    mask = (sampled_masks > 0.).all(dim=-1).cpu().numpy()
    face_mask = mask[mesh.faces].all(axis=1)
    
    mesh.update_vertices(mask)
    mesh.update_faces(face_mask)

    return mesh

@torch.no_grad()
def evaluage_cull_alpha(points, views, gaussians, pipeline, background, kernel_size):
    # final_sdf = torch.zeros((points.shape[0]), dtype=torch.float32, device="cuda")
    final_sdf = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    weight = torch.zeros((points.shape[0]), dtype=torch.int32, device="cuda")
    with torch.no_grad():
        for cam_id, view in enumerate(tqdm(views, desc="Rendering progress")):
            torch.cuda.empty_cache()
            ret = integrate(points, view, gaussians, pipeline, background, kernel_size)
            alpha_integrated = ret["alpha_integrated"]
            point_coordinate = ret["point_coordinate"]
            point_coordinate[:,0] = (point_coordinate[:,0]*2+1)/(views[cam_id].image_width-1) - 1
            point_coordinate[:,1] = (point_coordinate[:,1]*2+1)/(views[cam_id].image_height-1) - 1
            rendered_mask = ret["render"][7]
            mask = rendered_mask[None]
            if not view.mask is None:
                mask = mask * view.mask
            valid_point_prob = torch.nn.functional.grid_sample(mask.type(torch.float32)[None],point_coordinate[None,None],padding_mode='zeros',align_corners=False)
            valid_point_prob = valid_point_prob[0,0,0]
            valid_point = valid_point_prob>0.5
            final_sdf = torch.where(valid_point, torch.min(alpha_integrated,final_sdf), final_sdf)
            weight = torch.where(valid_point, weight+1, weight)
        final_sdf = torch.where(weight>0,0.5-final_sdf,-100)
    return final_sdf

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
def evaluage_alpha(points, views, gaussians, pipeline, background, kernel_size, depth_threshold, bias, return_color=False):
    final_alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    
    with torch.no_grad():
        for _, view in enumerate(tqdm(views, desc="Rendering progress")):

            mask_outside = final_alpha >= 0.49
            
            ret = integrate(points[mask_outside], view, gaussians, pipeline, background, kernel_size=kernel_size, depth_threshold=depth_threshold, bias=bias)
            alpha_integrated = ret["alpha_integrated"]
            final_alpha[mask_outside] = torch.min(final_alpha[mask_outside], alpha_integrated)
            
            
        alpha = 1 - final_alpha
    return alpha

# @torch.no_grad()
# def evaluage_alpha(points, views, gaussians, pipeline, background, kernel_size, depth_threshold, bias, return_color=False):
#     final_alpha = torch.ones((points.shape[0]), dtype=torch.float32, device="cuda")
    
#     with torch.no_grad():
#         for _, view in enumerate(tqdm(views, desc="Rendering progress")):
            
#             ret = integrate(points, view, gaussians, pipeline, background, kernel_size=kernel_size, depth_threshold=depth_threshold, bias=bias)
#             alpha_integrated = ret["alpha_integrated"]
#             final_alpha = torch.min(final_alpha, alpha_integrated)
            
#         alpha = 1 - final_alpha
#     return alpha

@torch.no_grad()
def marching_tetrahedra_with_binary_search(model_path, name, iteration, views, gaussians, pipeline, background, 
                                           kernel_size, add_internal_point, depth_threshold, bias, scale_ratio, filter_mesh : bool, texture_mesh : bool, near : float, far : float, bounds : None, bound_scale_ratio : float):
    render_path = os.path.join(model_path, "fusion_big")

    makedirs(render_path, exist_ok=True)

    # generate tetra points here
    points = gaussians.get_tetra_points(views, bounds, bound_scale_ratio, scale_ratio, add_internal_point, near, far)
    # colud = trimesh.points.PointCloud(points.cpu().numpy())
    # colud.export(os.path.join(render_path, "points.ply"))

    # alpha = evaluage_alpha(points, views, gaussians, pipeline, background, kernel_size, depth_threshold)
    # new_alpha = alpha.reshape(-1, 9)
    # print(new_alpha.shape)
    # print(torch.sum(torch.sum(new_alpha < 0.501, dim=-1) > 0))
    
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
    # print('--------------------0')
    # print(end_points.shape)

    faces=faces_list[0].cpu().numpy()
    points = (end_points[:, 0, :] + end_points[:, 1, :]) / 2.
    # points = end_points[:, 1, :]
        
    left_points = end_points[:, 0, :]
    right_points = end_points[:, 1, :]
    left_sdf = end_sdf[:, 0, :]
    right_sdf = end_sdf[:, 1, :]
    # distance = torch.norm(left_points - right_points, dim=-1)
    
    # step = 0
    n_binary_steps = 1
    for step in range(n_binary_steps):
        print("binary search in step {}".format(step))
        mid_points = (left_points + right_points) / 2
        alpha = evaluage_alpha(mid_points, views, gaussians, pipeline, background, kernel_size, depth_threshold, bias)
        mid_sdf = alpha_to_sdf(alpha).squeeze().unsqueeze(-1)
        print(alpha.shape)
        
        ind_low = ((mid_sdf < 0) & (left_sdf < 0)) | ((mid_sdf > 0) & (left_sdf > 0))

        left_sdf[ind_low] = mid_sdf[ind_low]
        right_sdf[~ind_low] = mid_sdf[~ind_low]
        left_points[ind_low.flatten()] = mid_points[ind_low.flatten()]
        right_points[~ind_low.flatten()] = mid_points[~ind_low.flatten()]
    
        points = (left_points + right_points) / 2
        
    # mesh = trimesh.Trimesh(vertices=points.cpu().numpy(), faces=faces, process=False)
    # mesh.export(os.path.join(render_path, f"mesh_{step}.ply"))

    # o3d_mesh = o3d.geometry.TriangleMesh()
    # o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    # o3d_mesh.triangles = o3d.utility.Vector3dVector(np.asarray(mesh.faces))

    # o3d_mesh = o3d.io.read_triangle_mesh(os.path.join(os.path.join(model_path, "fusion_big"), "mesh_0r.ply"))
    # o3d_mesh.compute_vertex_normals()
    # o3d_mesh_smoothed = o3d_mesh.filter_smooth_laplacian(number_of_iterations=3, lambda_filter=0.5)
    # o3d_mesh_smoothed.compute_vertex_normals()

    # o3d.io.write_triangle_mesh(os.path.join(os.path.join(os.path.join(model_path, "fusion_big"), "mesh_smooth.ply")), o3d_mesh_smoothed, 
    #                                    write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)

def extract_mesh(dataset : ModelParams, iterations : int, pipeline : PipelineParams, depth_threshold : float, filter_mesh : bool, texture_mesh : bool, near : float, far : float):
    update_config(dataset, None, True, args.cfg_path)
    dataset.preload_grayImg = False
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iterations, shuffle=False)
        
        # gaussians.load_ply(os.path.join(dataset.model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply"))
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size
        
        cams = scene.getTrainCameras()
        # gaussians.compute_3D_filter(cams)
        marching_tetrahedra_with_binary_search(dataset.model_path, "test", iterations, cams, gaussians, pipeline, 
                                               background, kernel_size, dataset.add_internal_point, depth_threshold, dataset.bias, dataset.scale_ratio, filter_mesh, texture_mesh, near, far, scene.bounds, dataset.bound_scale_ratio)

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
    parser.add_argument('-DT', "--depth_threshold", default=4, type=float)
    # parser.add_argument('-B', "--bias", default=0, type=float)
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
    
    extract_mesh(model.extract(args), args.iterations, pipeline.extract(args), args.depth_threshold, args.filter_mesh, args.texture_mesh, args.near, args.far)