'''
@file   pytorch3d_utils.py
@author Yiming Wang <w752531540@gmail.com>
'''

import pytorch3d
import torch
import trimesh
import sys
import cv2
import math

import numpy as np

def make_rotate(rx, ry, rz):

    sinX = np.sin(rx)
    sinY = np.sin(ry)
    sinZ = np.sin(rz)

    cosX = np.cos(rx)
    cosY = np.cos(ry)
    cosZ = np.cos(rz)

    Rx = np.zeros((3, 3))
    Rx[0, 0] = 1.0
    Rx[1, 1] = cosX
    Rx[1, 2] = -sinX
    Rx[2, 1] = sinX
    Rx[2, 2] = cosX

    Ry = np.zeros((3, 3))
    Ry[0, 0] = cosY
    Ry[0, 2] = sinY
    Ry[1, 1] = 1.0
    Ry[2, 0] = -sinY
    Ry[2, 2] = cosY

    Rz = np.zeros((3, 3))
    Rz[0, 0] = cosZ
    Rz[0, 1] = -sinZ
    Rz[1, 0] = sinZ
    Rz[1, 1] = cosZ
    Rz[2, 2] = 1.0

    R = np.matmul(np.matmul(Rz, Ry), Rx)
    return R

def normalize_v3(arr):
    ''' Normalize a numpy array of 3 component vectors shape=(n,3) '''
    lens = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2)
    eps = 0.00000001
    lens[lens < eps] = eps
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr

def compute_normal(vertices, faces):
    # Create a zeroed array with the same type and shape as our vertices i.e., per vertex normal
    norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    norm[faces[:, 0]] += n
    norm[faces[:, 1]] += n
    norm[faces[:, 2]] += n
    normalize_v3(norm)

    return norm



# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

from this import d
from pytorch3d.renderer import (
    BlendParams, blending, look_at_view_transform, FoVOrthographicCameras, PerspectiveCameras,
    PointLights, RasterizationSettings, PointsRasterizationSettings,
    PointsRenderer, AlphaCompositor, PointsRasterizer, MeshRenderer,
    MeshRasterizer, SoftPhongShader, SoftSilhouetteShader, TexturesVertex)
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer.mesh.renderer import MeshRendererWithFragments
from pytorch3d.renderer.mesh.shader import (
    BlendParams,
    HardFlatShader,
    HardGouraudShader,
    HardPhongShader,
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturedSoftPhongShader,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)

from pytorch3d.renderer.materials import Materials
# from lib.dataset.mesh_util import SMPLX, get_visibility

import torch
import numpy as np
from PIL import Image
from pytorch3d.io import load_obj
import os
import sys
import cv2
import math
import trimesh
import math
from typing import NewType
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import rasterize_meshes
from termcolor import colored
from torch import nn

# sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))

Tensor = NewType('Tensor', torch.Tensor)


def solid_angles(points: Tensor,
                 triangles: Tensor,
                 thresh: float = 1e-8) -> Tensor:
    ''' Compute solid angle between the input points and triangles
        Follows the method described in:
        The Solid Angle of a Plane Triangle
        A. VAN OOSTEROM AND J. STRACKEE
        IEEE TRANSACTIONS ON BIOMEDICAL ENGINEERING,
        VOL. BME-30, NO. 2, FEBRUARY 1983
        Parameters
        -----------
            points: BxQx3
                Tensor of input query points
            triangles: BxFx3x3
                Target triangles
            thresh: float
                float threshold
        Returns
        -------
            solid_angles: BxQxF
                A tensor containing the solid angle between all query points
                and input triangles
    '''
    # Center the triangles on the query points. Size should be BxQxFx3x3
    centered_tris = triangles[:, None] - points[:, :, None, None]

    # BxQxFx3
    norms = torch.norm(centered_tris, dim=-1)

    # Should be BxQxFx3
    cross_prod = torch.cross(centered_tris[:, :, :, 1],
                             centered_tris[:, :, :, 2],
                             dim=-1)
    # Should be BxQxF
    numerator = (centered_tris[:, :, :, 0] * cross_prod).sum(dim=-1)
    del cross_prod

    dot01 = (centered_tris[:, :, :, 0] * centered_tris[:, :, :, 1]).sum(dim=-1)
    dot12 = (centered_tris[:, :, :, 1] * centered_tris[:, :, :, 2]).sum(dim=-1)
    dot02 = (centered_tris[:, :, :, 0] * centered_tris[:, :, :, 2]).sum(dim=-1)
    del centered_tris

    denominator = (norms.prod(dim=-1) + dot01 * norms[:, :, :, 2] +
                   dot02 * norms[:, :, :, 1] + dot12 * norms[:, :, :, 0])
    del dot01, dot12, dot02, norms

    # Should be BxQ
    solid_angle = torch.atan2(numerator, denominator)
    del numerator, denominator

    torch.cuda.empty_cache()

    return 2 * solid_angle


def winding_numbers(points: Tensor,
                    triangles: Tensor,
                    thresh: float = 1e-8) -> Tensor:
    ''' Uses winding_numbers to compute inside/outside
        Robust inside-outside segmentation using generalized winding numbers
        Alec Jacobson,
        Ladislav Kavan,
        Olga Sorkine-Hornung
        Fast Winding Numbers for Soups and Clouds SIGGRAPH 2018
        Gavin Barill
        NEIL G. Dickson
        Ryan Schmidt
        David I.W. Levin
        and Alec Jacobson
        Parameters
        -----------
            points: BxQx3
                Tensor of input query points
            triangles: BxFx3x3
                Target triangles
            thresh: float
                float threshold
        Returns
        -------
            winding_numbers: BxQ
                A tensor containing the Generalized winding numbers
    '''
    # The generalized winding number is the sum of solid angles of the point
    # with respect to all triangles.
    return 1 / (4 * math.pi) * solid_angles(points, triangles,
                                            thresh=thresh).sum(dim=-1)


def batch_contains(verts, faces, points):

    B = verts.shape[0]
    N = points.shape[1]

    verts = verts.detach().cpu()
    faces = faces.detach().cpu()
    points = points.detach().cpu()
    contains = torch.zeros(B, N)

    for i in range(B):
        contains[i] = torch.as_tensor(
            trimesh.Trimesh(verts[i], faces[i]).contains(points[i]))

    return 2.0 * (contains - 0.5)


def dict2obj(d):
    # if isinstance(d, list):
    #     d = [dict2obj(x) for x in d]
    if not isinstance(d, dict):
        return d

    class C(object):
        pass

    o = C()
    for k in d:
        o.__dict__[k] = dict2obj(d[k])
    return o


def face_vertices(vertices, faces):
    """ 
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) *
                     nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, vertices.shape[-1]))
    
    return vertices[faces.long()]


class Pytorch3dRasterizer(nn.Module):
    """  Borrowed from https://github.com/facebookresearch/pytorch3d
    Notice:
        x,y,z are in image space, normalized
        can only render squared image now
    """
    def __init__(self, image_size=224):
        """
        use fixed raster_settings for rendering faces
        """
        super().__init__()
        raster_settings = {
            'image_size': image_size,
            'blur_radius': 0.0,
            'faces_per_pixel': 1,
            'bin_size': None,
            'max_faces_per_bin': None,
            'perspective_correct': True,
            'cull_backfaces': True,
        }
        raster_settings = dict2obj(raster_settings)
        self.raster_settings = raster_settings

    def forward(self, vertices, faces, attributes=None):
        fixed_vertices = vertices.clone()
        fixed_vertices[..., :2] = -fixed_vertices[..., :2]
        meshes_screen = Meshes(verts=fixed_vertices.float(),
                               faces=faces.long())
        raster_settings = self.raster_settings
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=raster_settings.image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )
        vismask = (pix_to_face > -1).float()
        D = attributes.shape[-1]
        attributes = attributes.clone()
        attributes = attributes.view(attributes.shape[0] * attributes.shape[1],
                                     3, attributes.shape[-1])
        N, H, W, K, _ = bary_coords.shape
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
        pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
        pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
        pixel_vals = torch.cat(
            [pixel_vals, vismask[:, :, :, 0][:, None, :, :]], dim=1)
        return pixel_vals


class cleanShader(torch.nn.Module):
    def __init__(self, device="cpu", cameras=None, blend_params=None):
        super().__init__()
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams(
        )

    def forward(self, fragments, meshes, **kwargs):
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of TexturedSoftPhongShader"

            raise ValueError(msg)

        # get renderer output
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = blending.softmax_rgb_blend(texels,
                                            fragments,
                                            blend_params,
                                            znear=-256,
                                            zfar=256)

        return images

def parse_extrinsics(extrinsics, world2camera=True):
    """ this function is only for numpy for now"""
    if extrinsics.shape[0] == 3 and extrinsics.shape[1] == 4:
        extrinsics = np.vstack([extrinsics, np.array([[0, 0, 0, 1.0]])])
    if extrinsics.shape[0] == 1 and extrinsics.shape[1] == 16:
        extrinsics = extrinsics.reshape(4, 4)
    if world2camera:
        extrinsics = np.linalg.inv(extrinsics).astype(np.float32)
    return extrinsics

def load_intrinsics(filepath, resized_width=None, invert_y=False):
    try:
        intrinsics = load_matrix(filepath)
        if intrinsics.shape[0] == 3 and intrinsics.shape[1] == 3:
            _intrinsics = np.zeros((4, 4), np.float32)
            _intrinsics[:3, :3] = intrinsics
            _intrinsics[3, 3] = 1
            intrinsics = _intrinsics
        if intrinsics.shape[0] == 1 and intrinsics.shape[1] == 16:
            intrinsics = intrinsics.reshape(4, 4)
        return intrinsics
    except ValueError:
        pass

    # Get camera intrinsics
    with open(filepath, 'r') as file:
        
        f, cx, cy, _ = map(float, file.readline().split())
    fx = f
    if invert_y:
        fy = -f
    else:
        fy = f

    # Build the intrinsic matrices
    full_intrinsic = np.array([[fx, 0., cx, 0.],
                               [0., fy, cy, 0],
                               [0., 0, 1, 0],
                               [0, 0, 0, 1]])
    return full_intrinsic

def load_matrix(path):
    lines = [[float(w) for w in line.strip().split()] for line in open(path)]
    if len(lines[0]) == 2:
        lines = lines[1:]
    if len(lines[-1]) == 2:
        lines = lines[:-1]
    return np.array(lines).astype(np.float32)

class Render():
    def __init__(self, size=512, device=torch.device("cuda:0")):
        self.device = device
        self.mesh_y_center = 100.0
        self.dis = 100.0
        self.scale = 1.0
        self.size = size
        self.cam_pos = [(0, 100, 100)]

        self.mesh = None
        self.deform_mesh = None
        self.pcd = None
        self.renderer = None
        self.meshRas = None
        self.type = None
        self.knn = None
        self.knn_inverse = None

        self.smpl_seg = None
        self.smpl_cmap = None

        # self.smplx = SMPLX()

        self.uv_rasterizer = Pytorch3dRasterizer(self.size)
        

    def get_camera(self, cam_id):
        R, T = look_at_view_transform(eye=[self.cam_pos[cam_id]],
                                      at=((0, self.mesh_y_center, 0), ),
                                      up=((0, 1, 0), ),
                                      device = self.device
                                      )

        camera = FoVOrthographicCameras(device=self.device,
                                        R=R,
                                        T=T,
                                        znear=100.0,
                                        zfar=-100.0,
                                        max_y=100.0,
                                        min_y=-100.0,
                                        max_x=100.0,
                                        min_x=-100.0,
                                        scale_xyz=(self.scale * np.ones(3), ))

        return camera
    
    def get_perspective_camera(self, camera):
        R = camera['R'].permute(0,2,1)
        # R[0, 0, 0] *= -1.0
        T = camera['T']
        image_size= self.size
        half_size = (image_size - 1.0) / 2
        focal = camera['focal']
        focal_length = focal/half_size
        principle = camera['principle']
        principal_point = []
        for i in range(principle.shape[0]):
            principal_point.append([(half_size - principle[i,0]) / half_size, \
                    (-principle[i,1] + half_size) / half_size])
        principal_point = torch.tensor(principal_point, dtype=torch.float32).reshape(principle.shape[0], 2).to(self.device)
        # principal_point = torch.tensor([(principle[0,0]) / half_size, \
                # (principle[0,1]) / half_size], dtype=torch.float32).reshape(1, 2).to(self.device)
        camera = PerspectiveCameras(device=self.device,
                                        R=R,
                                        T=T,
                                        focal_length = -focal_length,
                                        principal_point = principal_point,
                                        image_size=image_size)
        return camera

    def init_renderer(self, camera, type='clean_mesh', bg='gray'):

        if 'mesh' in type:
            
            # rasterizer
            self.raster_settings_mesh = RasterizationSettings(
                image_size=self.size,
                blur_radius=np.log(1.0 / 1e-4) * 1e-7,
                faces_per_pixel=30,
                perspective_correct=True
            )
            # self.raster_settings_mesh  = RasterizationSettings(
            #     image_size=self.size,  #设置输出图像的大小
            #     blur_radius=0.0, #由于只是为了可视化目的而渲染图像
            #     faces_per_pixel=1 #所以设置faces_per_pixel=1和blur_radius=0.0
            # )
            self.meshRas = MeshRasterizer(cameras=camera,
                                        raster_settings=self.raster_settings_mesh)
            
        if bg  == 'black':
            blendparam = BlendParams(1e-4, 1e-4, (0.0, 0.0, 0.0))
        elif bg == 'white':
            blendparam = BlendParams(1e-4, 1e-8, (1.0, 1.0, 1.0))
        elif bg == 'gray':
            blendparam = BlendParams(1e-4, 1e-8, (0.5, 0.5, 0.5))
            
        if type == 'ori_mesh':
            
            # lights = PointLights(device=self.device,
            #                  ambient_color=((0.8, 0.8, 0.8), ),
            #                  diffuse_color=((0.2, 0.2, 0.2), ),
            #                  specular_color=((0.0, 0.0, 0.0), ),
            #                 #  location=[[0.0, 200.0, 0.0]])
            #                  location=[[0.0, 1.0, 5.0]])
            lights = PointLights(device=self.device,
                            #  ambient_color=((0.8, 0.8, 0.8), ),
                            #  diffuse_color=((0.2, 0.2, 0.2), ),
                            #  specular_color=((0.0, 0.0, 0.0), ),
                            #  location=[[0.0, 200.0, 0.0]])
                             location=[[0.0, 0.0, 2.0]])
            
            self.renderer = MeshRenderer(rasterizer=self.meshRas,
                                     shader=SoftPhongShader(
                                         device=self.device,
                                         cameras=camera,
                                         lights=lights,
                                         blend_params=blendparam))

        if type == 'silhouette':
            self.raster_settings_silhouette = RasterizationSettings(
                image_size=self.size,
                blur_radius=np.log(1. / 1e-4 - 1.) * 5e-5,
                faces_per_pixel=50,
                cull_backfaces=True,
                perspective_correct=False  # This solve the problem of the grad nan error 
            )

            self.silhouetteRas = MeshRasterizer(
                cameras=camera, raster_settings=self.raster_settings_silhouette
                )
            self.renderer = MeshRenderer(rasterizer=self.silhouetteRas,
                                                shader=SoftSilhouetteShader())

        if type == 'zbuf':
            self.raster_settings_silhouette = RasterizationSettings(
                image_size=self.size,
                blur_radius=np.log(1. / 1e-4 - 1.) * 5e-5,
                faces_per_pixel=1,
                bin_size=None,
                max_faces_per_bin = None,
                cull_backfaces=True,
                perspective_correct=True  # This solve the problem of the grad nan error 
            )

            self.silhouetteRas = MeshRasterizer(
                cameras=camera, raster_settings=self.raster_settings_silhouette
                )
            self.renderer = MeshRendererWithFragments(rasterizer=self.silhouetteRas,
                                                shader=SoftSilhouetteShader())
            
        if type == 'pointcloud':
            self.raster_settings_pcd = PointsRasterizationSettings(
                image_size=self.size,
                radius=0.006,
                points_per_pixel=10)
            
            self.pcdRas = PointsRasterizer(cameras=camera,
                                        raster_settings=self.raster_settings_pcd)
            self.renderer = PointsRenderer(
                rasterizer=self.pcdRas,
                compositor=AlphaCompositor(background_color=(0, 0, 0)))


        if type == 'clean_mesh':
                
            self.renderer = MeshRenderer(
                rasterizer=self.meshRas,
                shader=cleanShader(device=self.device,
                                cameras=camera,
                                blend_params=blendparam))

        if type == 'textured_mesh':
            # lights = PointLights(device=self.device,
                            #  ambient_color=((0.0, 0.0, 0.0), ),
                            #  diffuse_color=((0.0, 0.0, 0.0), ),
                            #  specular_color=((0.0, 0.0, 0.0), ),
                            #  location=[[0.0, 200.0, 0.0]]).to(self.device)
            # lights = PointLights(device=self.device)

        # Place light behind the cow in world space. The front of
        # the cow is facing the -z direction.
            # lights.location = torch.tensor([0.0, 0.0, 2.0], device=self.device)[None]
            # materials = Materials(device=self.device).to(self.device)
            self.raster_settings_mesh = RasterizationSettings(
                image_size=self.size,
                blur_radius=0.0,
                faces_per_pixel=1,
                perspective_correct=True
            )
            
            self.renderer = MeshRenderer(
                rasterizer=MeshRasterizer(cameras=camera, raster_settings=self.raster_settings_mesh),
                shader = myShader(              
                    cameras= camera,
                    blend_params= blendparam)
                # shader=TexturedSoftPhongShader(
                #     lights=lights,
                #     materials = materials,
                #     cameras= camera,
                #     blend_params= blendparam,
                # ),
            )

        if type == 'shaded_mesh':
            raster_settings_silhouette = RasterizationSettings(
                image_size=(self.size,self.size), 
                blur_radius=0., 
                bin_size=int(2 ** max(np.ceil(np.log2(max(self.size,self.size))) - 4, 4)),
                faces_per_pixel=1,
                perspective_correct=True,
                clip_barycentric_coords=False,
                cull_backfaces=False
            )	
            self.renderer = MeshRendererWithFragments(
                rasterizer=MeshRasterizer(
                    cameras=camera, 
                    raster_settings=raster_settings_silhouette
                ),
                shader=SoftSilhouetteShader()
            )


        
        

    def load_mesh(self,
                  verts,
                  faces,
                  verts_rgb,
                  verts_dense=None,
                  load_seg=False):
        """load mesh into the pytorch3d renderer

        Args:
            verts ([N,3]): verts
            faces ([N,3]): faces
            verts_rgb ([N,3]): verts colors
            verts_dense ([N,3], optinoal): verts dense correspondense results. Defaults to None.
            load_seg (bool, optional): needs to render seg or not. Defaults to False.
        """

        self.type = type
        self.load_seg = load_seg

        # data format convert
        if not torch.is_tensor(verts):
            verts = torch.as_tensor(verts).float().unsqueeze(0).to(self.device)
            faces = torch.as_tensor(faces).int().unsqueeze(0).to(self.device)
            if verts_rgb is not None:
                verts_rgb = torch.as_tensor(
                    verts_rgb)[:, :3].float().unsqueeze(0).to(self.device)
        else:
            verts = verts.float().unsqueeze(0).to(self.device)
            faces = faces.int().unsqueeze(0).to(self.device)
            if verts_rgb is not None:
                verts_rgb = verts_rgb[:, :3].float().unsqueeze(0).to(
                    self.device)

        # dense correspondence results data format convert
        if verts_dense is not None:
            if not torch.is_tensor(verts_dense):
                verts_dense = torch.from_numpy(verts_dense)
            verts_dense = verts_dense[:, :3].unsqueeze(0).to(self.device)

        # camera setting
        self.mesh_y_center = (
            0.5 *
            (verts.max(dim=1)[0][0, 1] + verts.min(dim=1)[0][0, 1])).item()
        self.scale = 90.0 / (self.mesh_y_center + 1e-10)
        self.cam_pos = [(0, self.mesh_y_center, self.dis),
                        (self.dis, self.mesh_y_center, 0),
                        (0, self.mesh_y_center, -self.dis),
                        (-self.dis, self.mesh_y_center, 0)]

        # self.verts is for UV rendering, so it is [smpl_num, 3]
        # verts is for normal rendering, so it is [sample_num, 3]

        if verts_rgb is not None:
            self.type = 'color'
            textures = TexturesVertex(verts_features=verts_rgb)
            self.verts = verts_rgb.squeeze(0)[self.knn].squeeze(1)

        self.mesh = Meshes(verts=verts.to(self.device), faces=faces.to(self.device),
                           textures=textures.to(self.device)).to(self.device)

        # _, faces, aux = load_obj(self.smplx.tpose_path, device=self.device)
        # uvcoords = aux.verts_uvs[None, ...]  # (N, V, 2)
        # self.uvfaces = faces.textures_idx[None, ...]  # (N, F, 3)
        # self.verts = self.verts[None, ...]  # (N, V, 3)
        # self.faces = faces.verts_idx[None, ...]  # (N, F, 3)

        # # uv coords
        # uvcoords = torch.cat([uvcoords, uvcoords[:, :, 0:1] * 0. + 1.],
        #                      -1)  # [bz, ntv, 3]
        # self.uvcoords = uvcoords * 2 - 1
        # uvcoords[..., 1] = -uvcoords[..., 1]

    def load_simple_mesh(self, verts, faces, deform_verts=None):
        """load mesh into the pytorch3d renderer

        Args:
            verts ([N,3]): verts
            faces ([N,3]): faces
            offset ([N,3]): offset
        """

        # camera setting
        self.scale = 100.0
        self.mesh_y_center = 0.0

        self.cam_pos = [(0, self.mesh_y_center, 100.0),
                        (100.0, self.mesh_y_center, 0),
                        (0, self.mesh_y_center, -100.0),
                        (-100.0, self.mesh_y_center, 0)]

        self.type = 'color'

        if not torch.is_tensor(verts):
            verts = torch.tensor(verts)
        if not torch.is_tensor(faces):
            faces = torch.tensor(faces)

        if verts.ndimension() == 2:
            verts = verts.unsqueeze(0).float()
        if faces.ndimension() == 2:
            faces = faces.unsqueeze(0).long()

        verts = verts.to(self.device)
        faces = faces.to(self.device)

        # verts_rgb = (compute_normal_batch(verts, faces) + 1.0) * 0.5

        if deform_verts is not None:

            deform_verts_copy = deform_verts.clone()
            false_ids = torch.topk(torch.abs(deform_verts).sum(dim=1), 30)[1]
            deform_verts_copy[false_ids] = deform_verts_copy.mean(dim=0)

            self.mesh = Meshes(verts, faces).to(
                self.device).offset_verts(deform_verts_copy)
        else:
            self.mesh = Meshes(verts, faces).to(self.device)

        textures = TexturesVertex(
            verts_features=(self.mesh.verts_normals_padded() + 1.0) * 0.5)
        self.mesh.textures = textures

    def load_pcd(self, verts, verts_rgb):
        """load pointcloud into the pytorch3d renderer

        Args:
            verts ([N,3]): verts
            verts_rgb ([N,3]): verts colors
        """

        # data format convert
        if not torch.is_tensor(verts):
            verts = torch.as_tensor(verts).float().unsqueeze(0).to(self.device)
            if verts_rgb is not None:
                verts_rgb = torch.as_tensor(
                    verts_rgb)[:, :3].float().unsqueeze(0).to(self.device)
        else:
            verts = verts.float().unsqueeze(0).to(self.device)
            if verts_rgb is not None:
                verts_rgb = verts_rgb[:, :3].float().unsqueeze(0).to(
                    self.device)

        # camera setting
        self.mesh_y_center = (
            0.5 *
            (verts.max(dim=1)[0][0, 1] + verts.min(dim=1)[0][0, 1])).item()
        self.scale = 90.0 / (self.mesh_y_center + 1e-10)
        self.cam_pos = [(0, self.mesh_y_center, self.dis),
                        (self.dis, self.mesh_y_center, 0),
                        (0, self.mesh_y_center, -self.dis),
                        (-self.dis, self.mesh_y_center, 0)]

        pcd = Pointclouds(points=verts, features=verts_rgb).to(self.device)
        return pcd

    def get_image(self):
        images = torch.zeros(
            (self.size, self.size * len(self.cam_pos), 3)).to(self.device)
        for cam_id in range(len(self.cam_pos)):
            self.init_renderer(self.get_camera(cam_id), 'ori_mesh', 'gray')
            images[:, self.size * cam_id:self.size *
                   (cam_id + 1), :] = self.renderer(self.mesh)[0, :, :, :3]

        return images.cpu().numpy()

    def get_clean_image(self, cam_ids=[0, 2]):

        images = []
        for cam_id in range(len(self.cam_pos)):
            if cam_id in cam_ids:
                self.init_renderer(self.get_camera(cam_id), 'clean_mesh', 'gray')
                if len(cam_ids) == 4:
                    rendered_img = (self.renderer(
                        self.mesh)[0:1, :, :, :3].permute(0, 3, 1, 2) -
                                    0.5) * 2.0
                else:
                    rendered_img = (self.renderer(
                        self.mesh)[0:1, :, :, :3].permute(0, 3, 1, 2) -
                                    0.5) * 2.0
                if cam_id == 2 and len(cam_ids) == 2:
                    rendered_img = torch.flip(rendered_img, dims=[3])
                images.append(rendered_img)

        return images
        
    def get_perspective_image(self, camera):
    
        # self.init_renderer(self.get_perspective_camera(camera), 'clean_mesh', 'gray')
        self.init_renderer(self.get_perspective_camera(camera), 'clean_mesh', 'white')
        rendered_img = (self.renderer(self.mesh)[0, :, :, :3] - 0.5) * 2.0
     
        return rendered_img

    def get_textured_image(self, camera):
    
        # self.init_renderer(self.get_perspective_camera(camera), 'clean_mesh', 'gray')
        self.init_renderer(self.get_perspective_camera(camera), 'textured_mesh', 'white')
        # rendered_img = (self.renderer(self.mesh)[0, :, :, :3] - 0.5) * 2.0
        rendered_img = self.renderer(self.mesh)[0, :, :, :3]
     
        return rendered_img

    # def get_shaded_mesh(self, camera):
    
    #     # self.init_renderer(self.get_perspective_camera(camera), 'clean_mesh', 'gray')
    #     self.init_renderer(self.get_perspective_camera(camera), 'shaded_mesh', 'white')
    #     # rendered_img = (self.renderer(self.mesh)[0, :, :, :3] - 0.5) * 2.0
    #     # rendered_img = self.renderer(self.mesh)[0, :, :, :3]
    #     imgs,frags = self.renderer(self.mesh)
    #     import pdb
    #     pdb.set_trace()
    #     rendered_img = imgs[0, :, :, :3]
     
    #     return rendered_img

    def get_shaded_mesh(self, camera, lights = None):
    
        # self.init_renderer(self.get_perspective_camera(camera), 'clean_mesh', 'gray')
        # self.init_renderer(self.get_perspective_camera(camera), 'ori_mesh', 'white')
        # self.init_renderer(self.get_perspective_camera(camera), 'ori_mesh', 'gray')

        camera = self.get_perspective_camera(camera)

        blendparam = BlendParams(1e-4, 1e-4, (0.0, 0.0, 0.0))
        self.raster_settings_mesh = RasterizationSettings(
            image_size=self.size,
            blur_radius=np.log(1.0 / 1e-4) * 1e-7,
            faces_per_pixel=30,
            perspective_correct=True
        )
        # self.raster_settings_mesh  = RasterizationSettings(
        #     image_size=self.size,  #设置输出图像的大小
        #     blur_radius=0.0, #由于只是为了可视化目的而渲染图像
        #     faces_per_pixel=1 #所以设置faces_per_pixel=1和blur_radius=0.0
        # )
        self.meshRas = MeshRasterizer(cameras=camera,
                                    raster_settings=self.raster_settings_mesh)

        if lights == None:
            lights = PointLights(device=self.device,
                        #  ambient_color=((0.8, 0.8, 0.8), ),
                        #  diffuse_color=((0.2, 0.2, 0.2), ),
                        #  specular_color=((0.0, 0.0, 0.0), ),
                        #  location=[[0.0, 200.0, 0.0]])
                            location=[[0.0, 0.0, 2.0]])
                            
        self.renderer = MeshRenderer(rasterizer=self.meshRas,
                                    shader=SoftPhongShader(
                                        device=self.device,
                                        cameras=camera,
                                        lights=lights,
                                        blend_params=blendparam))
                                        
        # self.init_renderer(self.get_perspective_camera(camera), 'ori_mesh', 'black')
        # rendered_img = (self.renderer(self.mesh)[0, :, :, :3] - 0.5) * 2.0
        rendered_img = self.renderer(self.mesh)[0, :, :, :3]
     
        return rendered_img

    def get_all_perspective_image(self, camera):
        
        self.init_renderer(self.get_perspective_camera(camera), 'clean_mesh', 'gray')
        rendered_img = (self.renderer(self.mesh)[:, :, :, :3] - 0.5) * 2.0
     
        return rendered_img

    def get_rendered_video(self, images, save_path):

        self.cam_pos = []
        for angle in range(360):
            self.cam_pos.append(
                (100.0 * math.cos(np.pi / 180 * angle), self.mesh_y_center,
                 100.0 * math.sin(np.pi / 180 * angle)))

        old_shape = np.array(images[0].shape[:2])
        new_shape = np.around((self.size / old_shape[0]) * old_shape).astype(np.int)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(save_path, fourcc, 30, (self.size+new_shape[1]*len(images), self.size))
        
        print(colored(f"exporting video {os.path.basename(save_path)}, please wait for a while...", "blue"))
            
        for cam_id in range(len(self.cam_pos)):
            self.init_renderer(self.get_camera(cam_id), 'clean_mesh', 'gray')
            rendered_img = (self.renderer(
                self.mesh)[0, :, :, :3] * 255.0).detach().cpu().numpy().astype(np.uint8)
            img_lst = [np.array(Image.fromarray(img).resize(new_shape[::-1])).astype(np.uint8)[:,:,[2,1,0]] for img in images]
            img_lst.append(rendered_img)
            final_img = np.concatenate(img_lst,axis=1)
            video.write(final_img)
            
        video.release()

    def get_silhouette_image(self, cam_ids=[0, 2]):

        images = []
        for cam_id in range(len(self.cam_pos)):
            if cam_id in cam_ids:
                self.init_renderer(self.get_camera(cam_id), 'silhouette')
                rendered_img = self.renderer(self.mesh)[0:1, :, :,
                                                                   3]
                if cam_id == 2 and len(cam_ids) == 2:
                    rendered_img = torch.flip(rendered_img, dims=[2])
                images.append(rendered_img)

        return images

    def get_perspective_silhouette_image(self, camera):
    
        self.init_renderer(self.get_perspective_camera(camera), 'silhouette')
        rendered_img = self.renderer(self.mesh)[0, :, :, 3]
     
        return rendered_img
    
    def get_zbuf(self, camera):
        
        self.init_renderer(self.get_perspective_camera(camera), 'zbuf')
        images, fragments = self.renderer(self.mesh)
        zbuf = fragments.zbuf
        # cv2.imwrite('debug/pytorch3d_occlusion/silhouette.png',(images[0,:,:,3].detach().cpu().numpy()+1)/2*255.5)
        return zbuf

    def get_image_pcd(self, pcd):
        images = torch.zeros(
            (self.size, self.size * len(self.cam_pos), 3)).to(self.device)
        for cam_id in range(len(self.cam_pos)):
            self.init_renderer(self.get_camera(cam_id), 'pointcloud')
            images[:, self.size*cam_id:self.size*(cam_id+1), :] = \
                self.renderer(pcd)[0, :, :, :3]

        return images.cpu().numpy()

    def get_texture(self, smpl_color=None):
        '''
        warp vertices from world space to uv space
        vertices: [bz, V, 3]
        uv_vertices: [bz, 3, h, w]
        '''

        if self.type == 'color':
            assert smpl_color is not None, "smpl_color argument should not be empty"

        batch_size = self.verts.shape[0]
        face_vertices = face_vertices(
            self.verts, self.faces.expand(batch_size, -1, -1))
        uv_vertices = self.uv_rasterizer(
            self.uvcoords.expand(batch_size, -1, -1),
            self.uvfaces.expand(batch_size, -1, -1), face_vertices)[:, :3]
        uv_vertices = uv_vertices.squeeze(0).permute(1, 2, 0).cpu().numpy()

        if self.type == 'dense':
            face_vertices = face_vertices(
                self.smpl_cmap[None, ...],
                self.faces.expand(batch_size, -1, -1))
        elif self.type == 'color':
            face_vertices = face_vertices(
                smpl_color[:, :, :3].to(self.device),
                self.faces.expand(batch_size, -1, -1))
        else:
            face_vertices = face_vertices(
                self.smpl_seg[None, ...],
                self.faces.expand(batch_size, -1, -1))

        uv_vertices_cmap = self.uv_rasterizer(
            self.uvcoords.expand(batch_size, -1, -1),
            self.uvfaces.expand(batch_size, -1, -1), face_vertices)[:, :3]

        uv_vertices_cmap = uv_vertices_cmap.squeeze(0).permute(
            1, 2, 0).cpu().numpy()

        return np.concatenate(
            (np.flip(uv_vertices, 0), np.flip(uv_vertices_cmap, 0)), axis=1)

class myShader(torch.nn.Module):

    def __init__(
        self, device="cpu", cameras=None, blend_params=None
    ):
        super().__init__()
        self.cameras = cameras
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        cameras = kwargs.get("cameras", self.cameras)
        if cameras is None:
            msg = "Cameras must be specified either at initialization \
                or in the forward pass of TexturedSoftPhongShader"
            raise ValueError(msg)
        # get renderer output
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = softmax_rgb_blend(texels, fragments, blend_params)

        return images


def render_mesh(renderer, vertices, faces, ixt, ext, scale = 1.0, shaded = False, lights = None):
    # vertices = mesh.vertices.copy() 
    vertices /= scale
    R = make_rotate(0,math.radians(90),0) 
    # R = make_rotate(math.radians(90),0,0) 
    # R = make_rotate(math.radians(90), math.radians(90), math.radians(90)) 
    # R = make_rotate(0, 0, math.radians(10)) 
    # R = make_rotate(0, math.radians(0), 0) 
    # vertices = np.matmul(vertices, R.T)
    # vertices[:,0] *= -1
    normals = compute_normal(vertices, faces)
    # normals = normals @ ext[:3,:3].cpu().numpy()
    # self.renderer.load_mesh(verts=mesh.vertices,faces=faces,verts_rgb=normals*0.5+0.5)
    # self.renderer.load_mesh(verts=mesh.vertices/self.opt.scale,faces=faces,verts_rgb=normals*0.5+0.5)
    # self.renderer.load_mesh(verts=mesh.vertices/self.opt.scale,faces=faces,verts_rgb=normals*0.5+0.5)
    # renderer.load_mesh(verts=mesh.vertices,faces=faces,verts_rgb=torch.ones_like(torch.tensor(normals))*0.5)


    camera = {}
    camera['R'] = ext[:3,:3].unsqueeze(0).float()
    camera['T'] = ext[:3,3].unsqueeze(0).float()
    camera['focal'] = torch.tensor([ixt[0,0],ixt[1,1]]).unsqueeze(0).float()
    camera['principle'] = torch.tensor([ixt[0,2],ixt[1,2]]).unsqueeze(0).float()

    if shaded:
        renderer.load_mesh(verts=vertices,faces=faces,verts_rgb=torch.ones(vertices.shape))
        T_mask_F = renderer.get_shaded_mesh(camera=camera, lights = lights).cpu().numpy()*255
    else:
        renderer.load_mesh(verts=vertices,faces=faces,verts_rgb=normals*0.5+0.5)
        T_mask_F = renderer.get_textured_image(camera=camera).cpu().numpy()*255
    return T_mask_F