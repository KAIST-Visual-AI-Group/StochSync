import torch
from pytorch3d.io import load_objs_as_meshes, load_obj, save_obj, IO

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
	look_at_view_transform,
	FoVPerspectiveCameras, 
	FoVOrthographicCameras,
	AmbientLights,
	RasterizationSettings, 
	MeshRenderer, 
	MeshRasterizer,  
	TexturesUV
)

from .geometry import HardGeometryShader
from .shader import HardNChannelFlatShader
from .voronoi import voronoi_solve


# Pytorch3D based renderering functions, managed in a class
# Render size is recommended to be the same as your latent view size
# DO NOT USE "bilinear" sampling when you are handling latents.
# Stable Diffusion has 4 latent channels so use channels=4

class Renderer():
	def __init__(self, texture_size=96, sampling_mode="nearest", channels=3, device=None):
		self.channels = channels
		self.device = device
		self.lights = AmbientLights(ambient_color=((1.0,)*channels,), device=self.device)
		self.target_size = (texture_size,texture_size)
		#self.render_size = render_size
		self.sampling_mode = sampling_mode


	# Load obj mesh, rescale the mesh to fit into the bounding box
	def load_mesh(self, mesh_path, scale_factor=2.0, auto_center=True, autouv=False):
		mesh = load_objs_as_meshes([mesh_path], device=self.device)
		if auto_center:
			verts = mesh.verts_packed()
			faces = mesh.faces_packed()
			center = verts.mean(dim=0)
			verts = verts - center
			scale = torch.max(torch.norm(verts, p=2, dim=1))
			verts = verts / scale
			verts *= scale_factor # [-1.1933e-11,  7.1746e-10,  3.3919e-09], max : 0.4190
			# mesh.vertices = verts
			mesh = Meshes(verts=verts.unsqueeze(0), faces=faces.unsqueeze(0), textures=mesh.textures)

		else:
			mesh.scale_verts_((scale_factor))

		if autouv or (mesh.textures is None):
			print("Auto uv using Xatlas")
			mesh = self.uv_unwrap(mesh)

		self.mesh = mesh


	def load_glb_mesh(self, mesh_path, scale_factor=2.0, auto_center=True, autouv=False, scene_mesh=False):
		from pytorch3d.io.experimental_gltf_io import MeshGlbFormat
		io = IO()
		io.register_meshes_format(MeshGlbFormat())
		with open(mesh_path, "rb") as f:
			mesh = io.load_mesh(f, include_textures=True, device=self.device)
		
		if auto_center:
			verts = mesh.verts_packed()
			faces = mesh.faces_packed()
			center = verts.mean(dim=0)
			verts = verts - center
			scale = torch.max(torch.norm(verts, p=2, dim=1))
			verts = verts / scale
			verts *= scale_factor # [-1.1933e-11,  7.1746e-10,  3.3919e-09], max : 0.4190
			mesh = Meshes(verts=verts.unsqueeze(0), faces=faces.unsqueeze(0), textures=mesh.textures)

		else:
			mesh.scale_verts_((scale_factor))

		if autouv or (mesh.textures is None):
			mesh = self.uv_unwrap(mesh)
		self.mesh = mesh


	def load_ply_mesh(self, mesh_path, scale_factor=2.0, auto_center=True, autouv=True, scene_mesh=False):
		io = IO()
		with open(mesh_path, "rb") as f:
			mesh = io.load_mesh(f, device=self.device)
		
		if auto_center:
			verts = mesh.verts_packed()
			max_bb = (verts - 0).max(0)[0]
			min_bb = (verts - 0).min(0)[0]
			scale = (max_bb - min_bb).max()/2 
			center = (max_bb+min_bb) /2
			mesh.offset_verts_(-center)
			mesh.scale_verts_((scale_factor / float(scale)))
		else:
			mesh.scale_verts_((scale_factor))

		if autouv: # Skip even if it does not have texture map. 2024.01.14
			mesh = self.uv_unwrap(mesh)
		
		self.mesh = mesh



	# Save obj mesh
	def save_mesh(self, mesh_path, texture):
		save_obj(mesh_path, 
				self.mesh.verts_list()[0],
				self.mesh.faces_list()[0],
				verts_uvs= self.mesh.textures.verts_uvs_list()[0],
				faces_uvs= self.mesh.textures.faces_uvs_list()[0],
				texture_map=texture)

	# Code referred to TEXTure code (https://github.com/TEXTurePaper/TEXTurePaper.git)
	def uv_unwrap(self, mesh):
		verts_list = mesh.verts_list()[0]
		faces_list = mesh.faces_list()[0]

		import xatlas
		import numpy as np
		v_np = verts_list.cpu().numpy()
		f_np = faces_list.int().cpu().numpy()
		atlas = xatlas.Atlas()
		atlas.add_mesh(v_np, f_np)
		chart_options = xatlas.ChartOptions()
		chart_options.max_iterations = 4
		atlas.generate(chart_options=chart_options)
		vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]

		vt = torch.from_numpy(vt_np.astype(np.float32)).type(verts_list.dtype).to(mesh.device)
		ft = torch.from_numpy(ft_np.astype(np.int64)).type(faces_list.dtype).to(mesh.device)

		new_map = torch.zeros(self.target_size+(self.channels,), device=mesh.device)
		new_tex = TexturesUV(
			[new_map], 
			[ft], 
			[vt], 
			sampling_mode=self.sampling_mode
			)

		mesh.textures = new_tex
		return mesh


	'''
		A functions that disconnect faces in the mesh according to
		its UV seams. The number of vertices are made equal to the
		number of unique vertices its UV layout, while the faces list
		is intact.
	'''
	def disconnect_faces(self):
		mesh = self.mesh
		verts_list = mesh.verts_list()
		faces_list = mesh.faces_list()
		verts_uvs_list = mesh.textures.verts_uvs_list()
		faces_uvs_list = mesh.textures.faces_uvs_list()
		packed_list = [v[f] for v,f in zip(verts_list, faces_list)]
		verts_disconnect_list = [
			torch.zeros(
				(verts_uvs_list[i].shape[0], 3), 
				dtype=verts_list[0].dtype, 
				device=verts_list[0].device
			) 
			for i in range(len(verts_list))]
		for i in range(len(verts_list)):
			verts_disconnect_list[i][faces_uvs_list] = packed_list[i]
		assert not mesh.has_verts_normals(), "Not implemented for vertex normals"
		self.mesh_d = Meshes(verts_disconnect_list, faces_uvs_list, mesh.textures)
		return self.mesh_d


	'''
		A function that construct a temp mesh for back-projection.
		Take a disconnected mesh and a rasterizer, the function calculates
		the projected faces as the UV, as use its original UV with pseudo
		z value as world space geometry.
	'''
	def construct_uv_mesh(self):
		mesh = self.mesh_d
		verts_list = mesh.verts_list()
		verts_uvs_list = mesh.textures.verts_uvs_list()
		# faces_list = [torch.flip(faces, [-1]) for faces in mesh.faces_list()]
		new_verts_list = []
		for i, (verts, verts_uv) in enumerate(zip(verts_list, verts_uvs_list)):
			verts = verts.clone()
			verts_uv = verts_uv.clone()
			verts[...,0:2] = verts_uv[...,:]
			verts = (verts - 0.5) * 2
			verts[...,2] *= 1
			new_verts_list.append(verts)
		textures_uv = mesh.textures.clone()
		self.mesh_uv = Meshes(new_verts_list, mesh.faces_list(), textures_uv)
		return self.mesh_uv


	# Set texture for the current mesh.
	def set_texture_map(self, texture):
		new_map = texture.permute(1, 2, 0)
		new_map = new_map.to(self.device)

		# Load faces/verts to device 
		new_tex = TexturesUV(
			[new_map], 
			self.mesh.textures.faces_uvs_padded().to(self.device), 
			self.mesh.textures.verts_uvs_padded().to(self.device), 
			sampling_mode=self.sampling_mode
			)
		self.mesh.textures = new_tex


	# Set the initial normal noise texture
	# No generator here for replication of the experiment result. Add one as you wish
	def set_noise_texture(self, channels=None):
		if not channels:
			channels = self.channels
		noise_texture = torch.normal(0, 1, (channels,) + self.target_size, device=self.device)
		self.set_texture_map(noise_texture)
		return noise_texture


	# Set the cameras given the camera poses and centers
	def set_cameras(self, camera_poses, centers=None, camera_distance=2.7, scale=None, scene_mesh=False):
		if scene_mesh:
			# Scene view: outward directions from the center
			R = torch.tensor(camera_poses)[:, :3, :3]
			T = torch.tensor(camera_poses)[:, :3, -1]
		else:
			# Object centric view
			elev = torch.FloatTensor([pose[0] for pose in camera_poses])
			azim = torch.FloatTensor([pose[1] for pose in camera_poses])
			# R: B, 3, 3 
			# T: B, 3
			R, T = look_at_view_transform(dist=camera_distance, elev=elev, azim=azim, at=centers or ((0,0,0),))

		self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T)
		# self.cameras = FoVOrthographicCameras(device=self.device, R=R, T=T, scale_xyz=scale or ((1,1,1),))


	# Set all necessary internal data for rendering and texture baking
	# Can be used to refresh after changing camera positions
	def set_cameras_and_render_settings(self, camera_poses, centers=None, camera_distance=2.7, render_size=None, scale=None, scene_mesh=False, fill=True):
		self.set_cameras(camera_poses, centers, camera_distance, scale=scale, scene_mesh=scene_mesh)
		if render_size is None:
			render_size = self.render_size
		if not hasattr(self, "renderer"):
			self.setup_renderer(size=render_size)
		if not hasattr(self, "mesh_d"):
			self.disconnect_faces()
		if not hasattr(self, "mesh_uv"):
			self.construct_uv_mesh()
		self.calculate_tex_gradient()
		self.calculate_visible_triangle_mask()
		_,_,_,cos_maps,_, _ = self.render_geometry()
		self.calculate_cos_angle_weights(cos_maps, fill=fill)


	# Setup renderers for rendering
	# max faces per bin set to 30000 to avoid overflow in many test cases.
	# You can use default value to let pytorch3d handle that for you.
	def setup_renderer(self, size=64, blur=0.0, face_per_pix=1, perspective_correct=False, channels=None):
		if not channels:
			channels = self.channels
		
		self.raster_settings = RasterizationSettings(
			image_size=size, 
			blur_radius=blur, 
			faces_per_pixel=face_per_pix,
			perspective_correct=perspective_correct,
			cull_backfaces=True,
			max_faces_per_bin=30000,
			bin_size=-1, 
		)

		self.renderer = MeshRenderer(
			rasterizer=MeshRasterizer(
				cameras=None, 
				raster_settings=self.raster_settings,
			),
			shader=HardNChannelFlatShader(
				device=self.device, 
				cameras=None,
				lights=self.lights,
				channels=channels
				# materials=materials
			)
		)
	
	def set_image_size(self, size):
		self.raster_settings.image_size = size


	# Normalize absolute depth to inverse depth
	@torch.no_grad()
	def decode_normalized_depth(self, depths, batched_norm=False):
		view_z, mask = depths.unbind(-1)
		view_z = view_z * mask + 100 * (1-mask)
		inv_z = 1 / view_z
		inv_z_min = inv_z * mask + 100 * (1-mask)
		if not batched_norm:
			max_ = torch.max(inv_z, 1, keepdim=True)
			max_ = torch.max(max_[0], 2, keepdim=True)[0]

			min_ = torch.min(inv_z_min, 1, keepdim=True)
			min_ = torch.min(min_[0], 2, keepdim=True)[0]
		else:
			max_ = torch.max(inv_z)
			min_ = torch.min(inv_z_min)
		inv_z = (inv_z - min_) / (max_ - min_)
		inv_z = inv_z.clamp(0,1)
		inv_z = inv_z[...,None].repeat(1,1,1,3)

		return inv_z



	# Render the current mesh and texture from current cameras
	def render_textured_views(self, cameras):
		meshes = self.mesh.extend(len(cameras))
		images_predicted = self.renderer(meshes, cameras=cameras, lights=self.lights)

		return [image.permute(2, 0, 1) for image in images_predicted]


	# Move the internel data to a specific device
	def to(self, device):
		for mesh_name in ["mesh", "mesh_d", "mesh_uv"]:
			if hasattr(self, mesh_name):
				mesh = getattr(self, mesh_name)
				setattr(self, mesh_name, mesh.to(device))
		for list_name in ["visible_triangles", "visibility_maps", "cos_maps"]:
			if hasattr(self, list_name):
				map_list = getattr(self, list_name)
				for i in range(len(map_list)):
					map_list[i] = map_list[i].to(device)
