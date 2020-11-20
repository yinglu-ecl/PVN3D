#!/usr/bin/env python3
# tless_dataset.py
# ying 2020/10/15
# reference to pvn3d:linemod_dataset.py

import os
import cv2
# import pcl
import open3d as o3d
import torch
import os.path
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from common import Config
import pickle as pkl
from lib.utils.basic_utils import Basic_Utils
import yaml
import scipy.io as scio
import scipy.misc
from cv2 import imshow, waitKey
from lib.utils.tless_utils import * # this file should be added to lib

# DEBUG = False

class Tless_Dataset():
	def __init__(self, dataset_name, cls_type=1, debug=False):
		self.debug = debug
		self.config = Config(dataset_name='tless', cls_type=cls_type)
		self.bs_utils = Basic_Utils(self.config)
		self.dataset_name = dataset_name
		self.transcolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
		self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.224]) ## I've used the same for DF, but these are for linemod I guess
		# self.obj_dict = self.config.tless_obj_dict ## should be added in config

		self.cls_type = int(cls_type) # this is int
		self.cls_id = cls_type
		print("cls_id in tless_dataset.py: {}".format(self.cls_id))
		self.root = self.config.tless_root ## should verify
		self.rng = np.random
		if dataset_name == 'train':
			self.width = 400 # y size
			self.height = 400 # x size
			self.xmap = np.array([[j for i in range(400)] for j in range(400)])
			self.ymap = np.array([[i for i in range(400)] for j in range(400)])
			self.cls_root = os.path.join(self.root, 'train_primesense', '{:02d}'.format(self.cls_type))
			meta_file = os.path.join(self.cls_root,'gt.yml')
			self.meta = load_gt(meta_file)
			info_file = os.path.join(self.cls_root,'info.yml')
			self.info = load_info(info_file)
			self.add_noise = True
			#### real images
			self.real_tr_file_mask = os.path.join(self.root, 'train_primesense', '{:02d}'.format(self.cls_type), '{:s}', '{:04d}.png')
			self.real_list = list(self.meta.keys()) ## should be checked with get_item
			#### rendered images
			rnd_img_path = os.path.join(self.root, "renders/{}/file_list.txt".format(cls_type))
			try:
				self.rnd_list = self.bs_utils.read_lines(rnd_img_path)
			except: # no render data
				self.rnd_list = []
			#### fused images
			fuse_img_path = os.path.join(self.root, "fuse/{}/file_list.txt".format(cls_type))
			try:
				self.fuse_list = self.bs_utils.read_lines(fuse_img_path)
			except: # no fuse data
				self.fuse_list = self.rnd_list ## ??? why not be an empty list??
			self.all_list = self.real_list + self.rnd_list + self.fuse_list
		elif dataset_name == 'test' or dataset_name == 'val':
			self.add_noise = False
			self.width = 720 # y size
			self.height = 540 # x size
			self.xmap = np.array([[j for i in range(720)] for j in range(540)])
			self.ymap = np.array([[i for i in range(720)] for j in range(540)])
			self.test_root = os.path.join(self.root, 'test_primesense')
			self.test_file_mask = os.path.join(self.test_root, '{:02d}', '{:s}', '{:04d}.png')
			self.meta = {}
			self.info = {}
			self.test_list = []
			for i in range(20):
				meta = load_gt(os.path.join(self.test_root, '{:02d}'.format(i+1), 'gt.yml'))
				info = load_info(os.path.join(self.test_root, '{:02d}'.format(i+1), 'info.yml'))
				# from IPython import embed; embed()
				for im_id, gts in meta.items():
					for obj_ind in range(len(gts)):
						gt = gts[obj_ind]
						if gt['obj_id'] == self.cls_type:
							if i+1 in self.meta.keys():
								if im_id in self.meta[i+1].keys():
									self.meta[i+1][im_id].append(gt)
								else:
									self.meta[i+1][im_id] = [gt]
									self.info[i+1][im_id] = info[im_id]
							else:
								self.meta[i+1] = {}
								self.meta[i+1][im_id] = [gt]
								self.info[i+1] = {}
								self.info[i+1][im_id] = info[im_id]
							test_item = (i+1, im_id, len(self.meta[i+1][im_id])-1, obj_ind) # tuple: (scene_id, image_id, obj_id in current list, obj_id in org list)
							self.test_list.append(test_item)
			if dataset_name == 'test':
				self.all_list = self.test_list
			else:
				randind = np.random.permutation(len(self.test_list))
				# from IPython import embed; embed()
				self.all_list = [self.test_list[i] for i in randind[:100]]
		print('{} buffer loaded, total number: {}'.format(self.dataset_name, len(self.all_list)))
		# from IPython import embed; embed()
		# if debug:
		# 	print('self.meta', self.meta)
		# 	print('self.info', self.info)

	def real_syn_gen(self, real_ratio=0.3, fuse_ratio=0.4):
		# real_ratio = 0.3, fuse_ratio = 0.4*0.7, rnd_ratio = 0.6*0.7
		# generate a random data item from the list
		# if self.rng.rand() < real_ratio:
		# 	idx = self.rng.randint(0, len(self.real_list))
		# 	return self.real_list[idx]
		# elif self.rng.rand() < fuse_ratio:
		# 	idx = self.rng.randint(0, len(self.fuse_list))
		# 	return self.fuse_list[idx]
		# else:
		# 	idx = self.rng.randint(0, len(self.rnd_list))
		# 	return self.rnd_list[idx]
		rnd_ratio = (1-fuse_ratio)*(1-real_ratio)
		if self.rng.rand()<rnd_ratio and len(self.rnd_list)>0:
			idx = self.rng.randint(0, len(self.rnd_list))
			return self.rnd_list[idx]
		elif self.rng.rand()<fuse_ratio*(1-real_ratio)/(1-rnd_ratio) \
				and len(self.fuse_list)>0:
			idx = self.rng.randint(0, len(self.fuse_list))
			return self.fuse_list[idx]
		else:
			idx = self.rng.randint(0, len(self.real_list))
			return self.real_list[idx]

	def real_gen(self):
		# generate a real data item
		idx = self.rng.randint(0, len(self.real_list))
		return self.real_list[idx]

	def rand_range(self, rng, lo, hi):
		# generate a random number in a given range
		return rng.rand()*(hi-lo)+lo

	def gaussian_noise(self, rng, img, sigma):
		"""add gaussian noise of given sigma to image"""
		img = img + rng.randn(*img.shape) * sigma
		img = np.clip(img, 0, 255).astype('uint8')
		return img

	def linear_motion_blur(self, img, angle, length):
		""":param angle: in degree"""
		rad = np.deg2rad(angle)
		dx = np.cos(rad)
		dy = np.sin(rad)
		a = int(max(list(map(abs, (dx, dy)))) * length * 2)
		if a <= 0:
			return img
		kern = np.zeros((a, a))
		cx, cy = a // 2, a // 2
		dx, dy = list(map(int, (dx * length + cx, dy * length + cy)))
		cv2.line(kern, (cx, cy), (dx, dy), 1.0)
		s = kern.sum()
		if s == 0:
			kern[cx, cy] = 1.0
		else:
			kern /= s
		return cv2.filter2D(img, -1, kern)

	def rgb_add_noise(self, img):
		rng = self.rng
		# apply HSV augmentor
		if rng.rand() > 0:
			hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.uint16)
			hsv_img[:, : ,1] = hsv_img[:, :, 1] * self.rand_range(rng, 1-0.25, 1+.25)
			hsv_img[:, :, 2] = hsv_img[:, :, 2] * self.rand_range(rng, 1-.15, 1+.15)
			hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1], 0, 255)
			hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2], 0, 255)
			img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)

		if rng.rand() > 0.8:  # motion blur
			r_angle = int(rng.rand() * 360)
			r_len = int(rng.rand() * 15) + 1
			img = self.linear_motion_blur(img, r_angle, r_len)

		if rng.rand() > 0.8:
			if rng.rand() > 0.2:
				img = cv2.GaussianBlur(img, (3, 3), rng.rand())
			else:
				img = cv2.GaussianBlur(img, (5, 5), rng.rand())

		return np.clip(img, 0, 255).astype(np.uint8)

	# def get_normal(self, cld):
	# 	cloud = pcl.PointCloud()
	# 	cld = cld.astype(np.float32)
	# 	cloud.from_array(cld)
	# 	ne = cloud.make_NormalEstimation()
	# 	kdtree = cloud.make_kdtree()
	# 	ne.set_SearchMethod(kdtree)
	# 	ne.set_KSearch(50)
	# 	n = ne.compute()
	# 	n = n.to_array()
	# 	return n

	def get_normal(self, cld):
		cloud = o3d.geometry.PointCloud()
		cld = cld.astype(np.float32)
		cloud.points = o3d.utility.Vector3dVector(cld)
		cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=100, max_nn=50)) ## suppose distance on mm
		n = np.asarray(cloud.normals)
		return n

	def add_real_back(self, rgb, labels, dpt, dpt_msk):
		pass

	def get_item(self, item_name):
		# try:
			if type(item_name) is str and "pkl" in item_name:
				if self.debug: #DEBUG:
					print('loading render/fuse sample')
				# load render/fuse sample
				data = pkl.load(open(item_name, "rb"))
				dpt = data['depth']
				rgb = data['rgb']
				labels = data['mask']
				K = data['K']
				RT = data['RT']
				rnd_typ = data['rnd_typ']
				if rnd_typ == "fuse":
					labels = (labels == self.cls_id).astype("uint8")
				else:
					labels = (labels > 0).astype("uint8")
				cam_scale = 1.0
			elif type(item_name) is tuple:
				if self.debug: #DEBUG:
					print('loading test sample')
				# load test sample
				## get rgb image
				im_path = self.test_file_mask.format(item_name[0], 'rgb', item_name[1])
				with Image.open(im_path) as ri:
					if self.add_noise:
						ri=self.transcolor(ri)
					rgb = np.array(ri)[:,:,:3]
				## get depth image
				depth_path = self.test_file_mask.format(item_name[0], 'depth', item_name[1])
				with Image.open(depth_path) as di:
					dpt = np.array(di)
				## get mask image
				mask_path = self.test_file_mask.format(item_name[0], 'mask', item_name[1])
				with Image.open(mask_path) as li:
					labels = np.array(li)
					code = int(str(self.cls_type)+'{:03d}'.format(item_name[-1]))
					labels = (labels==code).astype(np.uint8)
				## get meta information
				# from IPython import embed; embed()
				meta = self.meta[item_name[0]][item_name[1]][item_name[2]]
				R = meta['cam_R_m2c']
				T = meta['cam_t_m2c'] / 1000.0
				RT = np.concatenate((R,T), axis=1)
				rnd_typ = 'real'
				K = self.info[item_name[0]][item_name[1]]['cam_K']
				cam_scale = 1000.0 / self.info[item_name[0]][item_name[1]]['depth_scale']
			else:
				if self.debug: #DEBUG:
					print('loading real train sample')
				# load real train sample
				## get rgb image
				im_path = self.real_tr_file_mask.format('rgb', item_name)
				with Image.open(im_path) as ri:
					if self.add_noise:
						ri=self.transcolor(ri)
					rgb = np.array(ri)[:,:,:3]
				## get depth image
				depth_path = self.real_tr_file_mask.format('depth', item_name)
				with Image.open(depth_path) as di:
					dpt = np.array(di)
				## get mask image
				mask_path = self.real_tr_file_mask.format('mask_cad', item_name)
				with Image.open(mask_path) as li:
					labels = np.array(li)
					labels = (labels>0).astype(np.uint8)
				## get meta information
				meta = self.meta[item_name][0]
				R = meta['cam_R_m2c']
				T = meta['cam_t_m2c'] / 1000.0
				RT = np.concatenate((R,T), axis=1)
				rnd_typ = 'real'
				K = self.info[item_name]['cam_K']
				cam_scale = 1000.0 / self.info[item_name]['depth_scale'] # should verify

			## 
			rgb = rgb[:, :, ::-1].copy()
			msk_dp = dpt > 1e-6
			if len(labels.shape)>2:
				labels = labels[:,:,0]
			rgb_labels = labels.copy()

			# #### debug
			# # imshow('{}_rgb'.format(item_name), rgb)
			# imshow('{}_labels'.format(item_name), rgb_labels*255)
			# cmd = waitKey(0)
			# if cmd == ord('q'):
			# 	exit()
			# ####

			##
			if self.add_noise and rnd_typ=='render':
				rgb = self.rgb_add_noise(rgb)
				# rgb_labels = labels.copy()
				rgb, dpt = self.add_real_back(rgb, rgb_labels, dpt, msk_dp)
				if self.rng.rand() > 0.8:
					rgb = self.rgb_add_noise(rgb)

			##
			rgb = np.transpose(rgb, (2,0,1)) # hwc 2 chw
			cld, choose = dpt_2_cld(self.xmap, self.ymap, dpt, cam_scale, K)

			labels = labels.flatten()[choose]
			rgb_lst = []
			for ic in range(rgb.shape[0]):
				rgb_lst.append(rgb[ic].flatten()[choose].astype(np.float32))
			rgb_pt = np.transpose(np.array(rgb_lst), (1,0)).copy()

			choose = np.array([choose])
			choose_2 = np.array([i for i in range(len(choose[0, :]))])

			if len(choose_2) < 400:
				return None
			if len(choose_2) > self.config.n_sample_points:
				c_mask = np.zeros(len(choose_2), dtype=int)
				c_mask[:self.config.n_sample_points] = 1
				np.random.shuffle(c_mask)
				choose_2 = choose_2[c_mask.nonzero()]
			else:
				choose_2 = np.pad(choose_2, (0, self.config.n_sample_points-len(choose_2)), 'wrap') # should verify

			cld_rgb = np.concatenate((cld, rgb_pt), axis=1)
			cld_rgb = cld_rgb[choose_2, :]
			cld = cld[choose_2, :]
			normal = self.get_normal(cld)[:, :3]
			normal[np.isnan(normal)] = 0.0
			cld_rgb_nrm = np.concatenate((cld_rgb, normal), axis=1)
			choose = choose[:, choose_2]
			labels = labels[choose_2].astype(np.int32)

			##
			RTs = np.zeros((self.config.n_objects, 3, 4)) # why is n_objects=1+1 ??
			kp3ds = np.zeros((self.config.n_objects, self.config.n_keypoints, 3))
			ctr3ds = np.zeros((self.config.n_objects, 3))
			cls_ids = np.zeros((self.config.n_objects, 1))
			kp_targ_ofst = np.zeros((self.config.n_sample_points, self.config.n_keypoints, 3))
			ctr_targ_ofst = np.zeros((self.config.n_sample_points, 3))
			for i, cls_id in enumerate([1]):
				RTs[i] = RT
				r = RT[:, :3]
				t = RT[:, 3]

				ctr = self.bs_utils.get_ctr(self.cls_type, ds_type="tless")[:, None]
				ctr = np.dot(ctr.T, r.T) + t
				ctr3ds[i, :] = ctr[0]
				msk_idx = np.where(labels == cls_id)[0]

				target_offset = np.array(np.add(cld, -1.0*ctr3ds[i, :]))
				ctr_targ_ofst[msk_idx,:] = target_offset[msk_idx, :]
				cls_ids[i, :] = np.array([1])

				kp_type = 'farthest{}'.format(self.config.n_keypoints)
				kps = self.bs_utils.get_kps(self.cls_type, kp_type=kp_type, ds_type='tless')
				kps = np.dot(kps, r.T) + t
				kp3ds[i] = kps

				target = []
				for kp in kps:
					target.append(np.add(cld, -1.0*kp))
				target_offset = np.array(target).transpose(1, 0, 2)  # [npts, nkps, c]
				kp_targ_ofst[msk_idx, :, :] = target_offset[msk_idx, :, :]

			# rgb, pcld, cld_rgb_nrm, choose, kp_targ_ofst, ctr_targ_ofst, cls_ids, RTs, labels, kp_3ds, ctr_3ds
			if self.debug: #DEBUG:
				return  torch.from_numpy(rgb.astype(np.float32)), \
						torch.from_numpy(cld.astype(np.float32)), \
						torch.from_numpy(cld_rgb_nrm.astype(np.float32)), \
						torch.LongTensor(choose.astype(np.int32)), \
						torch.from_numpy(kp_targ_ofst.astype(np.float32)), \
						torch.from_numpy(ctr_targ_ofst.astype(np.float32)), \
						torch.LongTensor(cls_ids.astype(np.int32)), \
						torch.from_numpy(RTs.astype(np.float32)), \
						torch.LongTensor(labels.astype(np.int32)), \
						torch.from_numpy(kp3ds.astype(np.float32)), \
						torch.from_numpy(ctr3ds.astype(np.float32)), \
						torch.from_numpy(K.astype(np.float32)), \
						torch.from_numpy(np.array(cam_scale).astype(np.float32))

			return  torch.from_numpy(rgb.astype(np.float32)), \
					torch.from_numpy(cld.astype(np.float32)), \
					torch.from_numpy(cld_rgb_nrm.astype(np.float32)), \
					torch.LongTensor(choose.astype(np.int32)), \
					torch.from_numpy(kp_targ_ofst.astype(np.float32)), \
					torch.from_numpy(ctr_targ_ofst.astype(np.float32)), \
					torch.LongTensor(cls_ids.astype(np.int32)), \
					torch.from_numpy(RTs.astype(np.float32)), \
					torch.LongTensor(labels.astype(np.int32)), \
					torch.from_numpy(kp3ds.astype(np.float32)), \
					torch.from_numpy(ctr3ds.astype(np.float32)), #\
					# torch.from_numpy(K.astype(np.float32)),
		# except:
		# 	return None

	def __len__(self):
		return len(self.all_list)

	def __getitem__(self, idx):
		if self.dataset_name == 'train':
			item_name = self.real_syn_gen()
			data = self.get_item(item_name)
			while data is None:
				item_name = self.real_syn_gen()
				data = self.get_item(item_name)
			if self.debug: #DEBUG:
				print('item_name: {}'.format(item_name))
			return data
		else:
			## no pp data for tless test
			item_name = self.all_list[idx]
			if self.debug: #DEBUG:
				print('item_name: {}'.format(item_name))
			return self.get_item(item_name)

def debug(cls=1):
	# global DEBUG
	# cls = 1
	# DEBUG = True
	ds = {}
	ds['train'] = Tless_Dataset('train', cls, debug=True)
	ds['val'] = Tless_Dataset('val', cls, debug=True) # there is no validation data
	ds['test'] = Tless_Dataset('test', cls, debug=True)
	idx = dict(
		train=0,
		val=0,
		test=0
	)
	while True:
		for cat in ['train', 'val', 'test']:
			datum = ds[cat].__getitem__(idx[cat])
			bs_utils = ds[cat].bs_utils
			idx[cat] += 1
			datum = [item.numpy() for item in datum]
			rgb, pcld, cld_rgb_nrm, choose, kp_targ_ofst, \
				ctr_targ_ofst, cls_ids, RTs, labels, kp3ds, ctr3ds, K, cam_scale = datum
			nrm_map = get_normal_map(ds[cat].width, ds[cat].height, cld_rgb_nrm[:, 6:], choose[0])
			imshow('nrm_map', nrm_map)
			rgb = rgb.transpose(1, 2, 0) # [...,::-1].copy()
			for i in range(22):
				# p2ds = bs_utils.project_p3d(pcld, 1.0, K) #cam_scale
				# rgb = bs_utils.draw_p2ds(rgb, p2ds, color=(0,255,0))
				kp3d = kp3ds[i]
				if kp3d.sum() < 1e-6:
					break
				## get point cloud
				# from IPython import embed; embed()
				# print("getting pointxyz for class {}".format(cls_ids[i,0]))
				mesh_pts = bs_utils.get_pointxyz(cls, ds_type="tless").copy()
				gt_r = RTs[i][:,:3]
				gt_t = RTs[i][:,3]
				gt_mesh_pts = np.dot(mesh_pts, gt_r.T) + gt_t
				gt_mesh_p2ds = bs_utils.project_p3d(gt_mesh_pts, cam_scale, K)
				rgb = bs_utils.draw_p2ds(rgb, gt_mesh_p2ds, color=(0,255,0))
				##
				kp_2ds = bs_utils.project_p3d(kp3d, cam_scale, K)
				rgb = bs_utils.draw_p2ds(
					rgb, kp_2ds, 3, (0, 0, 255) # bs_utils.get_label_color(cls_ids[i], mode=1)
				)
				ctr3d = ctr3ds[i]
				ctr_2ds = bs_utils.project_p3d(ctr3d[None, :], cam_scale, K)
				rgb = bs_utils.draw_p2ds(
					rgb, ctr_2ds, 4, (255, 0, 0) # bs_utils.get_label_color(cls_ids[i], mode=1)
				)
			imshow('{}_rgb'.format(cat), rgb)
			cmd = waitKey(0)
			if cmd == ord('q'):
				exit()
			else:
				continue

if __name__ == "__main__":
	import sys
	debug(sys.argv[1])




