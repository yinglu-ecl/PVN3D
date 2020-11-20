# tless_utils.py
# ying 2020/11/17
# useful functions for tless dataset
import yaml
import numpy as np

def load_gt(path):
	with open(path, 'r') as f:
		gts = yaml.load(f, Loader=yaml.CLoader)
		for im_id, gts_im in gts.items():
			for gt in gts_im:
				if 'cam_R_m2c' in gt.keys():
					gt['cam_R_m2c'] = np.array(gt['cam_R_m2c']).reshape((3, 3))
				if 'cam_t_m2c' in gt.keys():
					gt['cam_t_m2c'] = np.array(gt['cam_t_m2c']).reshape((3, 1))
	return gts

def load_info(path):
	with open(path, 'r') as f:
		info = yaml.load(f, Loader=yaml.CLoader)
		for eid in info.keys():
			if 'cam_K' in info[eid].keys():
				info[eid]['cam_K'] = np.array(info[eid]['cam_K']).reshape(
					(3, 3))
			if 'cam_R_w2c' in info[eid].keys():
				info[eid]['cam_R_w2c'] = np.array(
					info[eid]['cam_R_w2c']).reshape((3, 3))
			if 'cam_t_w2c' in info[eid].keys():
				info[eid]['cam_t_w2c'] = np.array(
					info[eid]['cam_t_w2c']).reshape((3, 1))
	return info

def dpt_2_cld(xmap, ymap, dpt, cam_scale, K):
		if len(dpt.shape) > 2:
			dpt = dpt[:, :, 0]
		msk_dp = dpt > 1e-6
		choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)
		if len(choose) < 1:
			return None, None

		dpt_mskd = dpt.flatten()[choose][:, np.newaxis].astype(np.float32)
		xmap_mskd = xmap.flatten()[choose][:, np.newaxis].astype(np.float32)
		ymap_mskd = ymap.flatten()[choose][:, np.newaxis].astype(np.float32)

		pt2 = dpt_mskd / cam_scale
		cam_cx, cam_cy = K[0][2], K[1][2]
		cam_fx, cam_fy = K[0][0], K[1][1]
		pt0 = (ymap_mskd - cam_cx) * pt2 / cam_fx
		pt1 = (xmap_mskd - cam_cy) * pt2 / cam_fy
		cld = np.concatenate((pt0, pt1, pt2), axis=1)
		return cld, choose

def get_normal_map(width, height, nrm, choose):
		nrm_map = np.zeros((height, width, 3), dtype=np.uint8)
		nrm = nrm[:, :3]
		nrm[np.isnan(nrm)] = 0.0
		nrm[np.isinf(nrm)] = 0.0
		nrm_color = ((nrm + 1.0) * 127).astype(np.uint8)
		nrm_map = nrm_map.reshape(-1, 3)
		nrm_map[choose, :] = nrm_color
		nrm_map = nrm_map.reshape((height, width, 3))
		return nrm_map
