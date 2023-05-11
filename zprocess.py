import os
import os.path as osp
from tqdm import tqdm
import numpy as np
from mmcv import load, dump
from tools.data.custom_2d_skeleton_track import pose_tracking

root = '/medical-data/zsxm/e3v/resized_data/'
infiles = ['train_annotations.pkl']
outfiles= ['train_annotations_track.pkl']

for infile, outfile in zip(infiles, outfiles):
    annotations = load(osp.join(root, infile))
    for anno in tqdm(annotations, desc=infile):
        kp = anno['keypoint'].astype(np.double)
        kps = anno['keypoint_score'].astype(np.double)
        kp = np.concatenate([kp, kps[..., None]], axis=-1)
        kp = pose_tracking(np.transpose(kp, (1,0,2,3)), max_tracks=2, num_joints=17).astype(np.double)
        kp, kps = kp[..., :2], kp[..., 2]
        gravitycenter = kp.mean(axis=2, keepdims=True)
        offset = np.linalg.norm(np.diff(kp-gravitycenter, axis=1), ord=2, axis=3).sum(axis=-1).mean(axis=-1)
        idx = np.argmax(offset)
        anno['keypoint'] = kp[idx:idx+1].astype(np.float16)
        anno['keypoint_score'] = kps[idx:idx+1].astype(np.float16)
        anno['num_person_raw'] = 1

    dump(annotations, osp.join(root, outfile))