import numpy as np
import torch
from torch.utils.data import Dataset

from feeders import tools

########START ANGLE ADD BY LSQ 2026.4.22######################
# NTU-25 bone pairs: (parent, child) following the kinematic tree
NTU_BONES = [
    (0,1), (1,20), (20,2), (2,3),        # spine chain
    (20,4), (4,5), (5,6), (6,7), (7,21), (7,22),  # left arm + hand
    (20,8), (8,9), (9,10), (10,11), (11,23), (11,24),  # right arm + hand
    (0,12), (12,13), (13,14), (14,15),    # left leg
    (0,16), (16,17), (17,18), (18,19),    # right leg
]

# Second time from CLd 2026.4.22 by lsq
# For each joint where an angle is defined, which two bones form it. 
# Indices refer to positions in NTU_BONES above.
PRIMARY_ANGLE_AT_JOINT = {
    1: (0, 1),    2: (2, 3),    20: (1, 2),
    4: (4, 5),    5: (5, 6),    6: (6, 7),   7: (7, 8),
    8: (10, 11),  9: (11, 12),  10: (12, 13), 11: (13, 14),
    12: (16, 17), 13: (17, 18), 14: (18, 19),
    16: (20, 21), 17: (21, 22), 18: (22, 23),
}

# Joints where 2+ bones meet → angle defined
# Map: joint_id → list of (bone_idx_1, bone_idx_2) pairs
ANGLE_JOINTS = {
    1: [(0,1)],      # spine-base to mid-spine
    20: [(1,2), (1,4), (1,10)],  # mid-spine branching
    2: [(2,3)],       # neck
    4: [(4,5)],       # left shoulder
    5: [(5,6)],       # left elbow
    6: [(6,7)],       # left wrist
    8: [(10,11)],     # right shoulder
    9: [(11,12)],     # right elbow
    10: [(12,13)],    # right wrist
    12: [(16,17)],    # left hip
    13: [(17,18)],    # left knee
    14: [(18,19)],    # left ankle
    16: [(20,21)],    # right hip
    17: [(21,22)],    # right knee
    18: [(22,23)],    # right ankle
    0: [(0,16), (0,20)],  # hip center branching
}

def compute_bone_angles(joint_data):
    """
    Compute bone angles at joints where two bones meet.
    Args:
        joint_data: (C, T, V, M) where C=3 (x,y,z)
    Returns:
        angles: (1, T, V, M) — angle features per joint per frame
    """
    C, T, V, M = joint_data.shape
    
    # Compute bone vectors
    bones = np.zeros((len(NTU_BONES), T, 3, M))
    for idx, (p, c) in enumerate(NTU_BONES):
        diff = joint_data[:, :, c, :] - joint_data[:, :, p, :]
        if hasattr(diff, 'permute'):
            bones[idx] = diff.permute(1, 0, 2)      # PyTorch Tensor
        else:
            bones[idx] = diff.transpose(1, 0, 2)    # NumPy Array
    
    # Compute angle at each joint, zero for leaf joints
    angles = np.zeros((1, T, V, M))
    
    for j_id, bone_pairs in ANGLE_JOINTS.items():
        b1_idx, b2_idx = bone_pairs[0]
        b1 = bones[b1_idx]  # (T, 3, M)
        b2 = bones[b2_idx]  # (T, 3, M)
        
        # Stable angle via atan2
        cross = np.cross(b1.transpose(0,2,1), b2.transpose(0,2,1))  # (T, M, 3)
        cross_norm = np.linalg.norm(cross, axis=-1)  # (T, M)
        dot = (b1.transpose(0,2,1) * b2.transpose(0,2,1)).sum(axis=-1)  # (T, M)
        angle = np.arctan2(cross_norm, dot)  # (T, M), range [0, pi]
        
        angles[0, :, j_id, :] = angle
    
    return angles

###END  ANGLE ADD BY LSQ 2026.4.22###########################


class Feeder(Dataset):
    def __init__(self, data_path, label_path=None, p_interval=1, split='train', random_choose=False, random_shift=False,
                 random_move=False, random_rot=False, window_size=-1, normalization=False, debug=False, use_mmap=False,
                 bone=False, vel=False):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.split = split
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.normalization = normalization
        self.use_mmap = use_mmap
        self.p_interval = p_interval
        self.random_rot = random_rot
        self.bone = bone
        self.vel = vel
        self.load_data()
        if normalization:
            self.get_mean_map()

    def load_data(self):
        npz_data = np.load(self.data_path)
        if self.split == 'train':
            self.data = npz_data['x_train']
            self.label = np.where(npz_data['y_train'] > 0)[1]
            self.sample_name = ['train_' + str(i) for i in range(len(self.data))]
        elif self.split == 'test':
            self.data = npz_data['x_test']
            self.label = np.where(npz_data['y_test'] > 0)[1]
            self.sample_name = ['test_' + str(i) for i in range(len(self.data))]
        else:
            raise NotImplementedError('data split only supports train/test')
        N, T, _ = self.data.shape
        self.data = self.data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)

    def get_mean_map(self):
        data = self.data
        N, C, T, V, M = data.shape
        self.mean_map = data.mean(axis=2, keepdims=True).mean(axis=4, keepdims=True).mean(axis=0)
        self.std_map = data.transpose((0, 2, 4, 1, 3)).reshape((N * T * M, C * V)).std(axis=0).reshape((C, 1, V, 1))

    def __len__(self):
        return len(self.label)

    def __iter__(self):
        return self

    def __getitem__(self, index):
        data_numpy = self.data[index]
        label = self.label[index]
        data_numpy = np.array(data_numpy)
        valid_frame_num = np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0)
        # reshape Tx(MVC) to CTVM
        data_numpy = tools.valid_crop_resize(data_numpy, valid_frame_num, self.p_interval, self.window_size)
        if self.random_rot:
            data_numpy = tools.random_rot(data_numpy)

        # ========== COMPUTE ANGLES FROM RAW JOINTS ==========
        # Must be computed BEFORE bone/vel transforms overwrite data_numpy.
        # Angles are rotation-invariant, so computing after random_rot is safe.
        angle_feat = compute_bone_angles(data_numpy)  # (1, T, V, M)
        # Normalize angle to [-1, 1] range to match xyz scale
        angle_feat = angle_feat / np.pi * 2.0 - 1.0   # [0, pi] -> [-1, 1]
        # =====================================================
        
        if self.bone:
            from.bone_pairs import ntu_pairs
            bone_data_numpy = np.zeros_like(data_numpy)
            for v1, v2 in ntu_pairs:
                bone_data_numpy[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
            data_numpy = bone_data_numpy
        if self.vel:
            data_numpy[:, :-1] = data_numpy[:, 1:] - data_numpy[:, :-1]
            data_numpy[:, -1] = 0

        # ========== CONCATENATE ANGLE CHANNEL ==========
        # BUG FIX (2026.5.2): Previously compute_bone_angles was called
        # TWICE — once before and once after bone/vel transforms. The second
        # call computed angles from bone vectors or velocities, producing
        # garbage. Now angle_feat is computed only once from raw joints above.
        data_numpy = np.concatenate([data_numpy, angle_feat], axis=0)  # (4, T, V, M)
        # ================================================

        return data_numpy, label, index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
