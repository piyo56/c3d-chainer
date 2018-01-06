import sys, os, glob
import re

import chainer
import numpy as np
from PIL import Image

regex_frame_number = re.compile(r'.*?([0-9]+)\.jpg')
def frame_number(name):
    match = re.search(regex_frame_number, name)
    return int(match.group(1))

class RgbdLabeledMugDataset(chainer.dataset.DatasetMixin):
    def __init__(self, x_data_path, t_data_path, video_length=16, num_labels=6):
        self.X_data = np.load(x_data_path)
        self.t_data = np.load(t_data_path).astype(np.int)
        # self.t_data = np.eye(num_labels)[self.t_data]

        self.video_length = video_length
        self.num_labels = num_labels

    def __len__(self):
        return len(self.t_data)

    def get_example(self, i):
        """return video shape: (ch, video_length, width, height, depth)"""

        video_path, categ = self.X_data[i], self.t_data[i]
    
        # read video
        rgb_files = sorted(glob.glob(os.path.join(video_path, 'rgb', '*.jpg')), key=frame_number)
        depth_files = sorted(glob.glob(os.path.join(video_path, 'depth', '*.jpg')), key=frame_number)
        video_rgb = np.asarray([np.asarray(Image.open(f)) for f in rgb_files])
        video_depth = np.asarray([np.asarray(Image.open(f)) for f in depth_files])[:,:,:,None]
    
        # normalize, etc
        video = np.concatenate((video_rgb, video_depth), axis=3)
        video = ((video - 128.) / 128.).astype(np.float32)
        video = video.transpose(3, 0, 1, 2)

        return video, categ
