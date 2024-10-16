import numpy as np
import torch
import ast
import odak

from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.data import Dataset
from src.util import *
from os import listdir
from os.path import join

class dataset(Dataset):
    def __init__(self, static_path, motion_path, stack_size=5, seed = 42):
        super().__init__()
        np.random.seed(seed)
        self.device = torch.device('cpu')
        self.datasets = []
        self.labels = []
        if motion_path:
            
            for group in listdir(motion_path):
                frames = sorted(listdir(join(motion_path, group)), key=extract_numbers)
                len_frames = len(frames) - stack_size + 1
                z_speed, y_speed, _ = ast.literal_eval(group)

                for i in range(stack_size-1, len(frames)):
                    if z_speed:
                        gd_z = (90/len_frames) * i                 
                        gd_y = 0
                    elif y_speed:
                        gd_z = 0
                        gd_y = (40/len_frames) * i    
                    else:
                        print("motion dataset error")
                        break
                    label = (gd_y, gd_z, 0)
                    self.labels.append(label)
                    frame_set = [join(motion_path, group, f) for f in frames[i-stack_size+1 :i+1]]
                    self.datasets.append(frame_set)
        if static_path:
            for group in listdir(static_path):
                frames = sorted(listdir(join(static_path, group)), key=extract_numbers)
                for i in range(stack_size-1, len(frames)):

                    frame_set = [join(static_path, group, f) for f in frames[i-stack_size+1 :i+1]]
                    self.datasets.append(frame_set)
                    label = ast.literal_eval(group)
                    self.labels.append(label)
    
    def __len__(self):
        return len(self.datasets)

    def read_image(self, file_paths):
        for idx, path in enumerate(file_paths):
            image = odak.learn.tools.load_image(
                                                path, 
                                                normalizeby = 255, 
                                                torch_style = False
                                               ).unsqueeze(0)
            if idx == 0:
                data = image
            else:
                data = torch.concatenate((data, image), dim=0)
        return data
    
    def __getitem__(self, idx):
        file_path = self.datasets[idx]
        data = self.read_image(file_path)
        label = self.labels[idx]

        yR, zR, zT = label
        label = [np.deg2rad(yR)/np.pi, np.deg2rad(zR)/np.pi, ((5000-zT)/5000)]
        label = [x if x != 0 else 1e-9 for x in label]
        label = torch.tensor(label, dtype=torch.float32)
        
        return data, label