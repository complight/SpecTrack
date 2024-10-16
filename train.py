import sys
sys.path.append("./")
import torch
import odak
import numpy as np
import argparse
from odak.learn.lensless import spec_track
from src.datasets import *
from torch.utils.data import random_split


__title__ = 'SpecTrack'


def main(
         settings_filename = 'settings/settings.txt',
         static_dataset_dir = None,
         motion_dataset_dir = None
        ):
    parser = argparse.ArgumentParser(description=__title__)
    parser.add_argument(
                        '--settings',
                        type = argparse.FileType('r'),
                        help = 'Filename for the settings file. Default is {}.'.format(settings_filename)
                       )
    parser.add_argument(
                        '--static',
                        type = argparse.FileType('r'),
                        help = 'The directory of the static setting dataset'
                       )
    parser.add_argument(
                        '--motion',
                        type = argparse.FileType('r'),
                        help = 'The directory of the dynamic setting dataset'
                       )

    args = parser.parse_args()
    if not isinstance(args.settings, type(None)):
        settings_filename = str(args.settings.name)
    settings = odak.tools.load_dictionary(settings_filename)
        
    if not isinstance(args.static, type(None)):
        static_dataset_dir = str(args.static.name)
    else:
        static_dataset_dir = settings["train dataset"]["static dataset directory"]
    if not isinstance(args.motion, type(None)):
        motion_dataset_dir = str(args.motion.name)
    else:
        motion_dataset_dir = settings["train dataset"]["motion dataset directory"]
    process(settings, static_dataset_dir, motion_dataset_dir)
    
def process(settings, static_dataset_dir, motion_dataset_dir):
    device = torch.device(settings["general"]["device"])

    network = spec_track(
        device=device
    )

    test_ratio = settings["model"]["train validation ratio"]
    seed = settings["model"]["seed"]
    num_epochs = settings["model"]["number of epochs"]
    batch_size = settings["model"]["batch size"]
    lr = settings["model"]["learning rate"]
    weight_decay = settings["model"]["weight decay"]
    output_dir = settings["model"]["output directory"]
    num_workers = settings["model"]["number of workers"]

    dataset = dataset(static_path=static_dataset_dir, motion_path=motion_dataset_dir)

    generator = torch.Generator().manual_seed(seed)
    train_set, test_set = random_split(dataset, [1-test_ratio, test_ratio], generator=generator)
    print(f"The train dataset & test dataset ratio: {len(train_set)} {len(test_set)}")
    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    network.fit(trainloader, testloader, number_of_epochs = num_epochs, learning_rate = lr, weight_decay = weight_decay, directory = output_dir)

if __name__ == "__main__":
    sys.exit(main())