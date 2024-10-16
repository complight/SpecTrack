import sys
sys.path.append("./")
import argparse
import torch
import odak
import numpy as np
import threading
import csv
import cv2
from odak.learn.lensless import spec_track
from src.util import pre_process, extract_numbers
from src.datasets import *
from os import listdir, makedirs


__title__ = 'SpecTrack'


def main(
         settings_filename = 'settings/settings.txt',
         samples_dir = None, 
         weights_dir = None, 
         output_dir = None,
         visual = False,
        ):
    parser = argparse.ArgumentParser(description=__title__)
    parser.add_argument(
                        '--settings',
                        type = argparse.FileType('r'),
                        help = 'Filename for the settings file. Default is {}.'.format(settings_filename)
                       )
    parser.add_argument(
                        '--weights',
                        type = argparse.FileType('r'),
                        help = 'Filename for the weights file.'
                       )
    parser.add_argument(
                        '--input',
                        type = argparse.FileType('r'),
                        help = 'Foldername for the input laser speckle data to estimate.'
                       )
    parser.add_argument(
                        '--output',
                        type = argparse.FileType('r'),
                        help = 'Output directory of the recorded estimations.'
                       )
    parser.add_argument(
                        '--visual',
                        action=argparse.BooleanOptionalAction,
                        help = 'Flag to control if show a window with speckle images and estimations.'
                       )
    
    args = parser.parse_args()
    if not isinstance(args.settings, type(None)):
        settings_filename = str(args.settings.name)
    settings = odak.tools.load_dictionary(settings_filename)
        
    if not isinstance(args.weights, type(None)):
        samples_dir = str(args.weights.name)
    else:
        samples_dir = settings["general"]["samples directory"]
    if not isinstance(args.input, type(None)):
        weights_dir = str(args.input.name)
    else:
        weights_dir = settings["general"]["weights directory"]
    if not isinstance(args.output, type(None)):
        output_dir = str(args.output.name)
    else:
        output_dir = settings["general"]["output directory"]    
    if not isinstance(args.visual, type(None)):
        visual = bool(args.visual)
    else:
        visual = settings["general"]["visual"]    

    process(
            settings, 
            samples_dir, 
            weights_dir,
            output_dir,
            visual,
           )
    
def process(settings, samples_dir, weights_dir, output_dir, visual):
    """
    Process samples, record outputs to a CSV file, and optionally display results using OpenCV.

    Parameters
    ----------
    settings    : dict
                  Dictionary containing general settings.
    samples_dir : str
                  Directory containing sample frames.
    weights_dir : str
                  Directory containing model weights.
    output_dir  : str
                  Directory to save the output CSV file.
    visual      : bool
                  Flag to control if to show the visual.
    """
    device = torch.device(settings["general"]["device"])

    network = spec_track(device=device)
    network.load_weights(filename=weights_dir)
    network.eval()

    frames = listdir(samples_dir)
    frames = sorted(frames, key=extract_numbers)

    makedirs(output_dir, exist_ok=True)
    csv_file = join(output_dir, 'output_results.csv')
    header = ['Frame', 'Y_Rotation', 'Z_Rotation', 'Depth_cm']

    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

        # Use tqdm for progress bar
        for idx in tqdm(range(len(frames) - 4), desc="Processing frames"):
            if frames[idx].endswith(".png"):
                data = torch.zeros((1, 0, 0, 0))
                for i in range(5):
                    frame_path = join(samples_dir, frames[idx + i])
                    frame = odak.learn.tools.load_image(
                        frame_path,
                        normalizeby=255,
                        torch_style=True
                    ).unsqueeze(0).unsqueeze(0)

                    frame = pre_process(frame)
                    frame = odak.learn.tools.crop_center(frame).to(device)
                    if data.shape[1] == 0:
                        data = frame
                        continue
                    if data.shape[1] < 5:
                        data = torch.concatenate((data, frame), dim=1)
                    if data.shape[1] == 5:
                        with torch.no_grad():
                            predict = network.forward(data.float())
                            yR, zR, zT = odak.tools.convert_to_numpy(predict.squeeze(0))
                            if yR:
                                yR = np.rad2deg(yR * np.pi) if yR > 0 else 0
                                zR = np.rad2deg(zR * np.pi) if zR > 0 else 0
                                zT = zT * 5000
                                depth_cm = round(16 + (12 / 5000 * zT), 1)

                                # Write to CSV
                                writer.writerow([idx, round(yR, 2), round(zR, 2), depth_cm])

                                # print(f"Frame: {idx}, Y rotation: {round(yR, 2)}°, Z rotation: {round(zR, 2)}°, Depth: {depth_cm} cm")
                                
                                if visual:
                                    # Load the image for visualization
                                    img = cv2.imread(frame_path)
                                    
                                    # Add text to the image
                                    text = f"Frame: {idx}, Y: {round(yR, 2)}°, Z: {round(zR, 2)}°, Depth: {depth_cm} cm"
                                    cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                    
                                    # Display the image
                                    cv2.imshow('Frame with Estimations', img)
                                    
                                    # Wait for a short time and check for 'q' key to quit
                                    if cv2.waitKey(1) & 0xFF == ord('q'):
                                        break

    print(f"Processing complete. Results saved to {csv_file}")
    
    if visual:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())