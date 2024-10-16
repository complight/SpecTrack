import torch
import ast
import tkinter as tk
import queue
from tkinter import font as tkFont

def pre_process(x):
    x_fft = torch.fft.fft2(x, norm = 'ortho')
    x_fft_shifted = torch.fft.fftshift(x_fft, dim = (-2, -1))
    magnitude = torch.abs(x_fft_shifted)
    magnitude = torch.log(magnitude + 1)
    magnitude = (magnitude - torch.min(magnitude)) / (torch.max(magnitude) - torch.min(magnitude))
    magnitude[magnitude==1] = 0
    magnitude = (magnitude - torch.min(magnitude)) / (torch.max(magnitude) - torch.min(magnitude))
    return magnitude

def extract_numbers(filename):
    return int(ast.literal_eval(filename.split(".png")[0])) 
