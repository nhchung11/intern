import my_lib
import numpy as np
import cv2
bin_path = r'D:\python\wavelet_intern\data\bin8.bin'
param_path = r'D:\python\wavelet_intern\data\param1.txt'

name, depth_scale, depth_width, depth_height, depth_cx, depth_cy, depth_fx, depth_fy = my_lib.get_parameter(param_path)
depth_data = my_lib.convert(bin_path, depth_width, depth_height)
original = depth_data.copy()


my_lib.o3d_visualize(original, depth_width, depth_height, depth_scale, depth_cx, depth_cy, depth_fx, depth_fy)