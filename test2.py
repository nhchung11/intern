import my_lib
import numpy as np
import cv2

# Lấy ma trận từ file bin và tham số
bin_path = r'D:\python\wavelet_intern\data\bin7.bin'
param_path = r'D:\python\wavelet_intern\data\param3.txt'

name, depth_scale, depth_width, depth_height, depth_cx, depth_cy, depth_fx, depth_fy = my_lib.get_parameter(param_path)
depth_data = my_lib.convert(bin_path, depth_width, depth_height)


my_lib.o3d_visualize(depth_data, depth_width, depth_height, depth_scale, depth_cx, depth_cy, depth_fx, depth_fy)
