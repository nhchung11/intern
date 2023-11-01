import my_lib
import numpy as np
import cv2

def func1(depth_data, depth_width, depth_height):
    bar1 = depth_data.copy()
    min_val = np.min(bar1[bar1 != 0])
    for i in range(depth_height):
        for j in range(int(depth_width / 2)):
            if bar1[i, j] > min_val + 60:
                bar1[i, j] = 0
            else:
                depth_data[i, j] = 0
        for j in range(int(depth_width / 2), depth_width):
            if bar1[i, j] > min_val + 180:
                bar1[i, j] = 0
            else:
                depth_data[i, j] = 0
    print(min_val)
    return depth_data, bar1

def func2(depth_data, depth_width, depth_height):
    bar2 = depth_data.copy()
    min_val = np.min(bar2[bar2 != 0])
    for i in range(depth_height):
        for j in range(depth_width):
            if bar2[i, j] > min_val + 60:
                bar2[i, j] = 0
            else:
                depth_data[i, j] = 0
    print(min_val)
    return depth_data, bar2

# Lấy ma trận từ file bin và tham số
bin_path = r'D:\python\wavelet_intern\data\bin1.bin'
param_path = r'D:\python\wavelet_intern\data\213622251778.txt'

name, depth_scale, depth_width, depth_height, depth_cx, depth_cy, depth_fx, depth_fy = my_lib.get_parameter(param_path)
depth_data = my_lib.convert(bin_path, depth_width, depth_height)

depth_data, bar1 = func1(depth_data, depth_width, depth_height)
depth_data, bar2 = func2(depth_data, depth_width, depth_height)
img = my_lib.get_16bit_image(depth_data)
bar1img = my_lib.get_16bit_image(bar1)
# cv2.imshow('window', img)
# cv2.waitKey(0)

my_lib.o3d_visualize(depth_data, depth_width, depth_height, depth_scale, depth_cx, depth_cy, depth_fx, depth_fy)








