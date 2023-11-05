import my_lib
import numpy as np
import cv2
bin_path = r'D:\python\wavelet_intern\data\bin2.bin'
param_path = r'D:\python\wavelet_intern\data\param1.txt'

name, depth_scale, depth_width, depth_height, depth_cx, depth_cy, depth_fx, depth_fy = my_lib.get_parameter(param_path)
depth_data = my_lib.convert(bin_path, depth_width, depth_height)

for i in range(depth_height):
    for j in range(depth_width):
        if depth_data[i, j] > 700:
            depth_data[i, j] = 0


# my_lib.o3d_visualize(depth_data, depth_width, depth_height, depth_scale, depth_cx, depth_cy, depth_fx, depth_fy)
img = my_lib.get_8bit_image(depth_data)
contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, 255, thickness=cv2.FILLED)
cv2.imshow('window', img)
cv2.waitKey(0)
