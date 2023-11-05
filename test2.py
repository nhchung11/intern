import my_lib
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Lấy ma trận từ file bin và tham số
bin_path = r'D:\python\wavelet_intern\data\bin1.bin'
param_path = r'D:\python\wavelet_intern\data\param1.txt'

name, depth_scale, depth_width, depth_height, depth_cx, depth_cy, depth_fx, depth_fy = my_lib.get_parameter(param_path)
depth_data = my_lib.convert(bin_path, depth_width, depth_height)

# my_lib.o3d_visualize(depth_data, depth_width, depth_height, depth_scale, depth_cx, depth_cy, depth_fx, depth_fy)
# cv2.imwrite(r'C:\Users\USER\Documents\Chung_intern\data\img3.jpg', img)
bar1_visible = False
bottom_left = np.array([0, 0, 0])
first_bar = np.zeros_like(depth_data)
for i in range(depth_height):
    for j in range(int(depth_width / 2)):
        if depth_data[i, j] < 520 and depth_data[i, j] != 0:
            first_bar[i, j] = depth_data[i, j]

img = my_lib.get_8bit_image(first_bar)
img = my_lib.get_largest_area(first_bar)
for i in range(depth_height):
    for j in range(depth_width):
        if img[i, j] == 0:
            first_bar[i, j] = 0
for i in range(depth_height):
    for j in range(depth_width):
        if first_bar[depth_height - i - 1, j] != 0:
            bottom_left[0] = j
            bottom_left[1] = depth_height - i - 1
            bottom_left[2] = first_bar[depth_height - i - 1, j]


upper_left = np.array([0, 0, 0])
for i in range(depth_height):
    for j in range(depth_width):
        if first_bar[i, j] != 0:
            upper_left[0] = j
            upper_left[1] = i 
            upper_left[2] = first_bar[i, j]

# x = np.arange(0, depth_width)
# y = np.arange(0, depth_height)
# x, y = np.meshgrid(x, y)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')
# ax.scatter(upper_left[0], upper_left[1], upper_left[2], c='red', marker='o')
# ax.scatter(bottom_left[0], bottom_left[1], bottom_left[2], c='red', marker='o')
# ax.plot_surface(x, y, first_bar)
# plt.show()
# my_lib.o3d_visualize(first_bar, depth_width, depth_height, depth_scale, depth_cx, depth_cy, depth_fx, depth_fy)

# cv2.imshow('window', img)
# cv2.waitKey(0)

