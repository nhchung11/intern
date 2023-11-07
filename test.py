import my_lib
import numpy as np
import cv2
bin_path = r'D:\python\wavelet_intern\data\bin1.bin'
param_path = r'D:\python\wavelet_intern\data\param1.txt'

name, depth_scale, depth_width, depth_height, depth_cx, depth_cy, depth_fx, depth_fy = my_lib.get_parameter(param_path)
depth_data = my_lib.convert(bin_path, depth_width, depth_height)
original = depth_data.copy()

leftbar, leftbar_avr, upper_left, bottom_left = my_lib.get_first_bar(depth_data, depth_height, depth_width)
for i in range(depth_height):
    for j in range(depth_width):
        if depth_data[i, j] > leftbar_avr[2] + 0.3 * j:
            depth_data[i, j] = 0
depth_data = depth_data - leftbar
img = my_lib.get_largest_area(depth_data)
# cv2.imshow('window', img)
# cv2.waitKey(0)
row = img[leftbar_avr[1],:]
point1 = np.array([0, 0])
point2 = np.array([0, 0])
point3 = np.array([0, 0])
for i in range(depth_width):
    if row[i] == 255:
        point1[0] = i + 20
        point1[1] = leftbar_avr[1]
        col = img[:, point1[0]]
        for j in range(depth_height):
            if col[j] == 255:
                point2[0] = point1[0]
                point2[1] = j
                break
        for j in range(depth_height):
            if col[depth_height - j - 1] == 255:
                point3[0] = point1[0]
                point3[1] = depth_height - j - 1
                break
        break


if upper_left[1] - bottom_left[1] < 150:
    bottom_left[1] = depth_height

if upper_left[1] < point2[1]:
    upper_left[1] = point2[1]
else:
    point2[1] = upper_left[1]
matrix = np.zeros((depth_height, depth_width), dtype=int)
pts = np.array([[upper_left[0], upper_left[1]], [bottom_left[0], bottom_left[1]], [point3[0], point3[1]], [point2[0], point2[1]]], dtype=np.int32)
cv2.fillPoly(matrix, [pts], 1)
original = original * matrix

body = my_lib.get_largest_area(original)
for i in range (depth_height):
    for j in range(depth_width):
        if body[i, j] == 0:
            original[i, j] = 0

img = my_lib.get_8bit_image(original)
cv2.imshow('window', img)
cv2.waitKey(0)