import my_lib
import numpy as np
from math import sqrt

bin_path = r'D:\python\wavelet_intern\data\bin4.bin'
param_path = r'D:\python\wavelet_intern\data\param2.txt'

name, depth_scale, depth_width, depth_height, depth_cx, depth_cy, depth_fx, depth_fy = my_lib.get_parameter(param_path)
depth_data = my_lib.convert(bin_path, depth_width, depth_height)
original = depth_data.copy()
points = []
# Lấy giá trị 4 góc và giá trị trung bình của 2 thanh
leftbar, leftbar_avr, upper_left, bottom_left = my_lib.get_first_bar(depth_data)
rightbar_avr, upper_right, bottom_right = my_lib.get_second_bar(depth_data, leftbar_avr, leftbar)

# Điều chỉnh 4 góc 
upper_left, bottom_left, upper_right, bottom_right = my_lib.get_4_corner_of_body(original, upper_left, bottom_left, upper_right, bottom_right)

if upper_left[1] < depth_height / 2:
    original[:upper_left[1], :] = 0
else:
    original[upper_left[1]:, :] = 0

for i in range(depth_height):
    for j in range(depth_width):
        if original[i, j] < leftbar_avr[2] + 0.3 * j or original[i, j] < 630:
        # if original[j, i] < 630:
            original[i, j] = 0
# Nâng ảnh cho 2 thanh bằng nhau
tan_alpha, tan_beta = my_lib.angular_deviation(leftbar_avr, rightbar_avr)
for i in range(depth_width):
    a = abs(rightbar_avr[0] - i)
    d = np.int32(a * tan_alpha)
    if i > rightbar_avr[0]:
    # Nếu phần ảnh nằm bên trái thì đẩy ảnh cao lên
        for j in range(depth_height):
            if original[j, i] != 0:
                original[j, i] += d
    elif i < rightbar_avr[0]:
        # Nếu phần ảnh nằm bên phải thì hạ xuống
        # original[:, i] -= d
        for j in range(depth_height):
            if original[j, i] != 0:
                original[j, i] -= d


# Cắt ảnh theo 4 góc
original = my_lib.cut(original, upper_left, bottom_left, upper_right, bottom_right)

body = original.copy()
line2d, belly1, belly2, tail2d = my_lib.get_backbone_line(body)

line3d = np.zeros_like(original)
for i in range(depth_width):
    arr = original[:, i]
    arr2 = []
    count = 0
    for j in range(depth_height):
        if arr[j] != 0:
            arr2.append(j)
            count += 1
    if count > 0:
        y = arr2[int(len(arr2) / 2)]
        line3d[y, i] = original[y, i]
max_val = np.max(line2d)
tail3d = np.array([0, tail2d[0], max_val - tail2d[1]])
arr3 = line3d[:, tail2d[0]]
# arr3 = np.max(arr3) - arr3
for i in range(depth_height):
    if arr3[i] != 0:
        tail3d[0] = i
        break
tmp = tail3d[0]
tail3d[0] = tail3d[1]
tail3d[1] = tmp

point1, point2 = my_lib.get_point_and_nearby(tail3d, line3d, depth_height)


p1, p2, p3 = my_lib.get3points(original, point1, point2)
d1, d2, arcos_degree = my_lib.get_result(p1, p2, p3)
# points.append(tail3d)
points.append(p1)
points.append(p2)
points.append(p3)
point = my_lib.get_tail3d(original, line3d)
points.append(point)
print(f"Rho1: {d1}")
print(f"Rho2: {d2}")
print(f"Góc lệch: {arcos_degree}")

# my_lib.plt_visualize(original, points) 
