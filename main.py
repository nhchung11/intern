import my_lib
import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import sqrt
bin_path = r'D:\python\wavelet_intern\data\bin1.bin'
param_path = r'D:\python\wavelet_intern\data\param1.txt'

name, depth_scale, depth_width, depth_height, depth_cx, depth_cy, depth_fx, depth_fy = my_lib.get_parameter(param_path)
depth_data = my_lib.convert(bin_path, depth_width, depth_height)
original = depth_data.copy()

# Lấy giá trị 4 góc và giá trị trung bình của 2 thanh
leftbar, leftbar_avr, upper_left, bottom_left = my_lib.get_first_bar(depth_data, depth_height, depth_width)
rightbar_avr, upper_right, bottom_right = my_lib.get_second_bar(depth_data, depth_height, depth_width, leftbar_avr, leftbar)

# Điều chỉnh 4 góc 
upper_left, bottom_left, upper_right, bottom_right = my_lib.get_4_corner_of_body(upper_left, bottom_left, upper_right, bottom_right, depth_height, depth_width)

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
# for i in range(depth_width):
#     a = abs(rightbar_avr[0] - i)
#     d = np.int32(a * tan_alpha)
#     if i > rightbar_avr[0]:
#     # Nếu phần ảnh nằm bên trái thì đẩy ảnh cao lên
#         for j in range(depth_height):
#             if original[j, i] != 0:
#                 original[j, i] += d
#     elif i < rightbar_avr[0]:
#         # Nếu phần ảnh nằm bên phải thì hạ xuống
#         # original[:, i] -= d
#         for j in range(depth_height):
#             if original[j, i] != 0:
#                 original[j, i] -= d


# Cắt ảnh theo 4 góc
matrix = np.zeros((depth_height, depth_width), dtype=int)
pts = np.array([[upper_left[0], upper_left[1]], [bottom_left[0], bottom_left[1]], [bottom_right[0], bottom_right[1]], [upper_right[0], upper_right[1]]], dtype=np.int32)
cv2.fillPoly(matrix, [pts], 1)
original = original * matrix

# # Xóa phần thừa
original = my_lib.get_largest_area(original)
body = original.copy()
line2d, belly1, belly2, tail2d = my_lib.get_backbone_line(body)
original = my_lib.get_largest_area(original)
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

# my_lib.o3d_visualize(original, depth_width, depth_height, depth_scale, depth_cx, depth_cy, depth_fx, depth_fy)
line3d = my_lib.get_largest_area(line3d)

# my_lib.backbone_visualize(line2d, belly1, belly2, tail2d)
# my_lib.draw_corner_and_bar_avr(line3d, upper_left, bottom_left, upper_right, bottom_right, leftbar_avr, rightbar_avr)
# my_lib.plt_visualize(line3d, depth_width, depth_height)
# theta = np.linspace(0, 2 * np.pi, 100)
# phi = np.linspace(0, np.pi, 100)
# theta, phi = np.meshgrid(theta, phi)
# radius = 200.0
# x_sphere = tail3d[0] + radius * np.sin(phi) * np.cos(theta)
# y_sphere = tail3d[1] + radius * np.sin(phi) * np.sin(theta)
# z_sphere = tail3d[2] + radius * np.cos(phi)
# x = np.arange(0, depth_width)
# y = np.arange(0, depth_height)
# x, y = np.meshgrid(x, y)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection = '3d')
# ax.plot_surface(x, y, line3d)
# # ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.5, color='c', label='Hình cầu')
# ax.scatter(tail3d[0], tail3d[1], tail3d[2], c='r', marker='x')
# # plt.show()

# # fig, ax = plt.subplots()

# # # Vẽ hình tròn

# # circle = plt.Circle((tail2d[0], tail2d[1]), radius, edgecolor='b', facecolor='none')
# # ax.add_patch(circle)
# # plt.plot(line2d)

# # c = 0
# # for i in range(len(line2d)):
# #     d = (i - tail2d[0])**2 + (backbone_line[i] - tail2d[1])**2 
# #     if abs(sqrt(d) - radius) < 1:
# #         c += 1
# # #     if abs(d - radius**2) < 10:
# #         ax.scatter(i, backbone_line[i], color = 'red')
# # plt.scatter(tail2d[0], tail2d[1], marker='x', c='green')
# # print(c)
# # plt.show()
point1 = np.array([0, 0, 0])
point2 = np.array([0, 0, 0])
for i in range(depth_height):
    for j in range(depth_width):
        if line3d[i, j] != 0:
            d = (j - tail3d[0])**2 + (i - tail3d[1])**2 + (line3d[i, j] - tail3d[2])**2
            # print(abs(sqrt(d) - radius))
            if abs(sqrt(d) - 50.0) < 1:
                point1[0] = j
                point1[1] = i
                point1[2] = line3d[i, j]
                
            if abs(sqrt(d) - 150.0) < 1:
                point2[0] = j
                point2[1] = i
                point2[2] = line3d[i, j]
                
# ax.scatter(point1[0], point1[1], point1[2], c='green', marker='x')
# ax.scatter(point2[0], point2[1], point2[2], c='orange', marker='x')
# ax.plot([point1[0], point2[0]], [point1[1], point2[1]], [point1[2], point2[2]], 'r-', label='Đường thẳng nối điểm')
# # plt.show()
vector = point1 - point2
# x1 = np.arange(0, depth_width)
# y1 = np.arange(0, depth_height)
# x1, y1 = np.meshgrid(x1, y1)
# z3 = (138316 - 151*x - 11*y)/63
# z1 = ((vector[0] * (x1 - point1[0]) + vector[1] * (y1 - point1[1])) / (-vector[2])) + point1[2]
# z2 = ((vector[0] * (x1 - point2[0]) + vector[1] * (y1 - point2[1])) / (-vector[2])) + point2[2]
# ax.plot_surface(x1, y1, z1)
# ax.plot_surface(x1, y1, z2)
# plt.show()
for i in range(depth_height):
    for j in range(depth_width):
        if original[i, j] != 0:
            if vector[0] * (j - point1[0]) + vector[1] * (i - point1[1]) + vector[2] * (original[i, j] - point1[2]) > 0 or vector[0] * (j - point2[0]) + vector[1] * (i - point2[1]) + vector[2] * (original[i, j] - point2[2]) < 0:
                original[i, j] = 0

my_lib.o3d_visualize(original, depth_width, depth_height, depth_scale, depth_cx, depth_cy, depth_fx, depth_fy)