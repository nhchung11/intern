import my_lib
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Lấy ma trận từ file bin và tham số
bin_path = r'D:\python\wavelet_intern\data\bin7.bin'
param_path = r'D:\python\wavelet_intern\data\param3.txt'

name, depth_scale, depth_width, depth_height, depth_cx, depth_cy, depth_fx, depth_fy = my_lib.get_parameter(param_path)
depth_data = my_lib.convert(bin_path, depth_width, depth_height)
original = depth_data.copy()
# my_lib.o3d_visualize(depth_data, depth_width, depth_height, depth_scale, depth_cx, depth_cy, depth_fx, depth_fy)
# cv2.imwrite(r'C:\Users\USER\Documents\Chung_intern\data\img3.jpg', img)
bar1_visible = False
upper_left = np.array([0, 0, 0])
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
# cv2.imshow('asnd', img)
# cv2.waitKey(0)  
for i in range(depth_height):
    for j in range(depth_width):
        if first_bar[depth_height - i - 1, j] != 0:
            upper_left[0] = j
            upper_left[1] = depth_height - i - 1
            upper_left[2] = first_bar[depth_height - i - 1, j]
            break

bottom_left = np.array([0, 0, 0])
for i in range(depth_height):
    for j in range(depth_width):
        if first_bar[i, j] != 0:
            bottom_left[0] = j
            bottom_left[1] = i 
            bottom_left[2] = first_bar[i, j]


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

upper_right = np.array([0, 0, 0])
bottom_right = np.array([0, 0, 0])

for i in range (depth_height):
    for j in range (depth_width):
        if depth_data[i, j] > 760:
            depth_data[i, j] = 0
        if depth_data[i, j] == first_bar[i, j]:
            depth_data[i, j] = 0
        if j > int(depth_width / 2):
            if depth_data[i, j] < upper_left[2] + 0.15 * i:
                depth_data[i, j] = 0
img = my_lib.get_largest_area(depth_data)
bar2 = depth_data.copy()
for i in range(depth_height):
    for j in range(depth_width):
        if img[i, j] == 0:
            bar2[i, j] = 0
# cv2.imshow('window', img)
# cv2.waitKey(0)
for i in range(depth_height):
    for j in range(depth_width):
        if bar2[depth_height - i - 1, j] != 0:
            upper_right[0] = j
            upper_right[1] = depth_height - i - 1
            upper_right[2] = bar2[depth_height - i - 1, j]
            break

for i in range(depth_height):
    for j in range(depth_width):
        if bar2[i, j] != 0:
            bottom_right[0] = j
            bottom_right[1] = i 
            bottom_right[2] = bar2[i, j]

if upper_left[1] - bottom_left[1] < 150:
    bottom_left[1] = depth_height

if upper_left[1] < upper_right[1]:
    upper_left[1] = upper_right[1]
else:
    upper_right[1] = upper_left[1]


left_avr = ((upper_left + bottom_left) / 2).astype(np.int32)
right_avr = ((upper_right + bottom_right) / 2).astype(np.int32)
# print(original.dtype)
tan_alpha, tan_beta = my_lib.angular_deviation(left_avr, right_avr)

for i in range(depth_width):
    a = abs(right_avr[0] - i)
    d = np.int32(a * tan_alpha)
    if i < right_avr[0]:
        # Nếu phần ảnh nằm bên trái thì đẩy ảnh cao lên
        original[:, i] += d
    else:
        # Nếu phần ảnh nằm bên phải thì hạ xuống
        original[:, i] -= d
for i in range(depth_height):
    for j in range(depth_width):
        if original[i, j] < 800:
            original[i, j] = 0

img = my_lib.get_8bit_image(original)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# cv2.circle(img, (upper_left[0], upper_left[1]), 5, (0,255,0), -1)
# cv2.circle(img, (bottom_left[0], bottom_left[1]), 5, (0,255,0), -1)
# cv2.circle(img, (upper_right[0], upper_right[1]), 5, (0,255,0), -1)
# cv2.circle(img, (bottom_right[0], bottom_right[1]), 5, (0,255,0), -1)
# cv2.circle(img, (left_avr[0], left_avr[1]), 5, (0,0,255), -1)
# cv2.circle(img, (right_avr[0], right_avr[1]), 5, (0,0,255), -1)
# cv2.imshow('window', img)
# cv2.waitKey(0)
matrix = np.zeros((depth_height, depth_width), dtype=int)
pts = np.array([[upper_left[0], upper_left[1]], [bottom_left[0], bottom_left[1]], [bottom_right[0], bottom_right[1]], [upper_right[0], upper_right[1]]], dtype=np.int32)
cv2.fillPoly(matrix, [pts], 1)
original = original * matrix

body = my_lib.get_largest_area(original)
for i in range (depth_height):
    for j in range(depth_width):
        if body[i, j] == 0:
            original[i, j] = 0
img = my_lib.get_largest_area(original)
# cv2.imshow('window', img)
# cv2.waitKey(0)
# depth_data = my_lib.get_backbone(original)
# backbone_line = my_lib.get_backbone_line(original)
# tail_val, tail_index = my_lib.get_tail(backbone_line)

# plt.scatter(tail_index, tail_val, c='r', marker='o')
# plt.plot(backbone_line)
# plt.xlabel('Cột')
# plt.ylabel('Độ cao')
# plt.title('Xương sống')
# plt.show()  
my_lib.o3d_visualize(original, depth_width, depth_height, depth_scale, depth_cx, depth_cy, depth_fx, depth_fy)

# print(left_avr, type(left_avr))
# print(right_avr, right_avr.dtype)
# print(tan_alpha, tan_beta)