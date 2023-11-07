import my_lib
import numpy as np
import cv2
import matplotlib.pyplot as plt
# Lấy ma trận từ file bin và tham số
bin_path = r'D:\python\wavelet_intern\data\bin1.bin'
param_path = r'D:\python\wavelet_intern\data\param1.txt'

name, depth_scale, depth_width, depth_height, depth_cx, depth_cy, depth_fx, depth_fy = my_lib.get_parameter(param_path)
depth_data = my_lib.convert(bin_path, depth_width, depth_height)


# Lấy thanh ngang thứ nhất và thứ hai
firstbar, firstbar_avr,_, _ = my_lib.get_first_bar(depth_data, depth_height, depth_width)
secondbar, secondbar_avr = my_lib.get_second_bar(depth_data)
bars = firstbar + secondbar

# my_lib.plt_bars_visualize(bars, depth_width, depth_height, firstbar_avr, secondbar_avr)

tan_alpha, tan_beta = my_lib.angular_deviation(firstbar_avr, secondbar_avr)

# Chuẩn hóa ảnh cho 2 thanh ngang bằng nhau
# Lấy thanh cao hơn làm gốc
# print(depth_data.dtype)
for i in range(depth_width):
    a = abs(secondbar_avr[0] - i)
    d = np.int32(a * tan_alpha)
    if i < secondbar_avr[0]:
        # Nếu phần ảnh nằm bên trái thì đẩy ảnh cao lên
        depth_data[:, i] += d
    else:
        # Nếu phần ảnh nằm bên phải thì hạ xuống
        depth_data[:, i] -= d
for i in range(depth_height):
    for j in range(depth_width):
        if depth_data[i, j] < 250:
            depth_data[i, j] = 0
# my_lib.o3d_visualize(depth_data, depth_width, depth_height, depth_scale, depth_cx, depth_cy, depth_fx, depth_fy)
# my_lib.plt_visualize(depth_data, depth_width, depth_height)
# my_lib.plt_bars_visualize(bars, depth_width, depth_height, firstbar_avr, secondbar_avr)

# Xóa phần thừa
for i in range(depth_height):
    for j in range(depth_width):
        if depth_data[i, j] != 0 and bars[i, j] != 0:
            depth_data[i, j] = 0
        if depth_data[i, j] < 800:
            depth_data[i, j] = 0

# Vector vuông góc với đường thẳng nối 2 điểm trung bình của 2 thanh
direction_vector = np.array([secondbar_avr[1] - firstbar_avr[1], firstbar_avr[0] - secondbar_avr[0], 0])

# Tọa độ 4 điểm chứa phần thân lợn
point1 = (firstbar_avr + (200 / np.linalg.norm(direction_vector)) * direction_vector).astype(np.int32)
point2 = (firstbar_avr - (200 / np.linalg.norm(direction_vector)) * direction_vector).astype(np.int32)
point3 = (secondbar_avr + (200 / np.linalg.norm(direction_vector)) * direction_vector).astype(np.int32) 
point4 = (secondbar_avr - (200 / np.linalg.norm(direction_vector)) * direction_vector).astype(np.int32) 
    

# my_lib.plt_bars_visualize(bars, depth_width, depth_height, firstbar_avr, secondbar_avr)
# my_lib.o3d_visualize(depth_data, depth_width, depth_height, depth_scale, depth_cx, depth_cy, depth_fx, depth_fy)
# print(point1, point2, point3, point4)
matrix = np.zeros((depth_height, depth_width), dtype=int)
pts = np.array([[point1[0], point1[1]], [point2[0], point2[1]], [point4[0], point4[1]], [point3[0], point3[1]]], dtype=np.int32)
cv2.fillPoly(matrix, [pts], 1)
depth_data = depth_data * matrix

body = my_lib.get_largest_area(depth_data)
for i in range (depth_height):
    for j in range(depth_width):
        if body[i, j] == 0:
            depth_data[i, j] = 0
# cv2.imshow('window', gray)
# cv2.waitKey(0)
depth_data = my_lib.get_backbone(depth_data)
backbone_line = my_lib.get_backbone_line(depth_data)
tail_val, tail_index = my_lib.get_tail(backbone_line)

plt.scatter(tail_index, tail_val, c='r', marker='o')
plt.plot(backbone_line)
plt.xlabel('Cột')
plt.ylabel('Độ cao')
plt.title('Xương sống')
plt.show()
# my_lib.o3d_visualize(depth_data, depth_width, depth_height, depth_scale, depth_cx, depth_cy, depth_fx, depth_fy)