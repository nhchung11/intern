import my_lib
import numpy as np
import cv2
bin_path = r'C:\Users\USER\Documents\Chung_intern\data\bin7.bin'
param_path = r'C:\Users\USER\Documents\Chung_intern\data\param3.txt'

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

# Nâng ảnh cho 2 thanh bằng nhau
tan_alpha, tan_beta = my_lib.angular_deviation(leftbar_avr, rightbar_avr)
for i in range(depth_width):
    a = abs(rightbar_avr[0] - i)
    d = np.int32(a * tan_alpha)
    if i < rightbar_avr[0]:
        # Nếu phần ảnh nằm bên trái thì đẩy ảnh cao lên
        original[:, i] += d
    else:
        # Nếu phần ảnh nằm bên phải thì hạ xuống
        original[:, i] -= d
    # Cắt phần thanh ở trên
    for j in range(depth_height):
        if original[j, i] < leftbar_avr[2] + 0.3 * i or original[j, i] < 630:
            original[j, i] = 0
        

# Cắt ảnh theo 4 góc
matrix = np.zeros((depth_height, depth_width), dtype=int)
pts = np.array([[upper_left[0], upper_left[1]], [bottom_left[0], bottom_left[1]], [bottom_right[0], bottom_right[1]], [upper_right[0], upper_right[1]]], dtype=np.int32)
cv2.fillPoly(matrix, [pts], 1)
original = original * matrix

# Xóa phần thừa
original = my_lib.get_largest_area(original)
# backbone_line = my_lib.get_backbone_line(original)

my_lib.o3d_visualize(original, depth_width, depth_height, depth_scale, depth_cx, depth_cy, depth_fx, depth_fy)
# my_lib.backbone_visualize(backbone_line)
# my_lib.draw_corner_and_bar_avr(original, upper_left, bottom_left, upper_right, bottom_right, leftbar_avr, rightbar_avr)
# print(original.dtype)