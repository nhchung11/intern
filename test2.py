import my_lib
import numpy as np
import cv2

# Lấy ma trận từ file bin và tham số
bin_path = r'D:\python\wavelet_intern\data\bin4.bin'
param_path = r'D:\python\wavelet_intern\data\213622251778.txt'

name, depth_scale, depth_width, depth_height, depth_cx, depth_cy, depth_fx, depth_fy = my_lib.get_parameter(param_path)
depth_data = my_lib.convert(bin_path, depth_width, depth_height)

for i in range(depth_height):
    for j in range (depth_width):
        if depth_data[i, j] == 0:
            depth_data[i, j] = 1200

num_partitions = 10
partition_width = depth_data.shape[1] // num_partitions

for i in range(num_partitions):
    start_col = i * partition_width
    end_col = (i + 1) * partition_width
    partition = depth_data[:, start_col:end_col]
    
    # Tìm giá trị nhỏ nhất trong từng phần
    min_value = np.min(partition)
    
    print(f"Phần {i + 1}: Giá trị nhỏ nhất = {min_value}")