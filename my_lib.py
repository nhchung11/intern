import numpy as np
import os
import struct
import matplotlib.pyplot as plt
import open3d as o3d
import cv2

# Đọc file bin 
def convert(bin_file_path, depth_width, depth_height):
    data_format = f"{depth_width * depth_height}H"
    data = []
    if(os.path.exists(bin_file_path)):
        with open(bin_file_path, 'rb') as f:
            data = struct.unpack(data_format, f.read(struct.calcsize(data_format)))
        
        depth_data = np.array(data).reshape(depth_height, depth_width)
        for i in range(depth_height):
            for j in range(depth_width):
                if depth_data[i, j] > 1150:
                    depth_data[i, j] = 0
        return depth_data
    else:
        print("No bin file founded")   
    

# Lấy các giá trị trong file tham số
def get_parameter(param_path):
    if(os.path.exists(param_path)):
        with open(param_path, 'r') as f:
            parameters = f.read().split(',')

        name = parameters[0] + "," +parameters[1]
        depth_scale = float(parameters[2])
        depth_width = int(parameters[3])
        depth_height = int(parameters[4])

        depth_cx = float(parameters[5])
        depth_cy = float(parameters[6])

        depth_fx = float(parameters[7])
        depth_fy = float(parameters[8])

        return name, depth_scale, depth_width, depth_height, depth_cx, depth_cy, depth_fx, depth_fy
    
    else:
        print("No param text founded")

# Hiển thị bằng matplotlib.pyplot
def plt_visualize(depth_data, depth_width, depth_height):
    x = np.arange(0, depth_width)
    y = np.arange(0, depth_height)
    x, y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.plot_surface(x, y, depth_data)
    plt.show()

# Hiển thị bằng Open3D
def o3d_visualize(depth_data, depth_width, depth_height, depth_scale, depth_cx, depth_cy, depth_fx, depth_fy): 
    depth = depth_data / depth_scale
    y, x = np.mgrid[0:depth_height, 0:depth_width]
    x3d = (x - depth_cx) * depth / depth_fx
    y3d = (y - depth_cy) * depth / depth_fy
    z3d = depth


    points3d = np.dstack((x3d, y3d, z3d))
    points3d = points3d.reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points3d)


    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    view_control = vis.get_view_control()
    view_control.set_front([0, 0, -1])
    view_control.set_up([0, 1, 0])
    vis.run()
    vis.destroy_window()


# Lấy ảnh xám từ ma trận depth
def get_8bit_image(depth_data):
    min_val = np.min(depth_data)
    max_val = np.max(depth_data)
    gray_image = ((depth_data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return gray_image

# Lấy thanh ngang đầu tiên
def get_first_bar(depth_data):
    first_bar = np.zeros_like(depth_data)
    x_sum = 0
    count = 0
    z_sum = 0
    y_sum = 0

    for i in range (480):
        for j in range (400):
            if (depth_data[i, j] <  520) and (depth_data[i, j] != 0):
                first_bar[i, j] = depth_data[i, j]
                x_sum += j
                y_sum += i
                z_sum += depth_data[i, j]
                count += 1       

    x_avr = int(x_sum / count)
    z_avr = int(z_sum / count)
    y_avr = int(y_sum / count)
    firstbar_avr = np.array([x_avr, y_avr, z_avr])

    return first_bar, firstbar_avr  
    # Các phần tử trong thanh ngang đầu tiên, phần tử trung bình ở giữa

# Lấy thanh ngang thứ hai
def get_second_bar(depth_data):
    second_bar = np.zeros_like(depth_data)
    x_sum = 0
    count = 0
    z_sum = 0
    y_sum = 0

    for i in range (480):
        for j in range (400, 848):
            if (depth_data[i, j] <  655) and (depth_data[i, j] > 645):
                second_bar[i, j] = depth_data[i, j]
                x_sum += j
                y_sum += i
                z_sum += depth_data[i, j]
                count += 1

    x_avr = int(x_sum / count)
    z_avr = int(z_sum / count)
    y_avr = int(y_sum / count)
    secondbar_avr = np.array([x_avr, y_avr, z_avr])

    return second_bar, secondbar_avr    
    # Các phần tử trong thanh ngang thứ 2 và phần tử trung bình ở giữa

# Hiển thị 2 thanh ngang 
def plt_bars_visualize(bars, depth_width, depth_height, firstbar_avr, secondbar_avr):
    x1 = firstbar_avr[0]
    y1 = firstbar_avr[1]
    z1 = firstbar_avr[2]

    x2 = secondbar_avr[0]
    y2 = secondbar_avr[1]
    z2 = secondbar_avr[2]

    direction_vector = np.array([y2- y1, x1- x2, 0])

    point1 = firstbar_avr + (200 / np.linalg.norm(direction_vector)) * direction_vector
    point2 = firstbar_avr - (200 / np.linalg.norm(direction_vector)) * direction_vector 
    point3 = secondbar_avr + (200 / np.linalg.norm(direction_vector)) * direction_vector 
    point4 = secondbar_avr - (200 / np.linalg.norm(direction_vector)) * direction_vector 
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    x = np.arange(0, depth_width)
    y = np.arange(0, depth_height)
    x, y = np.meshgrid(x, y)

    ax.scatter(x1, y1, z1, c='red', marker='o')
    ax.scatter(x2, y2, z2, c ='red', marker='o')
    ax.scatter(point1[0], point1[1], point1[2] + 100, c='g', marker='x', label = "Point 1")
    ax.scatter(point2[0], point2[1], point2[2] + 100, c='g', marker='x', label = "Point 2")
    ax.scatter(point3[0], point3[1], point3[2] + 100, c='g', marker='x', label = "Point 3")
    ax.scatter(point4[0], point4[1], point4[2] + 100, c='g', marker='x', label = "Point 4")

    ax.plot([x1, x2], [y1, y2], [z1, z2], c='r', label = 'Line')
    ax.plot_surface(x, y, bars)

    plt.show()

# Lấy góc lệch giữa 2 thanh ngang
def angular_deviation(firstbar_avr, secondbar_avr):
    x1 = firstbar_avr[0]
    y1 = firstbar_avr[1]
    z1 = firstbar_avr[2]

    x2 = secondbar_avr[0]
    y2 = secondbar_avr[1]
    z2 = secondbar_avr[2]

    a = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    tan_alpha = abs(z1 - z2) / a
    tan_beta = abs(y2 - y1) / abs(x2 - x1)
    return tan_alpha, tan_beta      
    # alpha: Góc lệch giữa 2 điểm trung bình và mặt phẳng xOy
    # beta: Góc lệch giữa 2 điểm trung bình và mặt phảng xOz



def get_largest_area(depth_data):
    depth_data = get_8bit_image(depth_data)
    depth_data = cv2.GaussianBlur(depth_data, (7,7), 0)
    _, thresholded_image = cv2.threshold(depth_data, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(thresholded_image)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    result_image = cv2.bitwise_and(thresholded_image, thresholded_image, mask=mask)
    return result_image

def get_backbone(depth_data):
    for col_index in range(depth_data.shape[1]):
        column = depth_data[:, col_index]
        nonzero_elements = column[column > 0]  
        if len(nonzero_elements) > 0:
            min_nonzero_element = np.min(nonzero_elements)  
            depth_data[column != min_nonzero_element, col_index] = 0
    return depth_data
    # return mảng 2 chiều

def get_backbone_line(depth_data):
    backbone_line = np.zeros(depth_data.shape[1])
    for col_index in range(depth_data.shape[1]):
        column = depth_data[:, col_index]
        non_zero_elements = column[column != 0]
        if non_zero_elements.size > 0:
            backbone_line[col_index] = np.mean(non_zero_elements)
    backbone_line = backbone_line[backbone_line != 0]
    max_val = np.max(backbone_line)
    backbone_line = max_val - backbone_line
    return backbone_line
    # return mảng 1 chiều

def get_tail(backbone_line):
    length = len(backbone_line)
    min_val = np.min(backbone_line)
    min_index = np.argmin(backbone_line)
    if min_index < length - 50:
        return min_val, min_index
        # return vị trí cột và độ cao của cuống đuôi

