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
def get_first_bar(depth_data, depth_height, depth_width):
    upper_left = np.array([0, 0, 0])
    bottom_left = np.array([0, 0, 0])
    first_bar = np.zeros_like(depth_data)
    for i in range(depth_height):
        for j in range(int(depth_width / 2)):
            if depth_data[i, j] < 520 and depth_data[i, j] != 0:
                first_bar[i, j] = depth_data[i, j]

    first_bar = get_largest_area(first_bar)
    
    for i in range(depth_height):
        for j in range(depth_width):
            if first_bar[depth_height - i - 1, j] != 0:
                upper_left[0] = j
                upper_left[1] = depth_height - i - 1
                upper_left[2] = first_bar[depth_height - i - 1, j]
                
            if first_bar[i, j] != 0:
                bottom_left[0] = j
                bottom_left[1] = i 
                bottom_left[2] = first_bar[i, j]
    if bottom_left[1] - upper_left[1] < 150:
        upper_left[1] = bottom_left[1]
        bottom_left[1] = depth_height
    left_avr = ((upper_left + bottom_left) / 2).astype(np.int32)
    return first_bar, left_avr, upper_left, bottom_left

# Lấy thanh ngang thứ hai
def get_second_bar(depth_data, depth_height, depth_width, leftbar_avr, leftbar):
    for i in range(depth_height):
        for j in range(depth_width):
            if depth_data[i, j] > leftbar_avr[2] + 0.3 * j:
                depth_data[i, j] = 0
     
    depth_data = depth_data - leftbar
    depth_data = get_largest_area(depth_data)  
    img = depth_data.copy()
    img[img != 0] = 255
    row = img[leftbar_avr[1],:]
    rightbar_avr = np.array([0, 0, 0])
    upper_right = np.array([0, 0])
    bottom_right = np.array([0, 0])
    for i in range(depth_width):
        if row[i] == 255:
            rightbar_avr[0] = i + 20
            rightbar_avr[1] = leftbar_avr[1]
            col = img[:, rightbar_avr[0]]
            for j in range(depth_height):
                if col[j] == 255:
                    upper_right[0] = rightbar_avr[0]
                    upper_right[1] = j
                    break
            for j in range(depth_height):
                if col[depth_height - j - 1] == 255:
                    bottom_right[0] = rightbar_avr[0]
                    bottom_right[1] = depth_height - j - 1
                    break
            break
    rightbar_avr[2] = depth_data[rightbar_avr[1], rightbar_avr[0]]
    return rightbar_avr, upper_right, bottom_right


# Chỉnh 4 góc cho khớp với thân
def get_4_corner_of_body(upper_left, bottom_left, upper_right, bottom_right, depth_height, depth_width):    
    if upper_left[1] < upper_right[1]:
        upper_left[1] = upper_right[1]
    else:
        upper_right[1] = upper_left[1]

    if upper_right[0] - upper_left[0] < depth_height / 2:
        upper_left = upper_right
        bottom_left = bottom_right
        upper_right = np.array([depth_width - 5, upper_right[1]])
        bottom_right = np.array([depth_width - 5, bottom_right[1]])


    return upper_left, bottom_left, upper_right, bottom_right
    

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


# Lọc ra phần có diện tích với nhất, sau khi dùng Otsu
def get_largest_area(depth_data):
    img = get_8bit_image(depth_data)
    img = cv2.GaussianBlur(img, (7,7), 0)
    _, thresholded_image = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(thresholded_image)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    result_image = cv2.bitwise_and(thresholded_image, thresholded_image, mask=mask)
    mask1 = (result_image == 0)
    depth_data[mask1] = 0
    return depth_data
    # return result_image


# Lấy ra phần xương sống
def get_backbone_line(depth_data):
    for col_index in range(depth_data.shape[1]):
        column = depth_data[:, col_index]
        nonzero_elements = column[column > 0]  
        if len(nonzero_elements) > 0:
            min_nonzero_element = np.min(nonzero_elements)  
            depth_data[column != min_nonzero_element, col_index] = 0
    backbone_line = np.zeros(depth_data.shape[1])
    for col_index in range(depth_data.shape[1]):
        column = depth_data[:, col_index]
        non_zero_elements = column[column != 0]
        if non_zero_elements.size > 0:
            backbone_line[col_index] = np.mean(non_zero_elements)
    # backbone_line = backbone_line[backbone_line != 0]
    max_val = np.max(backbone_line)
    backbone_line = max_val - backbone_line
    return backbone_line
    
# Hiển thị phần xương sống, bụng, đuôi 
def backbone_visualize(backbone_line):
    # length = len(backbone_line)
    min_val = np.min(backbone_line[backbone_line != 0])
    min_index = np.argmin(backbone_line)
    belly_index = min_index - 200
    belly_val = backbone_line[belly_index]
    plt.scatter(min_index, min_val, c='red', marker='x', label = 'Cuống đuôi')
    plt.scatter(belly_index, belly_val, c='green', marker='x', label = 'Vị trí bụng')
    plt.plot(backbone_line)
    plt.xlabel('Cột')
    plt.ylabel('Độ cao')
    plt.title('Xương sống')
    plt.legend()
    plt.show()



# Vẽ ảnh xám chứa 4 góc và 2 điểm trung bình của thanh
def draw_corner_and_bar_avr(depth_data, upper_left, bottom_left, upper_right, bottom_right, leftbar_avr, rightbar_avr):
    img = get_8bit_image(depth_data)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    cv2.circle(img, (upper_left[0], upper_left[1]), 5, (0,255,0), -1)
    cv2.circle(img, (bottom_left[0], bottom_left[1]), 5, (0,255,0), -1)
    cv2.circle(img, (upper_right[0], upper_right[1]), 5, (0,255,0), -1)
    cv2.circle(img, (bottom_right[0], bottom_right[1]), 5, (0,255,0), -1)

    cv2.circle(img, (leftbar_avr[0], leftbar_avr[1]), 5, (0,0,255), -1)
    cv2.circle(img, (rightbar_avr[0], rightbar_avr[1]), 5, (0,0,255), -1)
 
    cv2.imshow('window', img)
    cv2.waitKey(0)

