import my_lib
import numpy as np
import cv2

def func1(depth_data, depth_width, depth_height):
    x = 0
    y = 0
    z = 0 
    count = 0
    bar1 = depth_data.copy()
    min_val = np.min(bar1[bar1 != 0])
    for i in range(depth_height):
        for j in range(int(depth_width / 2)):
            if bar1[i, j] > min_val + 60:
                bar1[i, j] = 0
            else:
                depth_data[i, j] = 0
                x += j
                y += i
                z += bar1[i, j]
                count += 1
        for j in range(int(depth_width / 2), depth_width):
            if bar1[i, j] > min_val + 60:
                bar1[i, j] = 0
            else:
                depth_data[i, j] = 0  
    bar1[:, 400:] = 0
    bar1_avr = np.array([int(x/count), int(y/count), int(z/count) + 500])
    return depth_data, bar1, bar1_avr

def func2(depth_data, depth_width, depth_height):
    x = 0
    y = 0
    z = 0
    count = 0
    bar2 = depth_data.copy()
    min_val = np.min(bar2[bar2 != 0])
    for i in range(depth_height):
        for j in range(depth_width):
            if bar2[i, j] > min_val + 180:
                bar2[i, j] = 0
            else:
                depth_data[i, j] = 0
                x += j
                y += i
                z += bar1[i, j]
                count += 1
    bar2_avr = np.array([int(x/count), int(y/count), int(z/count)])
    print(min_val)
    return depth_data, bar2, bar2_avr

# Lấy ma trận từ file bin và tham số
bin_path = r'D:\python\wavelet_intern\data\bin2.bin'
param_path = r'D:\python\wavelet_intern\data\213622251778.txt'

name, depth_scale, depth_width, depth_height, depth_cx, depth_cy, depth_fx, depth_fy = my_lib.get_parameter(param_path)
depth_data = my_lib.convert(bin_path, depth_width, depth_height)

depth_data, bar1, bar1_avr = func1(depth_data, depth_width, depth_height)
# depth_data, bar2, bar2_avr = func2(depth_data, depth_width, depth_height)

# img = my_lib.get_16bit_image(depth_data)
# bar1img = my_lib.get_16bit_image(bar1)

# cv2.imshow('window', img)
# cv2.waitKey(0)
fig = my_lib.plt.figure()
ax = fig.add_subplot(111, projection = '3d')
x = np.arange(0, depth_width)
y = np.arange(0, depth_height)
x, y = np.meshgrid(x, y)
ax.scatter(bar1_avr[0], bar1_avr[1], bar1_avr[2], c = 'red', marker = 'o')
ax.plot_surface(x, y, bar1)
my_lib.plt.show()



