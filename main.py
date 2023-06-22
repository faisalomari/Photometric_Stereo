import numpy as np
import cv2
from scipy.ndimage import convolve
import math

def getNeighbors(i, j, image, length):
    rows, cols = image.shape
    half_length = math.floor(length/2)
    m1 = image[max(0, i - half_length):min(rows, i + half_length + 1), max(0, j - half_length):min(cols, j + half_length + 1)]
    smooth = m1.flatten()
    if (length - smooth.size) <= 0:
        padded_m1 = smooth
    else:
        padded_m1 = np.pad(smooth, (0, (length - smooth.size)), constant_values=0)
    return padded_m1

def transformation(image, window_size, length):
    height, width = image.shape
    image_census = np.zeros((height, width, window_size), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            center_pixel = image[i, j]
            V_ne = getNeighbors(i, j, image, window_size)
            V = []
            for l in range(window_size):
                if V_ne[l] < center_pixel:
                    V.append(0)
                else:
                    V.append(1)
            image_census[i, j] = V[:length]
    return image_census

def calculate_volume_cost(left_image, right_image, max_disparity, aggregation_window, direction):
    left_census = transformation(left_image, aggregation_window,aggregation_window**2 - 1)
    right_census = transformation(right_image, aggregation_window,aggregation_window**2 - 1)
    height, width = left_image.shape
    volume_cost = np.zeros((int(height), int(width), int(max_disparity)))
    if(direction == "left"):
        for j in range(height):
            for i in range(width):
                for d in range(max_disparity):
                    if i - d < 0:
                        volume_cost[j, i, d] = max_disparity
                    else:
                        distance = sum(left_bit != right_bit for left_bit, right_bit in zip(left_census[j ,i], right_census[j,i - d]))
                        volume_cost[j, i, d] = distance
    elif(direction == "right"):
        for j in range(height):
            for i in range(width):
                for d in range(max_disparity):
                    if i + d >= width:
                        volume_cost[j, i, d] = max_disparity
                    else:
                        distance = sum(left_bit != right_bit for left_bit, right_bit in zip(left_census[j ,i], right_census[j,i + d]))
                        volume_cost[j, i, d] = distance
    else:
        print("DIRECTION ERROR!")
    return volume_cost

def local_aggregation(volume_cost, aggregation_window):
    width, height, disparity_range = volume_cost.shape
    window_size = int(aggregation_window)
    window_size = np.ones((window_size, window_size)) / (window_size**2)
    cost = np.zeros_like(volume_cost)
    for d in range(disparity_range):
        cost[:, :, d] = convolve(volume_cost[:, :, d], window_size)
    return cost

def winner_takes_all(volume_cost):
    height, width, max_disparity = volume_cost.shape
    height = int(height)
    width = int(width)
    new_image = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            index = 0
            value = max_disparity
            for d in range(int(max_disparity)):
                if(volume_cost[i,j,d] <= value):
                    value = volume_cost[i,j,d]
                    index = d
            new_image[i,j] = index
    return new_image

def test_consistency(disp_L, disp_R, direction):
    height, width = disp_R.shape[:2]
    consistency_mask = np.zeros_like(disp_L)
    if(direction == "left"):
        for i in range(height):
            for j in range(width):
                disparity_l = int(disp_L[i, j])
                if j - disparity_l >= 0:
                    disparity_r = disp_R[i, j - disparity_l]
                    if np.abs(disparity_l - disparity_r) < 1:
                        consistency_mask[i, j] = disp_L[i, j]
    elif(direction == "right"):
        for i in range(height):
            for j in range(width):
                disparity_r = int(disp_R[i, j])
                if j + disparity_r < width:
                    disparity_l = disp_L[i, j + disparity_r]
                    if np.abs(disparity_r - disparity_l) < 1:
                        consistency_mask[i, j] = disp_R[i, j]
    return consistency_mask

def norm(image):
    return cv2.normalize(image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

def calculate_disparity_map(left_image, right_image, max_disparity, aggregation_window):
    # Calculate volume cost
    print("Calculate volume cost")
    left_volume_cost = calculate_volume_cost(left_image, right_image, max_disparity, aggregation_window, direction="left")
    right_volume_cost = calculate_volume_cost(right_image, left_image, max_disparity, aggregation_window, direction="right")
    # Perform local aggregation
    print("Perform local aggregation")
    left_aggregated_disparity = local_aggregation(left_volume_cost, 25)
    right_aggregated_disparity = local_aggregation(right_volume_cost, 25)
    # Winner-takes-all
    print("Winner-takes-all")
    left_disparity_map = winner_takes_all(left_aggregated_disparity)
    right_disparity_map = winner_takes_all(right_aggregated_disparity)
    # Test consistency
    print("Test consistency")
    left_disparity_map2 = test_consistency(left_disparity_map,right_disparity_map, "left")
    right_disparity_map2 = test_consistency(left_disparity_map,right_disparity_map, "right")
    #Normalization
    print("Norm")
    final_dispL = norm(left_disparity_map2)
    final_dispR = norm(right_disparity_map2)
    print("FINISHING")
    return final_dispL,final_dispR

path = 'Photometric_Stereo\data\set_5\\'
saving_path = 'Photometric_Stereo\Results\set_5\\'
#Load left and right images
left_image = cv2.imread(path + 'im_left.jpg')
left_imageGray = cv2.imread(path + 'im_left.jpg', 0)
right_image = cv2.imread(path +'im_right.jpg')
right_imageGray = cv2.imread(path + 'im_right.jpg', 0)
max_disparity = int(np.loadtxt(path + 'max_disp.txt'))
aggregation_window = 9
disparity_left, disparity_right = calculate_disparity_map(left_imageGray, right_imageGray, max_disparity, aggregation_window)
cv2.imwrite(saving_path + 'disparity_left.jpg', disparity_left)
cv2.imwrite(saving_path + 'disparity_right.jpg', disparity_right)

#Intrinsic matrix K
K = np.loadtxt(path + 'K.txt')
baseline = 0.1
focal_length = K[0][0]
left_depth_map = (baseline * focal_length) / (disparity_left+ 1e-6)
right_depth_map = (baseline * focal_length) / (disparity_right+ 1e-6)
cv2.imwrite(saving_path + 'depth_left.jpg', left_depth_map*255)
cv2.imwrite(saving_path + 'depth_right.jpg', right_depth_map*255)

R = np.array([[1,0,0,0],
              [0,1,0,0],
              [0,0,1,0]], dtype=np.float32)

T = np.array([[1,0,0,0],
              [0,1,0,0],
              [0,0,1,0],
              [0,0,0,1]], dtype=np.float32)
baseline = 1
points_3d = []
points_2d = []
 
#2D-3D
height, width = left_depth_map.shape
for v in range(height):
    for u in range(width):
        depth = left_depth_map[v, u]
        if depth > 0:
            point_3d = np.linalg.inv(K) @ np.array([u, v, 1]) * depth
            point_2d = K @ (point_3d / point_3d[2])
            points_3d.append(point_3d)
            points_2d.append(point_2d[:2])

points_3d = np.array(points_3d)
points_2d = np.array(points_2d)

synthesized_images = []

#11 camera positions
for i in range(11):
    # Update T by adding the distance to x
    T[0][3] = -0.01 * (i+1)
    RT = R @ T
    P = np.matmul(K, RT)
    reprojected_2d = []
    for j in range(points_3d.shape[0]):
        point_3d = np.append(points_3d[j], [1.0])
        # print(point_3d.shape)
        point_3d_world = P @ point_3d
        # point_3d_world = K @ point_3d_world
        point_2d = point_3d_world / point_3d_world[2]
        reprojected_2d.append(point_2d[:2])
    reprojected_2d = np.array(reprojected_2d)

    synthesized_image = np.zeros_like(left_image)
    for k, point in enumerate(reprojected_2d):
        x, y = point.round().astype(int)
        if 0 <= x < synthesized_image.shape[1] and 0 <= y < synthesized_image.shape[0]:
            synthesized_image[y, x] = left_image[y, x]

    synthesized_images.append(synthesized_image)

# Save the synthesized images
for i, image in enumerate(synthesized_images):
    cv2.imwrite(saving_path + 'synth_' + str(i+1) + '.jpg', image)
    cv2.imshow(f"Synthesized Image {i}", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
