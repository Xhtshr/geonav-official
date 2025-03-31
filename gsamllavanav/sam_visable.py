import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os
import sys
from scipy.ndimage import sobel
sys.path.append('/data1/XHT/citynav/')

from gsamllavanav.defaultpaths import GDINO_CHECKPOINT_PATH, GDINO_CONFIG_PATH, SAM_CHECKPOINT_PATH, MOBILE_SAM_CHECKPOINT_PATH, GSAM_MAPS_DIR

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

def show_waypoints(coords1, coords2, ax, marker_size=50):
    ax.scatter(coords1[:, 0], coords1[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=0.25)
    # ax.scatter(coords2[:, 0], coords2[:, 1], color='red', marker='o', s=marker_size, edgecolor='white', linewidth=0.25)

from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import rasterio

image = cv2.imread('data/rgbd/ortho_projection_images/cambridge_block_26.png')
cache = rasterio.open('/data1/XHT/citynav/data/rgbd/ortho_projection_images/cambridge_block_26.tif').read(1)
depth = 50.0 - cache
depth_map = depth[..., np.newaxis]
device = 'cuda'
sam = sam_model_registry["vit_h"](SAM_CHECKPOINT_PATH)
sam.to(device=device)
# invoke prediction model
predictor = SamPredictor(sam)
predictor.set_image(image)

input_point = np.array([[2050,1195]])
input_label = np.array([1])
# draw a sign
# plt.figure(figsize=(10,10))
# plt.imshow(image)
# show_points(input_point, input_label, plt.gca())
# plt.axis('on')
# plt.show()

# 用`SamPredictor.predict`进行预测。该模型返回掩码、这些掩码的质量预测和低分辨率的掩码对数，可传递给下一次迭代预测。
# masks, scores, logits = predictor.predict(
#     point_coords=input_point,
#     point_labels=input_label,
#     multimask_output=True, # 是否产生多个掩码
# )
# # 默认产生3个掩码
# print(masks.shape)  # (number_of_masks) x H x W

# # 将3个掩码可视化
# for i, (mask, score) in enumerate(zip(masks, scores)):
#     plt.figure(figsize=(10,10))
#     plt.imshow(image)
#     show_mask(mask, plt.gca())
#     show_points(input_point, input_label, plt.gca())
#     plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
#     plt.axis('off')
    # plt.show()

# 使用排除点
# input_point = np.array([[50, 195], [200, 200]])
# input_label = np.array([1, 0])

# mask_input = logits[np.argmax(scores), :, :]

# masks, scores, _ = predictor.predict(
#     point_coords=input_point,
#     point_labels=input_label,
#     mask_input=mask_input[None, :, :],
#     multimask_output=False,
# )
# for i, (mask, score) in enumerate(zip(masks, scores)):
#     plt.figure(figsize=(10, 10))
#     plt.imshow(image)
#     show_mask(masks, plt.gca())
#     show_points(input_point, input_label, plt.gca())
#     plt.title(f"Score: {score:.3f}", fontsize=18)
#     plt.axis('off')
#     plt.show() 

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.35)))

mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(image)
print(len(masks))
print(masks[0].keys())


# plt.figure(figsize=(10,10))
# # plt.imshow(image)
# show_anns(masks)
# plt.axis('off')
# plt.show() 

centroids = []
for mask in masks:
    bbox = mask['bbox']
    x_min, y_min, width, height = bbox
    centroids_x = x_min + width / 2
    centroids_y = y_min + height / 2
    centroids.append([centroids_x, centroids_y])
print("centroids:", centroids)

edge_points = []
# for mask in masks:
#     bbox = mask['bbox']
#     x_min, y_min, width, height = bbox
#     edge_points.extend([
#         [x_min, y_min],
#         [x_min + width, y_min],
#         [x_min, y_min + height],
#         [x_min + width, y_min + height],
#     ])

plt.figure(figsize=(10,10))
plt.imshow(image)

show_waypoints(np.array(centroids),np.array(edge_points),plt.gca())
plt.axis('off')
plt.show()
# 每条边均匀取样点
# num_samples = 5  # 每条边取样点数
# sampled_points = []
# for mask in masks:
#     bbox = mask['bbox']
#     x_min, y_min, width, height = bbox
#     x_max, y_max = x_min + width, y_min + height

#     # 水平边界取样（上边和下边）
#     top_edge = [(x, y_min) for x in np.linspace(x_min, x_max, num_samples)]
#     bottom_edge = [(x, y_max) for x in np.linspace(x_min, x_max, num_samples)]

#     # 垂直边界取样（左边和右边）
#     left_edge = [(x_min, y) for y in np.linspace(y_min, y_max, num_samples)]
#     right_edge = [(x_max, y) for y in np.linspace(y_min, y_max, num_samples)]

#     # 合并四条边的点
#     sampled_points.extend(top_edge + bottom_edge + left_edge + right_edge)

# print("边界均匀取样点:", sampled_points)

# # 结合深度信息选择特征点
# from scipy.ndimage import sobel

# depth_keypoints = []
# for mask in masks:
#     segmentation = mask['segmentation']
#     print(segmentation.shape)
#     # 在深度图中提取掩码区域
#     print(depth_map.shape)
#     masked_depth = np.squeeze(depth_map) * segmentation
#     # 计算深度梯度
#     grad_x = sobel(masked_depth, axis=1)
#     grad_y = sobel(masked_depth, axis=0)
#     grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
#     # 阈值化，选择梯度变化显著的点
#     threshold = np.percentile(grad_magnitude, 95)  # 选择显著梯度点
#     y_indices, x_indices = np.where(grad_magnitude > threshold)
#     depth_keypoints.extend(list(zip(x_indices, y_indices)))

# print("深度显著点:", depth_keypoints)