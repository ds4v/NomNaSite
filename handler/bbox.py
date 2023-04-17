import cv2
import numpy as np


# https://github.com/andfanilo/streamlit-drawable-canvas/issues/65
def transform_fabric_box(box):
    scaled_width = box['width'] * box['scaleX']
    scaled_height = box['height'] * box['scaleY']
    return np.array([
        [box['left'], box['top']], 
        [box['left'] + scaled_width, box['top']], 
        [box['left'] + scaled_width, box['top'] + scaled_height], 
        [box['left'], box['top'] + scaled_height]]
    )
    
    
def order_points_clockwise(box_points):
    points = np.array(box_points)
    s = points.sum(axis=1)
    diff = np.diff(points, axis=1)
    quad_box = np.zeros((4, 2), dtype=np.float32)
    quad_box[0] = points[np.argmin(s)]
    quad_box[2] = points[np.argmax(s)]
    quad_box[1] = points[np.argmin(diff)]
    quad_box[3] = points[np.argmax(diff)]
    return quad_box


def order_boxes4nom(boxes):
    return sorted(boxes, key=lambda box: (
        box[:, 0].max(), 
        box[:, 1].min()
    ), reverse=True)
    

def get_patch(page, points):
    points = order_points_clockwise(points)
    page_crop_width = int(max(
        np.linalg.norm(points[0] - points[1]),
        np.linalg.norm(points[2] - points[3]))
    )
    page_crop_height = int(max(
        np.linalg.norm(points[0] - points[3]),
        np.linalg.norm(points[1] - points[2]))
    )
    pts_std = np.float32([
        [0, 0], [page_crop_width, 0], 
        [page_crop_width, page_crop_height],[0, page_crop_height]
    ])
    M = cv2.getPerspectiveTransform(points, pts_std)
    return cv2.warpPerspective(
        page, M, (page_crop_width, page_crop_height), 
        borderMode=cv2.BORDER_REPLICATE, flags=cv2.INTER_CUBIC
    )