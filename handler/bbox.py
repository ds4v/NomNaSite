import cv2
import numpy as np


def generate_initial_drawing(boxes, size_ratio):
    initial_drawing = {'version': '4.4.0', 'objects': []}
    for box_pts in boxes:
        left = max(box_pts[0][0], box_pts[3][0])
        top = max(box_pts[0][1], box_pts[1][1])
        width = max(box_pts[1][0], box_pts[2][0]) - min(box_pts[0][0], box_pts[3][0])
        height = max(box_pts[2][1], box_pts[3][1]) - min(box_pts[0][1], box_pts[1][1])
        
        initial_drawing['objects'].append({
            'type': 'rect',
            'left': left * size_ratio,
            'top': top * size_ratio,
            'width': width * size_ratio,
            'height': height * size_ratio,
            'fill': 'rgba(76, 175, 80, 0.3)',
            'stroke': 'red',
            'strokeWidth': 2,
            'strokeUniform': True,
            'transparentCorners': False
        })
    return initial_drawing


# https://github.com/andfanilo/streamlit-drawable-canvas/issues/65
def transform_fabric_box(box, size_ratio):
    scaled_width = box['width'] * box['scaleX']
    scaled_height = box['height'] * box['scaleY']
    return np.array([
        [box['left'], box['top']], 
        [box['left'] + scaled_width, box['top']], 
        [box['left'] + scaled_width, box['top'] + scaled_height], 
        [box['left'], box['top'] + scaled_height]
    ]) / size_ratio
    
    
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


def order_boxes4nom(bounding_boxes):
    '''
    Sort bounding boxes for a document whose text is arranged in vertical columns.
    
    The document’s coordinate system is such that (0,0) is the top left and the reading order is:
    - The right column (sentences read top-to-bottom)
    - Followed by the left column (sentences read bottom-to-top)

    bounding_boxes: list of boxes where each box is defined as:
      [ [x1, y1], [x2, y2], [x3, y3], [x4, y4] ]
    with the order of points being [bottom left, bottom right, top right, top left].
    '''
    # Sort bounding boxes by x-coordinates (right to left)
    bounding_boxes.sort(key=lambda box: box[1][0], reverse=True)  # Sort by x2 (right boundary)
    
    # Group bounding boxes into columns
    columns = []
    for box in bounding_boxes:
        placed = False
        for col in columns:
            # Check if the box belongs to an existing column (close x-coordinates)
            if abs(box[1][0] - col[-1][1][0]) < abs(box[0][0] - box[1][0]) * 0.5: # Dynamic threshold based on width
                col.append(box)
                placed = True
                break
        if not placed: columns.append([box])
    
    # Sort each column from top to bottom
    for col in columns:
        col.sort(key=lambda box: box[2][1]) # Sort by y3 (top boundary)
    return sum(columns, []) # Flatten the sorted columns into the final order


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