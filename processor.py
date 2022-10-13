import cv2
import pyclipper
import numpy as np
from shapely.geometry import Polygon


class PostProcessor:
    def __init__(self, thresh=0.3, min_box_score=0.7, max_candidates=500, shrink_ratio=1.1, dilate_ratio=1.5):
        self.min_size = 3
        self.thresh = thresh
        self.min_box_score = min_box_score
        self.max_candidates = max_candidates
        self.shrink_ratio = shrink_ratio
        self.dilate_ratio = dilate_ratio
        

    def __call__(self, binarize_map, batch_true_sizes):
        segmentation = binarize_map > self.thresh
        batch_boxes, batch_scores = [], []
        
        for batch_idx, image_size in enumerate(batch_true_sizes):
            boxes_in_image, scores = self.bitmap2quads(binarize_map[batch_idx], segmentation[batch_idx], image_size)
            batch_boxes.append(boxes_in_image)
            batch_scores.append(scores)
        return batch_boxes, batch_scores
    

    def bitmap2quads(self, pred, bitmap, image_size):
        assert len(bitmap.shape) == 2
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        boxes, scores = [], []
        height, width = bitmap.shape
        original_height, original_width = image_size

        for contour in contours[:self.max_candidates]:
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size: continue
            
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.min_box_score > score: continue

            box = self.shrink_and_dilate(points)
            box, sside = self.get_mini_boxes(box.reshape(-1, 1, 2))
            if sside < self.min_size + 2: continue
            
            box = np.array(box)
            box[:, 0] = np.clip(np.round(box[:, 0] / width * original_width), 0, original_width)
            box[:, 1] = np.clip(np.round(box[:, 1] / height * original_height), 0, original_height)
            boxes.append(box)
            scores.append(score)
        return boxes, scores


    # https://www.panicbyte.xyz/DBNet
    def shrink_and_dilate(self, box):
        poly = Polygon(box)
        shrink_distance = poly.area * (1 - self.shrink_ratio**2) / poly.length
        clipper = pyclipper.PyclipperOffset()
        clipper.AddPath(box, pyclipper.JT_SQUARE, pyclipper.ET_CLOSEDPOLYGON)
        
        shrink_box = clipper.Execute(-shrink_distance)
        assert len(shrink_box) != 0
        shrink_box = shrink_box[0]

        poly = Polygon(shrink_box)
        dilate_distance = poly.area * self.dilate_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(shrink_box, pyclipper.JT_SQUARE, pyclipper.ET_CLOSEDPOLYGON)
        
        expanded = np.array(offset.Execute(dilate_distance)[0])
        return expanded


    def get_mini_boxes(self, contour):
        try: bounding_box = cv2.minAreaRect(contour)
        except: return [], 0
        
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])
        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        
        if points[1][1] > points[0][1]: index_1, index_4 = 0, 1
        else: index_1, index_4 = 1, 0
            
        if points[3][1] > points[2][1]: index_2, index_3 = 2, 3
        else: index_2, index_3 = 3, 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])


    def get_extremum_points(self, box_points, image_height, image_width):
        xmin = np.clip(np.floor(box_points[:, 0].min()).astype(np.int32), 0, image_width - 1)
        ymin = np.clip(np.floor(box_points[:, 1].min()).astype(np.int32), 0, image_height - 1)
        xmax = np.clip(np.ceil(box_points[:, 0].max()).astype(np.int32), 0, image_width - 1)
        ymax = np.clip(np.ceil(box_points[:, 1].max()).astype(np.int32), 0, image_height - 1)
        return xmin, ymin, xmax, ymax
    
    
    def box_score_fast(self, bitmap, box):
        h, w = bitmap.shape[:2]
        xmin, ymin, xmax, ymax = self.get_extremum_points(box, h, w)
        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        
        new_box = box.copy()
        new_box[:, 0] = new_box[:, 0] - xmin
        new_box[:, 1] = new_box[:, 1] - ymin
        
        cv2.fillPoly(mask, new_box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
