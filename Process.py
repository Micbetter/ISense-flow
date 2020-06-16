import numpy as np
import keras
import tensorflow as tf
import os
import time
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color


class Process(object):

    def __init__(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        keras.backend.tensorflow_backend.set_session(sess)
        model_path = os.path.join('.', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')
        self.model = models.load_model(model_path, backbone_name='resnet50')
        self.labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',
                           7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign',
                           12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
                           19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack',
                           25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
                           31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
                           36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
                           41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple',
                           48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
                           54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed',
                           60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
                           66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
                           72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear',
                           78: 'hair drier', 79: 'toothbrush'}

    @staticmethod
    def distance(point_a, point_b):
        x1, y1 = point_a
        x2, y2 = point_b
        return np.sqrt(np.square(x1 - x2) + np.square(y1 - y2))

    @staticmethod
    def overlap(box1, box2):
        """
        Check the overlap of two boxes
        """
        endx = max(box1[0] + box1[2], box2[0] + box2[2])
        startx = min(box1[0], box2[0])
        width = box1[2] + box2[2] - (endx - startx)

        endy = max(box1[1] + box1[3], box2[1] + box2[3])
        starty = min(box1[1], box2[1])
        height = box1[3] + box2[3] - (endy - starty)

        if (width <= 0 or height <= 0):
            return 0
        else:
            Area = width * height
            Area1 = box1[2] * box1[3]
            Area2 = box2[2] * box2[3]
            ratio = Area / (Area1 + Area2 - Area)

            return ratio

    # @staticmethod
    def process(self, image):
        draw = image.copy()
        h, w, _ = draw.shape
        # draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        image = preprocess_image(image)
        image, scale = resize_image(image)

        start = time.time()
        boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)

        boxes /= scale
        # visualize detections
        centers = []
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < 0.5 or self.labels_to_names[label] not in ['person']:
                break
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

            w, h = (x2 - x1), (y2 - y1)
            centers.append([x1, y1, w, h])
            color = label_color(label)
            b = box.astype(int)
            draw_box(draw, b, color=color)

            caption = "{} {:.3f}".format(self.labels_to_names[label], score)
            draw_caption(draw, b, caption)
        return draw, centers