from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import cv2,os,time,keras
import numpy as np
from PIL import Image
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

keras.backend.tensorflow_backend.set_session(get_session())

model_path = os.path.join('.', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')
model = models.load_model(model_path, backbone_name='resnet50')
# _ = model.predict(np.zeros((1,32,32,3)))
labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
def distance(point_a,point_b):
    x1,y1 = point_a
    x2,y2 = point_b
    return np.sqrt(np.square(x1-x2)+np.square(y1-y2))
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

def process(image):
    draw = image.copy()
    h, w, _ = draw.shape
    # draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    image = preprocess_image(image)
    image, scale = resize_image(image)

    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)

    boxes /= scale
    # visualize detections
    centers = []
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        # scores are sorted so we can break
        if score < 0.5 or labels_to_names[label] not in ['person']:
            break
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

        w, h = (x2 - x1), (y2 - y1)
        centers.append([x1,y1,w,h])
        color = label_color(label)
        b = box.astype(int)
        draw_box(draw, b, color=color)

        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw, b, caption)
    return draw,centers

def main():
    # video_path = '1.mp4'
    output_path = 'x.mp4'
    vid = cv2.VideoCapture(0)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    font = cv2.FONT_HERSHEY_SIMPLEX
    prev_time = timer()
    in_num = 0
    out_num = 0
    num = 0
    data = {}
    while True:
        print(in_num, out_num, "in_numin_numin_numin_numin_numin_numoutin_numin_numin_numin_numin_num")

        return_value, frame = vid.read()

        if frame is None:
            break

        result, centers = process(frame)
        for center in centers:
            x, y, w, h = center
            #         cv2.circle(result,(x,y), 30, (0,0,255), -1)
            if len(data) == 0:
                data[f'{x},{y},{w},{h},{num}'] = [x, y, w, h, x, y, w, h]  # 最初检测点 最后检测点
                continue
            for key in list(data.keys()):
                tx, ty, tw, th, tn = key.split(',')
                tx, ty, tw, th, tn = int(tx), int(ty), int(tw), int(th), int(tn)
                if num - tn > 4:
                    del data[key]
                    continue
                else:
                    print('distance', overlap([x, y, w, h], [tx, ty, tw, th]))
                    if overlap([x, y, w, h], [tx, ty, tw, th]) > 0.5:
                        value = data[key]
                        value[4], value[5], value[6], value[7] = x, y, w, h
                        del data[key]
                        data[f'{x},{y},{w},{h},{num}'] = value
                    else:
                        data[f'{x},{y},{w},{h},{num}'] = [x, y, w, h, x, y, w, h]
        print(data.keys(),"data.keys()data.keys()data.keys()")
        for key in list(data.keys()):
            value = data[key]
            y1 = value[1] + value[3] // 2
            y2 = value[5] + value[7] // 2
            #print(y1,y2,"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
            if y1 < 700 and y2 >= 700:
                del data[key]
                out_num += 1
                continue
            elif y1 > 700 and y2 < 700:
                del data[key]
                in_num += 1
                continue
            elif num == video_fps:
                num = 0
                tx, ty, tn = key.split(',')
                if video_fps - int(tn) > 4:
                    del data[key]
                    continue
                else:
                    del data[key]
                    data[f'{tx},{ty},{num}'] = value
        cv2.line(result, (0, 700), (800, 700), (0, 0, 255), 5)
        cv2.putText(result, f'in: {in_num}  out: {out_num}', (50, 780), font, 1.5, (0, 0, 255), 2)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()