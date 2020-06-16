import cv2
import time
from base_camera import BaseCamera
from Process import Process

global sess

class Camera(BaseCamera):
    video_source = 0
    process = Process()

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    # @staticmethod
    def frames(self):
        camera = cv2.VideoCapture(Camera.video_source)
        video_FourCC = int(camera.get(cv2.CAP_PROP_FOURCC))
        video_fps = camera.get(cv2.CAP_PROP_FPS)
        video_size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')
        print("!!! TYPE:", type(video_FourCC), type(video_fps), type(video_size))
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        font = cv2.FONT_HERSHEY_SIMPLEX
        # prev_time = timer()
        in_num = 0
        out_num = 0
        num = 0
        data = {}


        while True:
            # read current frame
            print(in_num, out_num, "in_numin_numin_numin_numin_numin_numoutin_numin_numin_numin_numin_num")
            _, img = camera.read()
            if img is None:
                break
            result, centers = self.process.process(img)
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
                        print('distance', self.process.overlap([x, y, w, h], [tx, ty, tw, th]))
                        if self.process.overlap([x, y, w, h], [tx, ty, tw, th]) > 0.5:
                            value = data[key]
                            value[4], value[5], value[6], value[7] = x, y, w, h
                            del data[key]
                            data[f'{x},{y},{w},{h},{num}'] = value
                        else:
                            data[f'{x},{y},{w},{h},{num}'] = [x, y, w, h, x, y, w, h]
                for key in list(data.keys()):
                    value = data[key]
                    y1 = value[1] + value[3] // 2
                    y2 = value[5] + value[7] // 2
                    # print(y1,y2,"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
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
            print(data.keys(), "data.keys()data.keys()data.keys()")


            # encode as a jpeg image and return it
            yield cv2.imencode('.jpg', img)[1].tobytes()
