# coding: UTF-8
import numpy as np
import cv2
from train import Model

cascade_path = "/usr/local/opt/opencv/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(cascade_path)

model = Model()
model.load()

output_count = 0

def DetectFace(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))

    return facerect

def FaceRecoginition(frame, facerect):
    global output_count

    if len(facerect) == 0:
        return

    print('face detect')
    for rect in facerect:
        x, y = rect[0:2]
        w, h = rect[2:4]

        image = frame[y - 10:y + h, x: x + w]
        result = model.predict(image)
        
        if result == 0:
            print('ns')
            cv2.imwrite("./output/ns/" + str(output_count) + ".jpg", image)
        else:
            print('ohter')
            cv2.imwrite("./output/other/" + str(output_count) + ".jpg", image)
        
        output_count += 1

if __name__ == '__main__':
    # カメラからのキャプチャ指定
    cap = cv2.VideoCapture(0)
    
    # コーデックの指定
    fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 

    while(True):

        # キャプチャ画像を取り出す
        _, frame = cap.read()

        facerect = DetectFace(frame)
        FaceRecoginition(frame, facerect)

        # 画像を画面に出力する
        cv2.imshow('frame',frame)

        # "q"が押されたら抜ける
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 後始末
    cap.release()
    cv2.destroyAllWindows()