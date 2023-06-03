import cv2
import mediapipe as mp
import time
mpface=mp.solutions.face_detection
face=mpface.FaceDetection()
mpDraw=mp.solutions.drawing_utils

cap=cv2.VideoCapture(0)
pTime=0
while True:
    _,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=face.process(imgRGB)
    if results.detections:
        for id,detection in enumerate(results.detections):
                   bboxC = detection.location_data.relative_bounding_box
                   ih, iw, ic = img.shape
                   bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                          int(bboxC.width * iw), int(bboxC.height * ih)
                   cv2.rectangle(img, bbox, (0, 255, 0), 2)
                   cv2.putText(img, f'{int(detection.score[0] * 100)}%', (bbox[0], bbox[1] - 20),
                               cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img, f'FPS:{int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 2)

    cv2.imshow('image',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

