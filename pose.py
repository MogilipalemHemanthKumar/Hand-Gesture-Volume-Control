import cv2
import mediapipe as mp
mp_pose=mp.solutions.pose
pose =mp_pose.Pose()
mpdraw=mp.solutions.drawing_utils
cap=cv2.VideoCapture(0)
while True:
    _,img=cap.read()
    rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=pose.process(rgb)
    if results.pose_landmarks:
            mpdraw.draw_landmarks(img,results.pose_landmarks,mp_pose.POSE_CONNECTIONS,mpdraw.DrawingSpec(color=(0,0,255)),mpdraw.DrawingSpec(color=(0,255,0)))
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break



