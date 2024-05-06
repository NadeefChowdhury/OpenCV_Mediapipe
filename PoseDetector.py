import cv2
import mediapipe as mp
import time


mpPose= mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils
pTime=0
cap = cv2.VideoCapture('Video.mp4')
while True:
    success, img = cap.read()
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results =  pose.process(imgRGB)
    if results.pose_landmarks:
        for ID, lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            cx,cy= int(lm.x*w), int(lm.y*h)
            print(ID, cx, cy)
            if ID == 0:
                cv2.circle(img,(cx,cy),20,(0,0,0),cv2.FILLED)
        
        
        mpDraw.draw_landmarks(img,results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    
    
    
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL) 
    cTime = time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,str(int(fps)),(10, 150), cv2.FONT_HERSHEY_PLAIN, 10, (0,0,255), 3 )        
    cv2.imshow("Video", img)
    if cv2.waitKey(20) & 0xFF==ord('d'):
        break

cv2.destroyAllWindows()