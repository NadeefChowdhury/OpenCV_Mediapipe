import mediapipe as mp
import time
import cv2


mpFaceDetection= mp.solutions.face_detection
faceDetection = mpFaceDetection.FaceDetection(min_detection_confidence=0.9)
mpDraw = mp.solutions.drawing_utils
pTime=0
cap = cv2.VideoCapture('Video.mp4')
while True:
    success, img = cap.read()
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results =  faceDetection.process(imgRGB)
    if results.detections:
        for ID, detection in enumerate(results.detections):
           mpDraw.draw_detection(img, detection)
           print(detection.score)
           bboxC = detection.location_data.relative_bounding_box
           ih, iw, ic = img.shape
           bbox = int(bboxC.xmin * iw),int(bboxC.ymin * ih), \
                  int(bboxC.width * iw),int(bboxC.height * ih)
           circleCoordinates = int(bboxC.xmin * iw + (bboxC.width * iw)/2), int(bboxC.ymin * ih + (bboxC.height * ih)/2)
           cv2.rectangle(img,bbox,(0,0,255), 10)
           cv2.circle(img, circleCoordinates,20,(255,255,255),cv2.FILLED)
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL) 
    cTime = time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,str(int(fps)),(10, 150), cv2.FONT_HERSHEY_PLAIN, 10, (0,0,255), 3 )        
    cv2.imshow("Video", img)
    if cv2.waitKey(20) & 0xFF==ord('d'):
        break

cv2.destroyAllWindows()