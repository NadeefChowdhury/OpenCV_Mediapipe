import mediapipe as mp
import cv2
import time

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
while True:
    success,img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results =  hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for ID, lm in enumerate(handLms.landmark):
                h,w,c = img.shape
                cx,cy= int(lm.x*w), int(lm.y*h)
                print(ID, cx, cy)
                if ID == 0:
                    cv2.circle(img,(cx,cy),20,(0,0,0),cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,str(int(fps)),(10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0,0,255), 3 )        
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL) 
    cv2.imshow("Video", img)
    
    if cv2.waitKey(20) & 0xFF==ord('d'):
        break

cv2.destroyAllWindows()