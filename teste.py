import cv2
import numpy as np

def draw_circle(event,x,y,flags,param):
    global mouse,started
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),100,(255,0,0),-1)
        mouse = (x,y)
        started = True


mouse, started = (None,None),False

cap = cv2.VideoCapture(0)

# img = np.zeros((512,512,3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    _, img = cap.read()
    cv2.imshow('image',img)
    k = cv2.waitKey(20) & 0xFF
    if k == 27:
        break
    if started == True:
        started = False
        print(mouse)
    