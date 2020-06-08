import cv2
import numpy as np
import detection as det
 
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
 
deadZone=100
global imgContour
 
def empty(a):
	pass
 
cv2.namedWindow("HSV")
cv2.resizeWindow("HSV",640,240)
cv2.createTrackbar("HUE Min","HSV",23,179,empty)
cv2.createTrackbar("HUE Max","HSV",95,255,empty)
cv2.createTrackbar("SAT Min","HSV",129,255,empty)
cv2.createTrackbar("SAT Max","HSV",255,255,empty)
cv2.createTrackbar("VALUE Min","HSV",0,255,empty)
cv2.createTrackbar("VALUE Max","HSV",255,255,empty)
 
cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters",640,240)
cv2.createTrackbar("Threshold1","Parameters",0,255,empty)
cv2.createTrackbar("Threshold2","Parameters",94,255,empty)
cv2.createTrackbar("MinArea","Parameters",848,30000,empty)
cv2.createTrackbar("Area Min","Parameters",2360,30000, empty)
cv2.createTrackbar("Area Max","Parameters",5093,30000, empty)
 
 
def stackImages(scale,imgArray):
	rows = len(imgArray)
	cols = len(imgArray[0])
	rowsAvailable = isinstance(imgArray[0], list)
	width = imgArray[0][0].shape[1]
	height = imgArray[0][0].shape[0]
	if rowsAvailable:
		for x in range ( 0, rows):
			for y in range(0, cols):
				if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
					imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
				else:
					imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
				if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
		imageBlank = np.zeros((height, width, 3), np.uint8)
		hor = [imageBlank]*rows
		hor_con = [imageBlank]*rows
		for x in range(0, rows):
			hor[x] = np.hstack(imgArray[x])
		ver = np.vstack(hor)
	else:
		for x in range(0, rows):
			if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
				imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
			else:
				imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
			if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
		hor= np.hstack(imgArray)
		ver = hor
	return ver
 

def getObjectsHSV(img,imgContour):
 
	contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	areaMinDetect = cv2.getTrackbarPos("MinArea", "Parameters")


	for cnt in contours:
		peri = cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
		x , y , w, h = cv2.boundingRect(approx)
		area = w*h #cv2.contourArea(cnt)
				
		
		if area > areaMinDetect:

			cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 7)
			cv2.rectangle(imgContour, (x , y ), (x + w , y + h ), (0, 255, 0), 5)
 
			cv2.putText(imgContour, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
						(0, 255, 0), 2)
			cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
						(0, 255, 0), 2)
			cv2.putText(imgContour, " " + str(int(x))+ " "+str(int(y)), (x - 20, y- 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
						(0, 255, 0), 2)
			
			detection = det.detection(x,y,w,h,0,(0,0,0),"target",1)
			# cx = int(x+int(w/2))
			# cy = int(y+int(h/2))
			movimentRules(img,detection)


def movimentRules(img,detection):
 
	(x,y,w,h,cx,cy,area) = (detection.x , detection.y , detection.w , detection.h , detection.centerX , detection.centerY , detection.area)
	
	widthPart = int((int(frameWidth/2)-deadZone) /2)
	
	areaMin = cv2.getTrackbarPos("Area Min","Parameters")
	areaMax = cv2.getTrackbarPos("Area Max","Parameters")

	cmd = ""

	if(area < areaMin):
		cv2.putText(imgContour, " GO FOWARD " , (20, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 3)
		cv2.rectangle(imgContour, (x , y ),
				(x + w , y + h ),
				(255, 0, 0), 5)
		cmd = "foward"
	elif(area > areaMax):
		cv2.putText(imgContour, " GO BACKWARD " , (20, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 3)
		cv2.rectangle(imgContour, (x , y ),
				(x + w , y + h ), 
				(0, 0, 255), 5)
		cmd = "backward"

	elif (cx < int(frameWidth/2)-deadZone):
		if (cx < widthPart):
			cv2.putText(imgContour, " GO LEFT " , (20, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 3)
			cv2.rectangle(imgContour,(0,int(frameHeight/2-deadZone)),
					(widthPart, int(frameHeight/2)+deadZone),
					(0,153,255),cv2.FILLED)
			cmd = "left"
		else:
			cv2.putText(imgContour, " GO LEFT ROTATE " , (20, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 3)
			cv2.rectangle(imgContour,(widthPart,int(frameHeight/2-deadZone)),
					(int(frameWidth/2)-deadZone, int(frameHeight/2)+deadZone),
					(0,153,255),cv2.FILLED)
			cmd = "counter_clockwise"
	elif (cx > int(frameWidth/2)+deadZone):
		if (cx < frameWidth-widthPart):
			cv2.putText(imgContour, " GO RIGHT ROTATE ", (20, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 3)
			cv2.rectangle(imgContour,(int(frameWidth/2)+deadZone, int(frameHeight/2-deadZone)),
					(frameWidth-widthPart,int(frameHeight/2)+deadZone),
					(0,153,255),cv2.FILLED)
			cmd = "clockwise"
		else:
			cv2.putText(imgContour, " GO RIGHT ", (20, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 3)
			cv2.rectangle(imgContour,(frameWidth-widthPart, int(frameHeight/2-deadZone)),
					(frameWidth,int(frameHeight/2)+deadZone),
					(0,153,255),cv2.FILLED)
			cmd = "right"
			
	elif (cy < int(frameHeight / 2) - deadZone):
		cv2.putText(imgContour, " GO UP ", (20, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 3)
		cv2.rectangle(imgContour,(int(frameWidth/2-deadZone),0),
				(int(frameWidth/2+deadZone),int(frameHeight/2)-deadZone),
				(0,153,255),cv2.FILLED)
		cmd = "up"
	elif (cy > int(frameHeight / 2) + deadZone):
		cv2.putText(imgContour, " GO DOWN ", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1,(0, 0, 255), 3)
		cv2.rectangle(imgContour,(int(frameWidth/2-deadZone),int(frameHeight/2)+deadZone),
				(int(frameWidth/2+deadZone),frameHeight),
				(0,153,255),cv2.FILLED)
		cmd = "down"


	cv2.line(imgContour, (int(frameWidth/2),int(frameHeight/2)),
			(cx,cy),
			(0, 0, 255), 3)
 
def display(img):
	cv2.line(img,(int(frameWidth/2)-deadZone,0),(int(frameWidth/2)-deadZone,frameHeight),(255,255,0),3)
	cv2.line(img,(int(frameWidth/2)+deadZone,0),(int(frameWidth/2)+deadZone,frameHeight),(255,255,0),3)
	cv2.circle(img,(int(frameWidth/2),int(frameHeight/2)),5,(0,0,255),5)
	cv2.line(img, (0,int(frameHeight / 2) - deadZone), (frameWidth,int(frameHeight / 2) - deadZone), (255, 255, 0), 3)
	cv2.line(img, (0, int(frameHeight / 2) + deadZone), (frameWidth, int(frameHeight / 2) + deadZone), (255, 255, 0), 3)
 
while True:
 
	_, img = cap.read()
	imgContour = img.copy()
	imgHsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
 
	h_min = cv2.getTrackbarPos("HUE Min","HSV")
	h_max = cv2.getTrackbarPos("HUE Max", "HSV")
	s_min = cv2.getTrackbarPos("SAT Min", "HSV")
	s_max = cv2.getTrackbarPos("SAT Max", "HSV")
	v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
	v_max = cv2.getTrackbarPos("VALUE Max", "HSV")
	print(h_min)
 
	lower = np.array([h_min,s_min,v_min])
	upper = np.array([h_max,s_max,v_max])
	mask = cv2.inRange(imgHsv,lower,upper)
	result = cv2.bitwise_and(img,img, mask = mask)
	mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
 
	imgBlur = cv2.GaussianBlur(result, (7, 7), 1)
	imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
	threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
	threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
	imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
	kernel = np.ones((5, 5))
	imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
	getObjectsHSV(imgDil, imgContour)
	display(imgContour)
 
	stack = stackImages(0.7,([img,result],[imgDil,imgContour]))
 
	cv2.imshow('Horizontal Stacking', stack)
	# cv2.imshow("result",imgContour)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
 
cap.release()
cv2.destroyAllWindows()