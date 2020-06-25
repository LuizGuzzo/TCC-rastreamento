import cv2
import numpy as np
global cmdPrint

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
 

def getObjectsHSV(img,imgCopy):
 
	contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	areaMinDetect = cv2.getTrackbarPos("MinArea", "Parameters")

	detections = []

	for cnt in contours:
		peri = cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
		x , y , w, h = cv2.boundingRect(approx)
		area = w*h #cv2.contourArea(cnt)
				
		
		if area > areaMinDetect:

			cv2.drawContours(imgCopy, cnt, -1, (255, 0, 255), 7)
			cv2.rectangle(imgCopy, (x , y ), (x + w , y + h ), (0, 255, 0), 5)
 
			cv2.putText(imgCopy, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
						(0, 255, 0), 2)
			cv2.putText(imgCopy, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
						(0, 255, 0), 2)
			cv2.putText(imgCopy, " " + str(int(x))+ " "+str(int(y)), (x - 20, y- 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
						(0, 255, 0), 2)
			
			detection = det.detection(x,y,w,h,0,(0,0,0),"target",1)
			
			detections.append(detection)
	
	return detections


def movimentRules(img,detection):
	
	areaMin = cv2.getTrackbarPos("Area Min","Moviment Rules")
	areaMax = cv2.getTrackbarPos("Area Max","Moviment Rules")
	xOffSet = cv2.getTrackbarPos("xOffSet","Moviment Rules")
	yOffSet = cv2.getTrackbarPos("yOffSet","Moviment Rules")

	frameHeight = img.shape[0]
	frameWidth = img.shape[1]

	overlay = img.copy()
	rectangleCoords = None
	color = None

	widthPart = int((int(frameWidth/2)-xOffSet) /2)
	(x,y,w,h,cx,cy,area) = (detection.x , detection.y , detection.w , detection.h , detection.centerX , detection.centerY , detection.area)

	cmd = ""

	if area is not None:
		if(area < areaMin):
			rectangleCoords = [x,y,x+w,y+h]
			color = (255,0,0)
			cmd = "Fwd"
		elif(area > areaMax):
			rectangleCoords = [x,y,x+w,y+h]
			color = (0,0,255)
			cmd = "Bwd"

	if (cx < int(frameWidth/2)-xOffSet):
		if (cx < widthPart):
			rectangleCoords = 	[0,int(frameHeight/2-yOffSet),
								widthPart, int(frameHeight/2)+yOffSet]
			color = (0,153,255)
			cmd = "Lft"
		else:
			rectangleCoords = 	[widthPart,int(frameHeight/2-yOffSet),
								int(frameWidth/2)-xOffSet, int(frameHeight/2)+yOffSet]
			color = (0,153,255)
			cmd = "!cw"
	elif (cx > int(frameWidth/2)+xOffSet):
		if (cx < frameWidth-widthPart):
			rectangleCoords = 	[int(frameWidth/2)+xOffSet, int(frameHeight/2-yOffSet),
								frameWidth-widthPart,int(frameHeight/2)+yOffSet]
			color = (0,153,255)
			cmd = "cw"
		else:
			rectangleCoords = 	[frameWidth-widthPart, int(frameHeight/2-yOffSet),
								frameWidth,int(frameHeight/2)+yOffSet]
			color = (0,153,255)
			cmd = "Rgt"
			
	elif (cy < int(frameHeight / 2) - yOffSet):
		rectangleCoords = 	[int(frameWidth/2-xOffSet),0,
							int(frameWidth/2+xOffSet),int(frameHeight/2)-yOffSet]
		color = (0,153,255)
		cmd = "Up"
	elif (cy > int(frameHeight / 2) + yOffSet):
		rectangleCoords = 	[int(frameWidth/2-xOffSet),int(frameHeight/2)+yOffSet,
							int(frameWidth/2+xOffSet),frameHeight]
		color = (0,153,255)
		cmd = "Dwn"

	if cmdPrint is True:
		cv2.putText(img, cmd , (20, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 3)
	
	if rectangleCoords is not None:
		cv2.rectangle(overlay,
					(rectangleCoords[0],rectangleCoords[1]),
					(rectangleCoords[2],rectangleCoords[3]),
					color,cv2.FILLED)
		alpha = 0.4
		cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0,img)
		

	display(img)
	cv2.line(img, (int(frameWidth/2),int(frameHeight/2)), (cx,cy), (0, 0, 255), 3)

	
	
	return cmd
 
def display(imgCopy):
	frameHeight = imgCopy.shape[0]
	frameWidth = imgCopy.shape[1]
	xOffSet = cv2.getTrackbarPos("xOffSet","Moviment Rules")
	yOffSet = cv2.getTrackbarPos("yOffSet","Moviment Rules")

	cv2.line(imgCopy,(int(frameWidth/2)-xOffSet,0),(int(frameWidth/2)-xOffSet,frameHeight),(255,255,0),3)
	cv2.line(imgCopy,(int(frameWidth/2)+xOffSet,0),(int(frameWidth/2)+xOffSet,frameHeight),(255,255,0),3)
	cv2.circle(imgCopy,(int(frameWidth/2),int(frameHeight/2)),5,(0,0,255),5)
	cv2.line(imgCopy, (0,int(frameHeight / 2) - yOffSet), (frameWidth,int(frameHeight / 2) - yOffSet), (255, 255, 0), 3)
	cv2.line(imgCopy, (0, int(frameHeight / 2) + yOffSet), (frameWidth, int(frameHeight / 2) + yOffSet), (255, 255, 0), 3)

def empty(a):
	pass

def createHsvTrackers():
	cv2.namedWindow("HSV")
	cv2.resizeWindow("HSV",640,240)
	cv2.createTrackbar("HUE Min","HSV",23,179,empty)
	cv2.createTrackbar("HUE Max","HSV",95,255,empty)
	cv2.createTrackbar("SAT Min","HSV",129,255,empty)
	cv2.createTrackbar("SAT Max","HSV",255,255,empty)
	cv2.createTrackbar("VALUE Min","HSV",0,255,empty)
	cv2.createTrackbar("VALUE Max","HSV",255,255,empty)

def createParamTrackers():
	cv2.namedWindow("Parameters")
	cv2.resizeWindow("Parameters",640,240)
	cv2.createTrackbar("Threshold1","Parameters",0,255,empty)
	cv2.createTrackbar("Threshold2","Parameters",94,255,empty)
	cv2.createTrackbar("MinArea","Parameters",848,30000,empty)

def createMovRulesTrackers():
	cv2.namedWindow("Moviment Rules")
	cv2.resizeWindow("Moviment Rules",640,240)
	cv2.createTrackbar("Area Min","Moviment Rules",0,1000000, empty)
	cv2.createTrackbar("Area Max","Moviment Rules",1000000,1000000, empty)
	cv2.createTrackbar("xOffSet","Moviment Rules",100,1080, empty)
	cv2.createTrackbar("yOffSet","Moviment Rules",100,1080, empty)

def main():

	cap = cv2.VideoCapture(0)

	createHsvTrackers()
	createParamTrackers()
	createMovRulesTrackers()	

	while True:
	
		_, img = cap.read()
		imgCopy = img.copy()
		imgHsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	
		h_min = cv2.getTrackbarPos("HUE Min","HSV")
		h_max = cv2.getTrackbarPos("HUE Max", "HSV")
		s_min = cv2.getTrackbarPos("SAT Min", "HSV")
		s_max = cv2.getTrackbarPos("SAT Max", "HSV")
		v_min = cv2.getTrackbarPos("VALUE Min", "HSV")
		v_max = cv2.getTrackbarPos("VALUE Max", "HSV")
	
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

		detections = getObjectsHSV(imgDil, imgCopy)

		for detection in detections:
			cmd = movimentRules(imgCopy,detection)
			print(cmd)
		
		
		# display(imgCopy)
	
		stack = stackImages(0.7,([img,result],[imgDil,imgCopy]))
	
		cv2.imshow('Horizontal Stacking', stack)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	
	cap.release()
	cv2.destroyAllWindows()

	

if __name__ == '__main__':
	import detection as det
	cmdPrint = True
	main()
else:
	import drone.telloHsvTrack.detection as det
	cmdPrint = False