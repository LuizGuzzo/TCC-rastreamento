import cv2
import numpy as np
import detection as det

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
 
def getObjectsHSV(img):

	stack = []
	stack.append(img)

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
	stack.append(result)

	imgBlur = cv2.GaussianBlur(result, (7, 7), 1)
	imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
	threshold1 = cv2.getTrackbarPos("Threshold1", "HSV")
	threshold2 = cv2.getTrackbarPos("Threshold2", "HSV")
	imgCanny = cv2.Canny(imgGray, threshold1, threshold2)
	kernel = np.ones((5, 5))
	imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
	stack.append(imgDil)
 
	contours, hierarchy = cv2.findContours(imgDil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	areaMinDetect = cv2.getTrackbarPos("MinArea", "HSV")

	detection = None

	for cnt in contours:
		peri = cv2.arcLength(cnt, True)
		approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
		x , y , w, h = cv2.boundingRect(approx)
		area = w*h #cv2.contourArea(cnt)
				
		
		if area > areaMinDetect:
			areaMinDetect = area
			
			cv2.drawContours(img, cnt, -1, (255, 0, 255), 7)
			cv2.rectangle(img, (x , y ), (x + w , y + h ), (0, 255, 0), 5)
 
			cv2.putText(img, "Points: " + str(len(approx)), (x + w + 20, y + 20), cv2.FONT_HERSHEY_COMPLEX, .7,
						(0, 255, 0), 2)
			cv2.putText(img, "Area: " + str(int(area)), (x + w + 20, y + 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
						(0, 255, 0), 2)
			cv2.putText(img, " " + str(int(x))+ " "+str(int(y)), (x - 20, y- 45), cv2.FONT_HERSHEY_COMPLEX, 0.7,
						(0, 255, 0), 2)
			cv2.circle(img, (int(x+w/2), int(y+h/2)), 3, (0,255,0), -1)
			
			#x,y,w,h,idx,color,category,confidence
			detection = det.detection(x,y,w,h,0,(0,0,0),"target",1)
	

	stack.append(img)

	return detection,stack

def movimentRules(img,detection):
    
    areaMin = cv2.getTrackbarPos("Area Min","Moviment Rules")
    areaMax = cv2.getTrackbarPos("Area Max","Moviment Rules")
    xOffSet = cv2.getTrackbarPos("xOffSet","Moviment Rules")
    yOffSet = cv2.getTrackbarPos("yOffSet","Moviment Rules")

    frameHeight = img.shape[0]
    frameWidth = img.shape[1]
    cw = int(frameWidth/2)
    ch = int(frameHeight/2)

    overlay = img.copy()
    rectangleCoords = None
    color = None
    cmd = ""

    if detection is None:
        img = displayOffSets(img)
        cv2.line(img, (cw-10,ch-10), (cw+10,ch+10), (0, 0, 255), 3)
        cv2.line(img, (cw+10,ch-10), (cw-10,ch+10), (0, 0, 255), 3)
        text = "Move commands are disabled during prediction"
        cv2.putText(img, text , (10,frameHeight-10), cv2.FONT_HERSHEY_PLAIN,1,(0, 0, 255), 1)
        return cmd,img

    widthPart = int((int(frameWidth/2)-xOffSet) /2)
    heightPart = int((int(frameHeight/2)-yOffSet)/2)
    (x,y,w,h,cx,cy,area) = (detection.x , detection.y , detection.w , detection.h , detection.centerX , detection.centerY , detection.area)
    
    color = (0,153,255)
    cmd = "Keep"

    #Prioridade 5
    if(cy < int(frameHeight/2)-yOffSet):
        rectangleCoords = 	[int(frameWidth/2-xOffSet),heightPart,
                                int(frameWidth/2+xOffSet),int(frameHeight/2)-yOffSet]
        cmd = "Up"
    if(cy > int(frameHeight/2)+yOffSet):
        rectangleCoords = 	[int(frameWidth/2-xOffSet),int(frameHeight/2)+yOffSet,
                                int(frameWidth/2+xOffSet),frameHeight - heightPart]
        cmd = "Dwn"

    #Prioridade 4
    if(cx < int(frameWidth/2)-xOffSet):
        rectangleCoords = 	[widthPart,heightPart,
                                int(frameWidth/2)-xOffSet, frameHeight-heightPart]
        cmd = "!cw"
    if(cx > int(frameWidth/2)+xOffSet):
        rectangleCoords = 	[int(frameWidth/2)+xOffSet, heightPart,
                                frameWidth-widthPart,frameHeight-heightPart]
        cmd = "cw"

    #Prioridade 3
    if (area is not None):
        if(area < areaMin):
            rectangleCoords = [x,y,x+w,y+h]
            color = (255,0,0)
            cmd = "Fwd"
        elif(area > areaMax):
            rectangleCoords = [x,y,x+w,y+h]
            color = (0,0,255)
            cmd = "Bwd"

    #Prioridade 2
    if(cy < heightPart):
        rectangleCoords = 	[widthPart,0,
                                frameWidth-widthPart,heightPart]
        color = (0,153,255)
        cmd = "Up+"
    if(cy > frameHeight-heightPart):
        rectangleCoords = 	[widthPart,frameHeight - heightPart,
                                frameWidth-widthPart,frameHeight]
        color = (0,153,255)
        cmd = "Dwn+"

    #Prioridade 1
    if(cx < widthPart):
        rectangleCoords = 	[0,0,
                                widthPart, frameHeight]
        color = (0,153,255)
        cmd = "Lft"
    if(cx > frameWidth-widthPart):
        rectangleCoords = 	[frameWidth-widthPart,0,
                                frameWidth,frameHeight]
        color = (0,153,255)
        cmd = "Rgt"
    
    if rectangleCoords is not None:
        cv2.rectangle(overlay,
                    (rectangleCoords[0],rectangleCoords[1]),
                    (rectangleCoords[2],rectangleCoords[3]),
                    color,cv2.FILLED)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0,img)
        
    img = displayOffSets(img)
    cv2.line(img, (int(frameWidth/2),int(frameHeight/2)), (cx,cy), (0, 0, 255), 3)

    return cmd,img

def displayOffSets(img):

    frameHeight = img.shape[0]
    frameWidth = img.shape[1]
    xOffSet = cv2.getTrackbarPos("xOffSet","Moviment Rules")
    yOffSet = cv2.getTrackbarPos("yOffSet","Moviment Rules")

    cv2.line(img,(int(frameWidth/2)-xOffSet,0),(int(frameWidth/2)-xOffSet,frameHeight),(255,255,0),3)
    cv2.line(img,(int(frameWidth/2)+xOffSet,0),(int(frameWidth/2)+xOffSet,frameHeight),(255,255,0),3)
    cv2.circle(img,(int(frameWidth/2),int(frameHeight/2)),2,(0,0,255),2)
    cv2.line(img, (0,int(frameHeight / 2) - yOffSet), (frameWidth,int(frameHeight / 2) - yOffSet), (255, 255, 0), 3)
    cv2.line(img, (0, int(frameHeight / 2) + yOffSet), (frameWidth, int(frameHeight / 2) + yOffSet), (255, 255, 0), 3)

    return img

def empty():
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
	cv2.createTrackbar("Threshold1","HSV",0,255,empty)
	cv2.createTrackbar("Threshold2","HSV",94,255,empty)
	cv2.createTrackbar("MinArea","HSV",1000,30000,empty)

def createMovRulesTrackers():
	cv2.namedWindow("Moviment Rules")
	cv2.resizeWindow("Moviment Rules",640,240)
	cv2.createTrackbar("Area Min","Moviment Rules",4500,1000000, empty)
	cv2.createTrackbar("Area Max","Moviment Rules",7500,1000000, empty)
	cv2.createTrackbar("xOffSet","Moviment Rules",70,1080, empty)
	cv2.createTrackbar("yOffSet","Moviment Rules",70,1080, empty)
