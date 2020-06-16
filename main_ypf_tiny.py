# USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os

import particle_filter.pf_tools as pf
import yolo.yolOO as yoo
import drone.telloHsvTrack.colorDetection as con

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", default = 'inout/DJI_0127.mp4',	help="path to input video")
ap.add_argument("-o", "--output", default = 'inout/test.avi',	help="path to output video")
ap.add_argument("-y", "--yolo", default = 'yolo/yolo-coco-tiny',	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.3,	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,	help="threshold when applyong non-maxima suppression")
ap.add_argument("-p","--particles", type=float, default=500, help="total of particles on the particle filter")
ap.add_argument("-mf","--maxframelost",type=float, default=30, help="the max of frames can be lost")
ap.add_argument("-dt","--deltat",type=float, default=0.003, help="")
ap.add_argument("-vm","--velmax",type=float, default=4000, help="")

args = vars(ap.parse_args())

def getMousePosition(event,x,y,flags,param):
    global mouse,filterStarted
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouse = (x,y)
        filterStarted = True
	
def display(img):
	y = 20
	cv2.rectangle(img,(0,0),(200,(y*5)+5 ),(255,255,255),cv2.FILLED)
	for info in infos:
		cv2.putText(img, info , (10, y), cv2.FONT_HERSHEY_PLAIN,1,(255,119,0), 2)
		y += 20

global infos

vs = cv2.VideoCapture(0)
# time.sleep(2.0)
writer = None

mouse = None
filterStarted = False
infos = [None]*5
infos[0] = "System Status:"
infos[1] = "FPS: "
infos[2] = "Drone Command: None "
infos[3] = "Track: None "
infos[4] = "Class: None "

yoloCNN = yoo.yoloCNN(args["yolo"], args["confidence"], args["threshold"])

while True:
	(grabbed, frame) = vs.read()
	framecpy = frame.copy()

	start = time.time()

	if not grabbed:
		break

	if filterStarted is False:

		
		objects_detected = yoloCNN.get_objects(frame)
		for obj in objects_detected:
			obj.draw(framecpy)

		cv2.namedWindow('output')
		cv2.setMouseCallback('output',getMousePosition)
		cv2.putText(framecpy, "CHOSE THE OBJECT", (10, 400), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 3)
		# cv2.imshow('output',framecpy)
		cv2.waitKey(1)

		if mouse is not None:
			centroid_predicted = mouse
			
			alvo = None
			for obj in objects_detected:
				if obj.check_centroid(centroid_predicted) is True:
					alvo = obj
					infos[4] = "class: {}".format(alvo.category)
			
			if alvo is None:
				print("[ERROR] - chose again the object")
				mouse = None
				filterStarted = False
				continue

			particleFilter = pf.ParticleFilter(args["particles"],centroid_predicted,
								args["maxframelost"],args["deltat"],args["velmax"])
			centroid_predicted = particleFilter.filter_steps(centroid_predicted)
			mouse = None

			cv2.destroyAllWindows()
			con.createMovRulesTrackers()

	else:

		#CNN
		objects_detected = yoloCNN.get_objects(frame)

		find = False
		for obj in objects_detected:
			obj.draw(framecpy)
			if ((obj.check_centroid(centroid_predicted) is True) and (find is False)):
				if obj.check_category(alvo.category) is True:
					centroid_predicted = particleFilter.filter_steps(obj.get_centroid())
					# obj.set_color((0,255,0))
					alvo = obj
					find = True
		
		particleFilter.drawBox(framecpy)
		cmd = con.movimentRules(framecpy,alvo)
		infos[2] = "Drone Command: {}".format(str(cmd))
		infos[3] = "Track: Tracking"
		

		if (centroid_predicted is False): # lose tracking (max exceeded)
			print("[ERROR] - chose again the object")
			mouse = None
			filterStarted = False
			alvo = None
			cmd = None
			infos[2] = "Drone Command: None "
			infos[3] = "Track: Lost Tracking"
			infos[4] = "class: None"
		
		elif ((find is False)): # lose tracking (non max exceeded)
			centroid_predicted = None
			centroid_predicted = particleFilter.filter_steps(centroid_predicted)
			infos[3] = "Track: Predicting"
			if centroid_predicted is not False:
				alvo.area = None # avoid unwanted approach
				alvo.centerX , alvo.centerY = (centroid_predicted[0],centroid_predicted[1])

		

		

	end = time.time()
	elap = (end - start)
	fps = round(1/elap,2)

	infos[1] = "FPS: {}".format(str(fps))

	display(framecpy)

	# print("FPS: {}".format(str(round(fps, 2))))
	cv2.imshow("output",framecpy)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	if writer is None:

		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, int(fps+1), (frame.shape[1], frame.shape[0]), True)
	
	writer.write(framecpy)

	

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()