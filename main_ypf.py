# USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os

from djitellopy import Tello
import particle_filter.pf_tools as pf
import yolo.yolOO as yoo
import drone.telloHsvTrack.colorDetection as con
from centroidtracker import CentroidTracker

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", default = 0,	help="path to input video")
ap.add_argument("-o", "--output", default = 'inout/test.avi',	help="path to output video")
ap.add_argument("-y", "--yolo", default = 'yolo/yolo-coco-tiny',	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.4,	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.1,	help="threshold when applyong non-maxima suppression")
ap.add_argument("-p","--particles", type=float, default=500, help="total of particles on the particle filter")
ap.add_argument("-mf","--maxframelost",type=float, default=30, help="the max of frames can be lost")
ap.add_argument("-dt","--deltat",type=float, default=0.015, help="")
ap.add_argument("-vm","--velmax",type=float, default=4000, help="")

args = vars(ap.parse_args())

def getMousePosition(event,x,y,flags,param):
    global mouse,filterStarted
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouse = (x,y)
	
def display(img):
	y = 20
	size = len(infos)
	cv2.rectangle(img,(0,0),(200,size*y+5 ),(255,255,255),cv2.FILLED)
	for key,value in infos.items():
		cv2.putText(img,
			key+": "+str(value),
			(10, y), cv2.FONT_HERSHEY_PLAIN,1,(255,119,0), 2)
		y += 20

global infos

vs = cv2.VideoCapture(args["input"])

try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

writer = None
mouse = None
filterStarted = False
infos = {
	"System Status":"",
	"FPS": 0,
	"Drone": None,
	"Track": None,
	"Class": None,
}

# try:
# 	me = Tello()
# 	me.connect()
# 	me.for_back_velocity = 0
# 	me.left_right_velocity = 0
# 	me.up_down_velocity = 0
# 	me.yaw_velocity = 0
# 	me.speed = 0
# 	print(me.get_battery())

# 	me.streamoff()
# 	me.streamon()
# 	telloAvaible = True
# except:
# 	telloAvaible = False
	
# print(telloAvaible)


yoloCNN = yoo.yoloCNN(args["yolo"], args["confidence"], args["threshold"])

while True:
	(grabbed, frame) = vs.read()
	framecpy = frame.copy()


	start = time.time()

	if not grabbed:
		break

	if filterStarted is False:
		
		mouse = None
		alvo = None
		centroid_predicted = None
		cmd = None
		ct = None
		
		objects_array = yoloCNN.get_objects(frame)
		for obj in objects_array:
			obj.draw(framecpy)

		cv2.namedWindow('output')
		cv2.setMouseCallback('output',getMousePosition)
		cv2.putText(framecpy, "CHOSE THE OBJECT", (10, 400), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 3)
		cv2.waitKey(1)

		if mouse is not None:

			centroid_predicted = mouse
			
			for obj in objects_array:
				if obj.check_centroid(centroid_predicted) is True:
					alvo = obj
					break
			
			if alvo is None:
				print("[ERROR] - chose again the object")
				continue
			else:
				
				ct = CentroidTracker(args["maxframelost"])
				infos["Class"] = alvo.category

				objectsSameClass = []
				for (i,obj) in enumerate(objects_array):
					if obj.check_category(alvo.category):
						objectsSameClass.append(obj)
						if obj.check_centroid(centroid_predicted) is True:
							alvo.id = i
				
				ct.update(objectsSameClass)
				
				particleFilter = pf.ParticleFilter(args["particles"],centroid_predicted,
									args["maxframelost"],args["deltat"],args["velmax"])
				centroid_predicted = particleFilter.filter_steps(centroid_predicted)

				filterStarted = True
				cv2.destroyAllWindows()
				con.createMovRulesTrackers()

	else:

		#CNN
		objects_array = yoloCNN.get_objects(frame)
		objectsSameClass = []
		for obj in objects_array:
			if (obj.check_category(alvo.category) is True):
				objectsSameClass.append(obj)
			else:
				obj.draw(framecpy) # draw generic objects

			
		#MultiTrack
		objects_dic = ct.update(objectsSameClass)

		find = False
		for i,obj in objects_dic.items():
			if (ct.disappeared[i] == 0):
				if (alvo.id == i):
					alvo = obj
					find = True
				else:
					obj.draw(framecpy) # draw objects with same class
			else:
				(cx,cy) = obj.get_centroid()
				text1 = "ID: {}".format(i)
				text2 = "{}/{}".format(ct.disappeared[i],args["maxframelost"])
				cv2.putText(framecpy, text1, (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj.color, 2)
				cv2.circle(framecpy, (cx, cy), 2, obj.color, -1)
				cv2.putText(framecpy, text2, (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj.color, 2)

				if(alvo.id == i):
					obj.set_prediction(centroid_predicted)
		

		if find is True:
			centroid_predicted = particleFilter.filter_steps(alvo.get_centroid())
			cmd = con.movimentRules(framecpy,alvo)

			alvo.set_color((0,255,0))
			alvo.draw(framecpy) # draw the target
			infos["Track"] = "Tracking"		

		# lose tracking (non max exceeded)
		if find is False: 
			alvo.area = None # avoid unwanted approach
			alvo.centerX , alvo.centerY = (centroid_predicted[0],centroid_predicted[1])

			centroid_predicted = particleFilter.filter_steps(None)
			cmd = con.movimentRules(framecpy,None)
			
			infos["Track"] = "Predicting"

		# lose tracking (max exceeded)
		if centroid_predicted is False:
			print("[ERROR] - chose again the object")
			filterStarted = False
			infos["Track"] = "Lost Tracking"
			infos["Class"] = None
			infos["Drone"] = None
			continue
		
		particleFilter.drawBox(framecpy)
		infos["Drone"] = cmd

		# #controller

		# if cmd == "Fwd":

		# elif cmd == "Bwd":
			
		# elif cmd == "Lft":
			
		# elif cmd == "!cw":
			
		# elif cmd == "cw":

		# elif cmd == "Rgt":

		# elif cmd == "Up":

		# elif cmd == "Dwn":

		# else:


			
		

	end = time.time()
	elap = (end - start)
	fps = round(1/elap,2)

	infos["FPS"] = fps

	display(framecpy)

	# print("FPS: {}".format(str(round(fps, 2))))
	cv2.imshow("output",framecpy)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	if writer is None:

		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, int(fps+1), (frame.shape[1], frame.shape[0]), True)

		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f} | in minutes> {:.2f}".format(elap * total, (elap * total)/60))
	
	writer.write(framecpy)

	

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()