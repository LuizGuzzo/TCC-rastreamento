import numpy as np
import argparse
import imutils
import time
import cv2
import os

import drone.imutility as imt
from drone import telloController as tc
from particle_filter import pf_tools as pf
from yolo import yolOO as yoo
from centroidtracker import CentroidTracker


def getMousePosition(event,x,y,flags,param):
    global mouse,targetAcquired
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouse = (x,y)
	
def show_CNN_FP_info():
	y = 20
	size = len(infos)
	img = 200 * np.ones((size*y+5,200,3), np.uint8)
	cv2.rectangle(img,(0,0),(200,size*y+5),(0,0,0),2)

	for key,value in infos.items():
		cv2.putText(img,
			key+": "+str(value[0]),
			(10, y), cv2.FONT_HERSHEY_PLAIN,1,value[1], 2)
		cv2.line(img,(0,y+5),(200,y+5),(0,0,0),1)
		y += 20

	return img


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", default = 2,	help="path to input video")
ap.add_argument("-o", "--output", default = 'inout/test.avi',	help="path to output video")
ap.add_argument("-y", "--yolo", default = 'yolo/yolo-coco-tiny',	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.1,	help="threshold when applyong non-maxima suppression")
ap.add_argument("-p","--particles", type=float, default=500, help="total of particles on the particle filter")
ap.add_argument("-mf","--maxframelost",type=float, default=30, help="the max of frames can be lost")
ap.add_argument("-dt","--deltat",type=float, default=0.015, help="")
ap.add_argument("-vm","--velmax",type=float, default=4000, help="")
ap.add_argument("-f","--flight",type=int, default=0, help="")

args = vars(ap.parse_args())

FLIGHT = False if args["flight"] == 0 else True
if FLIGHT:
	sTello = tc.simpleTello()
else:
	cap = cv2.VideoCapture(int(args["input"]))

try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(cap.get(prop))
	print("[INFO] {} total frames in video".format(total))
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

writer = None
mouse = None
targetAcquired = False

global infos
infos = {
	"System Status":["",(255,119,0)],
	"FPS": ["0",(255,119,0)],
	"Drone": ["None",(255,119,0)],
	"Track": ["None",(255,119,0)],
	"Class": ["None",(255,119,0)],
}


yoloCNN = yoo.yoloCNN(args["yolo"], args["confidence"], args["threshold"])

while True:
	if FLIGHT:
		frame = sTello.getFrame()
	else:
		(grabbed, frame) = cap.read()
		if not grabbed:		break
	
	framecpy = frame.copy()
	start = time.time()

	objects_array = yoloCNN.get_objects(frame)

	if targetAcquired is False:

		if FLIGHT:
			sTello.takeoffSearching(True)
		
		mouse = None
		alvo = None
		centroid_predicted = None
		cmd = None
		multiTracker = None
		
		for obj in objects_array:
			obj.draw(framecpy)

		cv2.namedWindow('output')
		cv2.setMouseCallback('output',getMousePosition)
		cv2.putText(framecpy, "CHOOSE THE OBJECT", (50, 400), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
		cv2.waitKey(1)

		if mouse is not None:
			
			for obj in objects_array:
				if obj.check_centroid(mouse) is True:
					alvo = obj
					break
			
			if alvo is None:
				print("[ERROR] - chose again the object")
			else:
			
				if FLIGHT:
					sTello.takeoffSearching(False)

				multiTracker = CentroidTracker(args["maxframelost"])
				infos["Class"] = [str(alvo.category),(255,119,0)]

				objectsSameClass = []
				for (i,obj) in enumerate(objects_array):
					if obj.check_category(alvo.category):
						objectsSameClass.append(obj)
						if obj.check_centroid(mouse) is True:
							alvo.id = i
							cv2.destroyAllWindows()
							imt.createMovRulesTrackers(obj.area)
							
				
				multiTracker.update(objectsSameClass)
				
				particleFilter = pf.ParticleFilter(args["particles"],mouse,
									args["maxframelost"],args["deltat"],args["velmax"])
				centroid_predicted = particleFilter.filter_steps(mouse)

				targetAcquired = True
				

	else: #TargetAcquired

		objectsSameClass = []
		for obj in objects_array:
			if (obj.check_category(alvo.category) is True):
				objectsSameClass.append(obj)
			else:
				obj.draw(framecpy) # draw generic objects

		objects_dic = multiTracker.update(objectsSameClass)

		find = False
		for i,obj in objects_dic.items():
			if (multiTracker.disappeared[i] == 0):
				if (alvo.id == i):
					alvo = obj
					find = True
				else:
					obj.draw(framecpy) # draw objects with same class
			else:
				(cx,cy) = obj.get_centroid()
				text1 = "ID: {}".format(i)
				text2 = "{}/{}".format(multiTracker.disappeared[i],args["maxframelost"])
				cv2.putText(framecpy, text1, (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj.color, 2)
				cv2.circle(framecpy, (cx, cy), 2, obj.color, -1)
				cv2.putText(framecpy, text2, (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj.color, 2)

				if(alvo.id == i):
					obj.set_centroid(centroid_predicted)
		

		if find is True:
			centroid_predicted = particleFilter.filter_steps(alvo.get_centroid())
			cmd,framecpy = imt.movimentRules(framecpy,alvo)

			alvo.set_color((0,255,0))
			alvo.draw(framecpy) # draw the target
			infos["Track"] = ["Tracking",(0,255,0)]

		# lose tracking (non max exceeded)
		if find is False: 
			alvo.area = None # avoid unwanted approach
			alvo.centerX , alvo.centerY = (centroid_predicted[0],centroid_predicted[1])

			centroid_predicted = particleFilter.filter_steps(None)
			cmd,framecpy = imt.movimentRules(framecpy,None)
			
			infos["Track"] = ["Predicting",(51,153,255)]

		# lose tracking (max exceeded)
		if centroid_predicted is False:
			print("[ERROR] - chose again the object")
			targetAcquired = False
			infos["Track"] = ["Lost Tracking",(0,0,255)]
			infos["Class"] = ["None",(0,0,255)]
			infos["Drone"] = ["None",(255,119,0)]
			continue
		
		particleFilter.drawBox(framecpy)
		infos["Drone"] = [cmd,(255,119,0)]

		if FLIGHT:
			sTello.setCommand(cmd)

	#end if - TargetAcquired

	end = time.time()
	elap = (end - start)
	fps = round(1/elap,2)

	infos["FPS"] = [str(fps),(255,119,0)]

	images = []

	infoCnnFp = show_CNN_FP_info()
	
	#concat info and output in one image
	final_image = np.zeros((framecpy.shape[0],framecpy.shape[1]+infoCnnFp.shape[1],3),dtype=np.uint8)

	final_image[0:framecpy.shape[0],0:framecpy.shape[1]] = framecpy
	final_image[0:infoCnnFp.shape[0],framecpy.shape[1]:framecpy.shape[1]+infoCnnFp.shape[1]] = infoCnnFp

	if FLIGHT:
		infoDrone = sTello.showDroneInfo()
		final_image[infoCnnFp.shape[0]:infoCnnFp.shape[0]+infoDrone.shape[0],framecpy.shape[1]:framecpy.shape[1]+infoDrone.shape[1]] = infoDrone


	cv2.imshow("output",final_image)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	if writer is None:

		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 6, (final_image.shape[1], final_image.shape[0]), True)
		writer_raw = cv2.VideoWriter("inout/raw.avi", fourcc, 6, (frame.shape[1], frame.shape[0]), True)

		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f} | in minutes> {:.2f}".format(elap * total, (elap * total)/60))
	
	writer.write(final_image)
	writer_raw.write(frame)
	

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
writer_raw.release()
if FLIGHT: 
	print("[INFO] - Drone is Landing")
	sTello.off()
else:
	cap.release()