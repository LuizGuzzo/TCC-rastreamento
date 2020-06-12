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

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--input", default = 'inout/DJI_0127.mp4',	help="path to input video")
ap.add_argument("-o", "--output", default = 'inout/test.avi',	help="path to output video")
ap.add_argument("-y", "--yolo", default = 'yolo/yolo-coco-tiny',	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,	help="threshold when applyong non-maxima suppression")
ap.add_argument("-p","--particles", type=float, default=500, help="total of particles on the particle filter")
ap.add_argument("-mf","--maxframelost",type=float, default=60, help="the max of frames can be lost")
ap.add_argument("-dt","--deltat",type=float, default=0.003, help="")
ap.add_argument("-vm","--velmax",type=float, default=4000, help="")

args = vars(ap.parse_args())


vs = cv2.VideoCapture(0)
# time.sleep(2.0)
writer = None

mouse = None
filterStarted = False


def getMousePosition(event,x,y,flags,param):
    global mouse,filterStarted
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouse = (x,y)
        filterStarted = True
	


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
		cv2.imshow('output',framecpy)
		cv2.waitKey(1)

		if mouse is not None:
			centroid_predicted = mouse

			alvo = None
			for obj in objects_detected:
				if obj.check_centroid(centroid_predicted) is True:
					alvo = obj
			
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

	else:

		#CNN
		objects_detected = yoloCNN.get_objects(frame)

		find = False
		for obj in objects_detected:

			if obj.check_centroid(centroid_predicted) is True:
				# print("centroid confirmed")
				if obj.check_category(alvo.category) is True:
					# print("category confirmed")
					centroid_predicted = particleFilter.filter_steps(obj.get_centroid())
					obj.set_color((0,255,0)) # green
					alvo = obj # alvo poderia ser apenas a classe do objeto (e oque garante hipoteticamente que seja o mesmo objeto)
					find = True

			obj.draw(framecpy)

		if find is False: # lose tracking (non max exceeded)
			centroid_predicted = None
			centroid_predicted = particleFilter.filter_steps(centroid_predicted)

		particleFilter.drawBox(framecpy)

		if centroid_predicted is False : # lose tracking (max exceeded)
			print("[ERROR] - chose again the object")
			mouse = None
			filterStarted = False
			alvo = None

	if writer is None:
		
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 5, (frame.shape[1], frame.shape[0]), True)

		# 	# some information on processing single frame
		# 	if total > 0:
		# 		elap = (end - start)
		# 		print("[INFO] single frame took {:.4f} seconds".format(elap))
		# 		# print("[INFO] estimated total time to finish: {:.4f} | in minutes> {:.2f}".format(elap * total, (elap * total)/60))

	end = time.time()
	elap = (end - start)
	fps = 1/elap
	cv2.putText(framecpy, "FPS: {}".format(str(round(fps, 2))), (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 3)
	print("FPS: {}".format(str(round(fps, 2))))
	cv2.imshow("output",framecpy)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	
	writer.write(framecpy)

	

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()