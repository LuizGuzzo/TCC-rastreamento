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

PARTICLES = 500
MAXFRAMELOST = 10

# TODO:
# calibrar o FP para a situação da gravação
# modificar a frequencia de atualização para ser em 0.5 sec em vez de frame por frame para o FP
# colocar um input descente
# 
# 
# 
# future:
# acelerar processamento
# pegar o algoritmo da bola do air stick cam e tentar molda-lo aq    


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", default = 'inout/DJI_0127_croped.mp4',	help="path to input video")
ap.add_argument("-o", "--output", default = 'inout/DJI_0127_croped.avi',	help="path to output video")
ap.add_argument("-y", "--yolo", default = 'yolo/yolo-coco',	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())


# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1

filter_is_on = False

yoloCNN = yoo.yoloCNN(args["yolo"], args["confidence"], args["threshold"])

# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	if filter_is_on is False:

		framecpy = frame.copy()
		objects_detected = yoloCNN.get_objects(framecpy)
		for obj in objects_detected:
			obj.draw(framecpy)

		cv2.imshow("objects_detected",framecpy)
		cv2.waitKey(1)
		cv2.destroyAllWindows()

		# centroid = input("digite o centroide desejado")
		centroid_predicted = (1018,553)

		alvo = None
		for obj in objects_detected:
			if obj.check_centroid(centroid_predicted) is True:
				alvo = obj
		
		particleFilter = pf.ParticleFilter(PARTICLES,centroid_predicted,MAXFRAMELOST)
		centroid_predicted = particleFilter.filter_steps(centroid_predicted) # preve antes de ver os obj

		filter_is_on = True


	start = time.time()

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
				alvo = obj
				find = True

		obj.draw(frame)

	if find is False:
		centroid_predicted = None
		centroid_predicted = particleFilter.filter_steps(centroid_predicted)

	particleFilter.drawBox(frame)

	if centroid_predicted is False : # lose tracking (max exceeded)
		filter_is_on = False

	end = time.time()
		

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30, (frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f} | in minutes> {:.2f}".format(elap * total, (elap * total)/60))

	cv2.imshow("result",frame)
	cv2.waitKey(1)
	# cv2.destroyAllWindows()
	# write the output frame to disk
	writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()