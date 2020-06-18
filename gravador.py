# USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", default = 'inout/result.avi',	help="path to input video")
ap.add_argument("-o", "--output", default = 'inout/test.avi',	help="path to output video")
ap.add_argument("-f", "--fps", default = 10,	help="Frames needed per sec")

args = vars(ap.parse_args())

global infos

vs = cv2.VideoCapture(args["input"])
writer = None


while True:
	(grabbed, frame) = vs.read()

	cv2.imshow("output",frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

	if writer is None:

		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, args["fps"], (frame.shape[1], frame.shape[0]), True)


	writer.write(frame)

	

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()