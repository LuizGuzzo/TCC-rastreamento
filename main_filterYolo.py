# USAGE
# python yolo_video.py --input videos/airport.mp4 --output output/airport_output.avi --yolo yolo-coco

# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os

import alvo as al
import particle_filter.pf_tools as pf

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", default = 'videos/DJI_0127.mp4',	help="path to input video")
ap.add_argument("-o", "--output", default = 'output/DJI_0127.avi',	help="path to output video")
ap.add_argument("-y", "--yolo", default = 'yolo-coco',	help="base path to YOLO directory")
ap.add_argument("-c", "--confidence", type=float, default=0.5,	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())


# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

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

tracked = False
lost = 0
alvo = al.alvo()
find = False
vet_P_None = None

# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break

	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
	net.setInput(blob)

	start = time.time()

	layerOutputs = net.forward(ln)
	
	

	######################################################################

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"]:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height), int(centerX), int(centerY)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],	args["threshold"])


	######################################################


	# recebe a imagem
	# recebe objetos identificados
	# printa eles na tela
	# seleciona um
	# inicializa o vet_P
	# proximo frame

	# recebe a imagem
	# recebe objetos identificados
	# preve a movimentação do alvo selecionado anteriormente



	#inicializa o vet_P

	# preve a movimentação do alvo - mantem o vet P anterior
	# recebe os objetos
	# compara se essa posição tem um objeto da mesma classe do alvo
		# se sim, passa o vet P anterior para atualizar o vet P conforme o novo centroide
		# se não mantem a predição como novo vet P

	# atualiza as coordenadas do alvo
	# desenha vet P e a sua media central
	
	
	


	# ensure at least one detection exists
	if len(idxs) > 0:
		

		# loop over the indexes we are keeping
		for i in idxs.flatten():
			# extract the bounding box coordinates
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			(centerX,centerY)= (boxes[i][4], boxes[i][5]) # center of the box
			color = (0,0,255)

			if alvo.se_alvo_na_area(x,y,w,h,LABELS[classIDs[i]]):
				color = [int(c) for c in COLORS[classIDs[i]]]
				vet_P , alvo.vet_PP = pf.filter_steps(vet_P,(x,y))
				
				find = True
			


			# draw a bounding box rectangle and label on the frame
			cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
			text = "i:{} | {}: {:.4f}".format(i,LABELS[classIDs[i]], confidences[i])
			cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			
			# draw in the center of the box and label the coordinates below the box
			cv2.circle(frame,(centerX,centerY),2,color)
			centertext1 = "x: {}|y: {}".format(x,y)
			centertext2 = "w: {}|h: {}".format(w,h)
			centertext3 = "center: {}|centerY: {}".format(centerX,centerY)

			cv2.putText(frame,centertext1,(x, y + h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			cv2.putText(frame,centertext2,(x, y + h+40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			cv2.putText(frame,centertext3,(x, y + h+60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
		
		if find is False:
			lost = lost +1
			vet_P = vet_P_None

		if lost > 4:
			lost = 0
			tracked = False

		find = False

		if tracked is False:
			#re-escolha o centro
			center = (969,544) # poem um input
			vet_P = pf.start(center)
			vet_P, vet_PP = pf.filter_steps(vet_P,center) # HERE

			alvo = al.alvo()
			alvo.setAll(LABELS[0],vet_PP)
			tracked = True
		else:
			vet_P_None = pf.filter_steps(vet_P,None) # e se ele prever errado? preciso cv com max

	#end if not exist detection			
			
	alvo.draw_particles(frame)
				
	end = time.time()
			

			

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))

	cv2.imshow("Image", frame)
	cv2.waitKey(0)
	# write the output frame to disk
	writer.write(frame)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
vs.release()