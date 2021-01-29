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

def bb_intersection_over_union(boxA, boxB):
		# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
	# return the intersection over union value
	return iou*100


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
ap.add_argument("-iou","--iou",type=int, default=50, help="")

args = vars(ap.parse_args())

FLIGHT = False if args["flight"] == 0 else True
if FLIGHT:
	sTello = tc.simpleTello()
else:
	# cap = cv2.VideoCapture(int(args["input"]))
	cap = cv2.VideoCapture("inout/trims/trim_5.avi")

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

VP = 0 #Verdadeiro Positivo, quando Iou >= 50
FP = 0 #Falso Positivo, quando Iou < 50
FN = 0 #Falso Negativo, quando Yolo falha
VN = 0 #Verdadeiro Negativo, nao aplicado
falhas = -1
trimCount = 0

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
				falhas +=1


	else: #TargetAcquired

		objectsSameClass = []
		for obj in objects_array:
			if (obj.check_category(alvo.category) is True):
				objectsSameClass.append(obj)
			else:
				obj.draw(framecpy) # draw generic objects

		oldObjects_dic = multiTracker.getList()
		multiTracker.update(objectsSameClass)
		objects_dic = multiTracker.getList()

		find = False
		for id,obj in objects_dic.items():
			if (multiTracker.disappeared[id] == 0):
				
				if (alvo.id == id):
					alvo = obj
					find = True

				else:
					obj.draw(framecpy) # draw objects with same class
			else:
				
				(cx,cy) = obj.get_centroid()
				text1 = "ID: {}".format(id)
				text2 = "{}/{}".format(multiTracker.disappeared[id],args["maxframelost"])
				cv2.putText(framecpy, text1, (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj.color, 2)
				cv2.circle(framecpy, (cx, cy), 2, obj.color, -1)
				cv2.putText(framecpy, text2, (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj.color, 2)

				if(alvo.id == id):
					obj.set_centroid(centroid_predicted)
		
		oldSTD = particleFilter.calcDesvioPadrao()

		if find is True:

			centroid_predicted = particleFilter.filter_steps(alvo.get_centroid())
			cmd,framecpy = imt.movimentRules(framecpy,alvo)

			alvo.set_color((0,255,0))
			alvo.draw(framecpy) # draw the target
			infos["Track"] = ["Tracking",(0,255,0)]

			if alvo.id in oldObjects_dic.keys() and alvo.id in objects_dic.keys():
				#calculate IoU for the target
				inputBox = [objects_dic[alvo.id].x , objects_dic[alvo.id].y , objects_dic[alvo.id].x + objects_dic[alvo.id].w, objects_dic[alvo.id].y + objects_dic[alvo.id].h]
				oldBoxOverFilterParticles = [centroid_predicted[0] - int(oldObjects_dic[alvo.id].w / 2) , centroid_predicted[1] - int(oldObjects_dic[alvo.id].h / 2),
					centroid_predicted[0] + int(oldObjects_dic[alvo.id].w / 2), centroid_predicted[1] + int(oldObjects_dic[alvo.id].h / 2)]

				cv2.rectangle(framecpy, 
					(oldBoxOverFilterParticles[0],oldBoxOverFilterParticles[1]),
					(oldBoxOverFilterParticles[2],oldBoxOverFilterParticles[3]),
					(255, 0, 0), 1)
				
				iou = bb_intersection_over_union(inputBox,oldBoxOverFilterParticles)
				text = "IoU:{:.0f}%".format(iou)
				cv2.putText(framecpy, text, (inputBox[0] + 5, inputBox[1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

				if iou > args["iou"]:
					VP = VP + 1
				else:
					FP = FP + 1
				
				####################

			

		# lose tracking (non max exceeded)
		if find is False:
			FN += 1 # alvo nao identificado
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
		
		newSTD = particleFilter.calcDesvioPadrao()
		expansion = newSTD/oldSTD
		# print("oldSTD: {:.2f} | newSTD: {:.2f} | expansion: {:.2f}".format(oldSTD,newSTD,expansion))

		particleFilter.drawBox(framecpy)
		infos["Drone"] = [cmd,(255,119,0)]

		if FLIGHT:
			sTello.setCommand(cmd)

	#end if - TargetAcquired

	end = time.time()
	elap = (end - start)
	fps = round(1/elap,2)

	infos["FPS"] = [str(fps),(255,119,0)]

	infoCnnFp = show_CNN_FP_info()
	
	#concat info and output in one image
	final_image = np.zeros((framecpy.shape[0],framecpy.shape[1]+infoCnnFp.shape[1],3),dtype=np.uint8)

	final_image[0:framecpy.shape[0],0:framecpy.shape[1]] = framecpy
	final_image[0:infoCnnFp.shape[0],framecpy.shape[1]:framecpy.shape[1]+infoCnnFp.shape[1]] = infoCnnFp

	if FLIGHT:
		infoDrone = sTello.showDroneInfo()
		final_image[infoCnnFp.shape[0]:infoCnnFp.shape[0]+infoDrone.shape[0],framecpy.shape[1]:framecpy.shape[1]+infoDrone.shape[1]] = infoDrone


	cv2.imshow("output",final_image)

	if cv2.waitKey(1) & 0xFF == ord('c'):
		#preciso cortar esse video e n acho um bom editor, logo vou fazer o meu..
		writer_trim.release()
		trimCount+=1
		writer_trim = cv2.VideoWriter("inout/trim_{}.avi".format(trimCount), fourcc, 6, (frame.shape[1], frame.shape[0]), True)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
	

	if writer is None:

		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 6, (final_image.shape[1], final_image.shape[0]), True)
		writer_raw = cv2.VideoWriter("inout/raw.avi", fourcc, 6, (frame.shape[1], frame.shape[0]), True)
		writer_trim = cv2.VideoWriter("inout/trim_{}.avi".format(trimCount), fourcc, 6, (frame.shape[1], frame.shape[0]), True)
		

		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f} | in minutes> {:.2f}".format(elap * total, (elap * total)/60))
	
	writer.write(final_image)
	writer_raw.write(frame)
	writer_trim.write(frame)



if total > 0:
	rev = VP/(VP+FN)
	precisao = VP/(VP+FP)
	acc = (VP+VN)/(VP+VN+FP+FN)
print("VP: {} FP: {} FN: {}".format(VP,FP,FN))
print("rev:{:.2f}% | precisao: {:.2f}% | acuracia: {:.2f}".format(rev*100,precisao*100,acc*100))
print("Falhou:",falhas)

# release the file pointers
print("[INFO] cleaning up...")
writer.release()
writer_raw.release()
writer_trim.release()
if FLIGHT: 
	print("[INFO] - Drone is Landing")
	sTello.off()
else:
	cap.release()