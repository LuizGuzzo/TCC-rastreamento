import cv2
import numpy as np
import detection as det
from djitellopy import Tello
import colorDetection as cd

def main():

	width = 640  # WIDTH OF THE IMAGE
	height = 480  # HEIGHT OF THE IMAGE
	
	
	
	# CONNECT TO TELLO
	if FLIGHT:
		me = Tello()
		me.connect()
		me.forward_backward_velocity = 0
		me.left_right_velocity = 0
		me.up_down_velocity = 0
		me.yaw_velocity = 0
		me.speed = 0
		
		startCounter =0
		
		battery = 15 #me.get_battery()
		if battery <= 15:
			print("[ERROR] - battery under 15% ")
			SystemExit
		
		me.streamoff()
		me.streamon()
	######################## 
	else:
		cap = cv2.VideoCapture(2)
	
	cd.createHsvTrackers()
	cd.createParamTrackers()
	cd.createMovRulesTrackers()	

	while True:
		# GET THE IMAGE FROM TELLO
		if FLIGHT:
			frame_read = me.get_frame_read()
			myFrame = frame_read.frame
			img = cv2.resize(myFrame, (width, height))
		else:
			_, img = cap.read()

		imgCopy = img.copy()

		detection,stack = cd.getObjectsHSV(imgCopy)
		imgCopy = stack[3]
		
		stackHSV = cd.stackImages(0.7,(stack.copy()))
		

		cmd,imgCopy = cd.movimentRules(imgCopy,detection)
		print("cmd: ",cmd)
		
		################# FLIGHT
		if FLIGHT:
			if startCounter == 0:
				me.takeoff()
				startCounter = 1

			if cmd == "Fwd":
				me.forward_backward_velocity = 20
				me.up_down_velocity = 0; me.yaw_velocity = 0
			elif cmd == "Bwd":
				me.forward_backward_velocity = -20
				me.up_down_velocity = 0; me.yaw_velocity = 0
			elif cmd == "Lft":
				me.yaw_velocity = -70 # me.left_right_velocity = -30
				me.forward_backward_velocity = 0;me.up_down_velocity = 0
			elif cmd == "!cw":
				me.yaw_velocity = -40
				me.forward_backward_velocity = 0;me.up_down_velocity = 0
			elif cmd == "cw":
				me.yaw_velocity = 40
				me.forward_backward_velocity = 0;me.up_down_velocity = 0
			elif cmd == "Rgt":
				me.yaw_velocity = 70 # me.left_right_velocity = 30
				me.forward_backward_velocity = 0;me.up_down_velocity = 0
			elif cmd == "Up":
				me.up_down_velocity= 30
				me.forward_backward_velocity = 0; me.yaw_velocity = 0
			elif cmd == "Dwn":
				me.up_down_velocity= -30
				me.forward_backward_velocity = 0; me.yaw_velocity = 0
			else:
				me.left_right_velocity = 0; me.forward_backward_velocity = 0;me.up_down_velocity = 0; me.yaw_velocity = 0

			# SEND VELOCITY VALUES TO TELLO
		
			if me.send_rc_control:
				me.send_rc_control(me.left_right_velocity, me.forward_backward_velocity, me.up_down_velocity, me.yaw_velocity)
		####################

		cv2.imshow('Horizontal Stacking', stackHSV)
		cv2.imshow("moviment Result",imgCopy)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			if FLIGHT: 
				me.land()
				me.streamoff()
			break

	cv2.destroyAllWindows()

	

if __name__ == '__main__':

	FLIGHT = False
	main()
