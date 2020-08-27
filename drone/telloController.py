import cv2
import numpy as np
from djitellopy import Tello
import sys
global cmdPrint

class simpleTello():
	def __init__(self):
		self.width = 640
		self.height = 480

		me = Tello()
		me.connect()
		me.forward_backward_velocity = 0
		me.left_right_velocity = 0
		me.up_down_velocity = 0
		me.yaw_velocity = 0
		me.speed = 0
	
		battery = me.get_battery()
		if battery <= 15:
			print("[ERROR] - battery under 15% ")
			sys.exit(0)
		
		me.streamoff()
		me.streamon()

		self.me = me
		self.takeoff = False
		self.frame = None
		

	def getFrame(self):
		frame_read = self.me.get_frame_read()
		myFrame = frame_read.frame
		self.frame = cv2.resize(myFrame, (self.width, self.height))
		return self.frame

	def takeoffSearching(self,flag):
		me = self.me

		if self.takeoff == False:
			me.takeoff()
			self.takeoff = True
		
		me.left_right_velocity = 0; me.forward_backward_velocity = 0;me.up_down_velocity = 0; me.yaw_velocity = 0

		if flag:
			me.yaw_velocity = 10

		if me.send_rc_control:
			me.send_rc_control(me.left_right_velocity, me.forward_backward_velocity, me.up_down_velocity, me.yaw_velocity)

	def setCommand(self,cmd):
		me = self.me

		me.left_right_velocity = 0; me.forward_backward_velocity = 0;me.up_down_velocity = 0; me.yaw_velocity = 0

		if cmd == "Fwd":
			me.forward_backward_velocity = 20
		if cmd == "Bwd":
			me.forward_backward_velocity = -20
		if cmd == "Lft":
			me.yaw_velocity = -60 # me.left_right_velocity = -30
		if cmd == "!cw":
			me.yaw_velocity = -30
		if cmd == "cw":
			me.yaw_velocity = 30
		if cmd == "Rgt":
			me.yaw_velocity = 60 # me.left_right_velocity = 30
		if cmd == "Up":
			me.up_down_velocity= 30
		if cmd == "Dwn":
			me.up_down_velocity= -30
		if cmd == "Up+":
			me.up_down_velocity= 50
		if cmd == "Dwn+":
			me.up_down_velocity= -50

		# SEND VELOCITY VALUES TO TELLO
	
		if me.send_rc_control:
			me.send_rc_control(me.left_right_velocity, me.forward_backward_velocity, me.up_down_velocity, me.yaw_velocity)
	
	def showDroneInfo(self): 

		drone = self.me

		battery = drone.get_battery()
		temperature = drone.get_temperature()
		ToF = drone.get_distance_tof()
		wifiSignal = 0 #drone.query_wifi_signal_noise_ratio()
		flightTime = drone.get_flight_time()

		texts = []
		color = (255,119,0)
		if(int(battery) <= 50):
			color = (51,153,255)
		if(int(battery) <= 25):
			color = (0,0,255)
		texts.append(["Battery: {}%".format(str(battery)),color])

		color = (255,119,0)
		if(temperature >= 80):
			color = (51,153,255)
		texts.append(["Temperature: {} C".format(str(temperature)),color])

		color = (255,119,0)
		if(ToF >= 200):
			color = (51,153,255)
		texts.append(["ToF: {}".format(str(ToF)),color])

		# color = (255,119,0)
		# if(int(wifiSignal) <= 60):
		# 	color = (51,153,255)
		# texts.append(["Wifi-Signal: {} [Disable]".format(str(wifiSignal)),color])

		color = (255,119,0)
		if(flightTime >= 120):
			color = (51,153,255)
		texts.append(["Flight Time: {}".format(str(flightTime)),color])
		
		height = 20
		size = len(texts)
		img = 200 * np.ones((size*height+5,200,3), np.uint8)
		cv2.rectangle(img,(0,0),(200,size*height+5),(0,0,0),2)

		for line in texts:
			cv2.putText(img, line[0] , (10,height), cv2.FONT_HERSHEY_PLAIN,1,line[1], 2)
			cv2.line(img,(0,height+5),(200,height+5),(0,0,0),1)
			height += 20
		
		return img

	def off(self):
		self.me.land()
		self.me.streamoff()

def main():
	imu.createHsvTrackers()
	imu.createMovRulesTrackers()	
	
	
	if FLIGHT:
		sTello = simpleTello()

	else:
		cap = cv2.VideoCapture(2)
	
	

	while True:
		if FLIGHT:
			img = sTello.getFrame()
		else:
			_, img = cap.read()

		imgCopy = img.copy()

		#GET OBJECTS
		detection,stack = imu.getObjectsHSV(imgCopy)
		imgCopy = stack[3]
		
		stackHSV = imu.stackImages(0.7,(stack.copy()))
		
		#GET CMD
		cmd,imgCopy = imu.movimentRules(imgCopy,detection)
		cv2.putText(img, cmd , (20, 50), cv2.FONT_HERSHEY_COMPLEX,1,(0, 0, 255), 3)
		print("cmd: ",cmd)
		
		################# FLIGHT
		if FLIGHT:
			sTello.setCommand(cmd)
		####################

		cv2.imshow('Horizontal Stacking', stackHSV)
		cv2.imshow("moviment Result",imgCopy)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			if FLIGHT: 
				sTello.off()
			else:
				cap.release()
			break

	cv2.destroyAllWindows()
	

if __name__ == '__main__':
	import detection as det
	import argparse
	import imutility as imu

	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--flight", default = '0')
	args = vars(ap.parse_args())

	if args["flight"] == '0':
		FLIGHT = False
	else:
		FLIGHT = True


	main()
else:
	from . import detection as det
	# import drone.telloHsvTrack.detection as det

	

	
	