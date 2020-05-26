import cv2
# import particle_filter.particle as particle
import particle_filter.pf_tools as pf

class alvo():

	def __init__(self):
		self.X = -1
		self.Y = -1
		self.classe = -1
		self.vet_PP = -1
		

	def setAll(self,classe,vet_PP):
		self.classe = classe
		self.vet_PP = vet_PP
		(x,y) = pf.calc_avg_particles(self.vet_PP)
		self.X = x
		self.Y = y
	
	def se_alvo_na_area(self,x,y,w,h,classe):
		if (x <= self.X <= x+w) and (y <= self.Y <= y+h):
			if classe == self.classe:
				return True
		return False

	def print(self):
		print("x:{} | y:{} | classe: {}".format(self.X,self.Y,self.classe))

	def draw_particles(self,frame):
		return pf.drawBox(self.vet_PP,frame)

	def draw(self,frame):
		roxo = (153,51,153)
		cv2.circle(frame,(int(self.X),int(self.Y)),100,roxo,2)
		cv2.circle(frame,(int(self.X),int(self.Y)),2,roxo,-1)
		return frame
