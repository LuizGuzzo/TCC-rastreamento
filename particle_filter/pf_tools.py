import imutils
import cv2
import math
import time
import random
import numpy as np

# import particle as p
import particle_filter.particle as p # para rodar pela main ypf

from pointpats.centrography import std_distance



class ParticleFilter():

    def __init__(self,MAXPARTICLES,center,MAXFRAMELOST, DELTA_T = None ,VELMAX = None):
        self.MAXPARTICLES = MAXPARTICLES
        self.MAXFRAMELOST = MAXFRAMELOST
        self.__vet_particles = self.__start(center,DELTA_T,VELMAX)
        self.vet_particles_predicted = None
        self.__countToMaxFrameLost = 0

    def __start(self,center,DELTA_T,VELMAX):
        vet_particles = []
        for _ in range(self.MAXPARTICLES):
            particle = p.particle(DELTA_T,VELMAX)
            particle.start(center)
            vet_particles.append(particle)

        return vet_particles

    def __prediction(self):
        for particle in self.__vet_particles:
            particle = particle.prediction()
        
        return self.__vet_particles

    def __correction(self,center):
        for particle in self.__vet_particles:
            particle = particle.correction(center)

    def __normalize(self):
        sumvet = 0
        for particle in self.__vet_particles:
            sumvet += particle.toNormalize
        
        for particle in self.__vet_particles:
            particle = particle.normaliza(sumvet)

    def __getParticle_byWeight(self,ref):
        # roda no vet de particulas e quando o peso passa do adquirido atribui essa particula ao resultado
        acumulator = 0
        for i,particle in enumerate(self.__vet_particles,0):
            acumulator += particle.weight
            if acumulator >= ref:
                return particle
    
    def __resort(self):
        sorted_vet_particulas = [p.particle() for _ in range(self.MAXPARTICLES)]
                
        frag = 1/self.MAXPARTICLES
        reference = random.uniform(0,1)
        # print(reference)
        for i in range(self.MAXPARTICLES):

            if reference > 1:
                reference -= 1
            
            new_particle = self.__getParticle_byWeight( reference )
            sorted_vet_particulas[i].setAll(new_particle) # pega a particula 'gorda'

            reference += frag

        self.__vet_particles = sorted_vet_particulas
        return sorted_vet_particulas

    
    def drawBox(self,frame):
        # frame = cv2.copyMakeBorder(frame,50,50,50,50,cv2.BORDER_CONSTANT,value= (255,255,255))
        cor = (255,0,255)
        

        sumX = 0
        sumY = 0

        for particle in self.vet_particles_predicted:
            frame = cv2.circle(frame, (int(particle.X), int(particle.Y)),2, (242,147,244), 1) #desenha as particulas
            # calculating the direction of particle and drawing it
            a = particle.Vx
            b = particle.Vy
            c = math.sqrt(math.pow(a,2)+math.pow(b,2))
            beta = math.asin(b/c)
            x = 3 * math.cos(beta)
            y = 3 * math.sin(beta)

            frame = cv2.line(frame, (int(particle.X), int(particle.Y)),
                    (int(particle.X+x), int(particle.Y+y)), (0,0,255),1 )
            sumX = sumX + particle.X
            sumY = sumY + particle.Y

        avgX = int(sumX / self.MAXPARTICLES)
        avgY = int(sumY / self.MAXPARTICLES)

        raio = int(self.calcDesvioPadrao())
        cv2.circle(frame,(int(avgX),int(avgY)),raio,cor,2)
        cv2.circle(frame,(int(avgX),int(avgY)),2,cor,-1)
        cv2.circle(frame,(int(avgX),int(avgY)),1,(0,255,255),-1)
        
        text = "cX:{} | cY:{} | SD:{}".format(avgX,avgY,raio)
        cv2.putText(frame,text,(avgX+raio,avgY+raio),cv2.FONT_HERSHEY_SIMPLEX,0.5, cor,2)
            
        return frame

    def __print_vet_particles(self,vet):
        print("v ---------------------- v")
        for particle in vet:
            particle.print()
        print("^ ---------------------- ^")

    def __centroid_predicted(self):
        sumX = 0
        sumY = 0

        for particle in self.vet_particles_predicted:
            sumX = sumX + particle.X
            sumY = sumY + particle.Y

        avgX = int(sumX / self.MAXPARTICLES)
        avgY = int(sumY / self.MAXPARTICLES)
        return (avgX,avgY)

    def filter_steps(self,center): # nome sugerido: predict_movement()
        self.vet_particles_predicted = self.__prediction()

        if center is not None:
            # print("[INFO] - < tracking >")
            self.__correction(center)
            self.__normalize()
            self.__resort()
            self.__countToMaxFrameLost = 0
        else:
            print("[INFO] - < missing center > | cont: {} | max: {} to lose tracking.".format(self.__countToMaxFrameLost,self.MAXFRAMELOST))
            self.__vet_particles = self.vet_particles_predicted
            self.__countToMaxFrameLost = self.__countToMaxFrameLost + 1

        if self.__countToMaxFrameLost >= self.MAXFRAMELOST:
            self.__vet_particles = None
            print("[INFO] - < LOST TRACKING >")
            return False

        # return self.vet_particles_predicted
        return self.__centroid_predicted()

    def calcDesvioPadrao(self):
        ordinaryParticles = []
        for particles in self.vet_particles_predicted:
            ordinaryParticles.append([particles.X,particles.Y])
        
        return std_distance(ordinaryParticles)