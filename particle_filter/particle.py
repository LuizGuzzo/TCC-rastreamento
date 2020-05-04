import imutils
import cv2
import math
import time
import random
import numpy as np

# DELTA_T = 1/30
# VELMAX = 3000

class particle():

    def __init__(self,DELTA_T = 1/30, VELMAX = 3000):
        self.DELTA_T = DELTA_T
        self.VELMAX = VELMAX
        self.X = None
        self.Y = None
        self.Vx = None
        self.Vy = None
        self.weight = 0
        self.toNormalize = 0

        # ToDo: deixar tais parametros configuraveis pelo cliente
        # self.deltaT = self.DELTA_T
        # self.velMax = self.VELMAX

    def setAll(self,particle):
        self.X = int(particle.X)
        self.Y = int(particle.Y)
        self.Vx = particle.Vx
        self.Vy = particle.Vy
        self.weight = particle.weight
        self.toNormalize = particle.toNormalize
        self.DELTA_T = particle.DELTA_T
        self.VELMAX = particle.VELMAX

    def start(self,center):
        # self.X = random.gauss(center[0],50)
        # self.Y = random.gauss(center[1],50)
        self.X = int(center[0])
        self.Y = int(center[1])
        self.Vx = random.uniform(-self.VELMAX, self.VELMAX)  # velocidade 1549 pixels/sec
        self.Vy = random.uniform(-self.VELMAX, self.VELMAX)  # velocidade 1549 pixels/sec
        self.weight = 0
        self.toNormalize = 0

    def prediction(self):
        # REVER ESSA FUNÇÃO DE VELOCIDADE QUE TA MUITO ESTRANHA, ELA ESTA BATENDO O LIMITE TODA HORA

        # print(" ------------- PRED INI ------------- ")

        a = pow(self.VELMAX * 0.01, 2)
        b = random.gauss(0, a)
        self.Vx = self.Vx + b
        # print("pow(self.VELMAX * 0.1, 2): {: >10.3f}| random.gauss(0, a): {: >10.3f}| Vx: {: >10.3f}|".format(a, b, self.Vx))

        b = random.gauss(0, a)
        self.Vy = self.Vy + b
        # print("pow(self.VELMAX * 0.1, 2): {: >10.3f}| random.gauss(0, a): {: >10.3f}| Vy: {: >10.3f}|".format(a, b, self.Vy))

        if self.Vx > self.VELMAX :
            self.Vx = self.VELMAX

        if self.Vy > self.VELMAX :
            self.Vy = self.VELMAX

        if self.Vx < -self.VELMAX :
            self.Vx = -self.VELMAX

        if self.Vy < -self.VELMAX :
            self.Vy = -self.VELMAX

        a = self.DELTA_T * self.Vx
        self.X = int(self.X + a)
        # print("self.DELTA_T * self.Vx: {}|self.X + a: {}".format(a, self.X))

        a = self.DELTA_T * self.Vy
        self.Y = int(self.Y + a)
        # print("self.DELTA_T * self.Vy: {}|self.Y + a: {}".format(a, self.Y))
        # print(" ------------- PRED END ------------- ")

        return self

    def correction(self,center):
        Dt = math.sqrt(pow((center[0] - self.X), 2) + pow((center[1] - self.Y), 2))
        # self.toNormalize = 1/(np.exp( Dt ))
        self.toNormalize = np.exp( -Dt )
        return self

    def normaliza(self,sumToNormalize):
        self.weight = self.toNormalize/sumToNormalize
        return self

    def print(self):
        print("X: {: >5}|Y: {: >5}| Vx: {: >10.3f}| Vy: {: >10.3f}| weight: {: >8.3f}| tN: {: >5.15f}|".format(self.X, self.Y, self.Vx, self.Vy, self.weight, self.toNormalize))
        return self