#!/usr/bin/python3

# --------------------------------------------------------------------------

print('### Script:', __file__)

# --------------------------------------------------------------------------

import math
import sys
import time
import sim
import cv2 as cv
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
plt.show(block=True)

#--------------------------------------------------------------------------
##CREATING VARIABLES FOR THE FUZZY INTERVARLS
ballDesp=None
turnL=None
turnR=None

# --------------------------------------------------------------------------

def  getRobotHandles(clientID):
    # Motor handles
    _,lmh = sim.simxGetObjectHandle(clientID, 'Pioneer_p3dx_leftMotor',
                                     sim.simx_opmode_blocking)
    _,rmh = sim.simxGetObjectHandle(clientID, 'Pioneer_p3dx_rightMotor',
                                     sim.simx_opmode_blocking)

    # Sonar handles
    str = 'Pioneer_p3dx_ultrasonicSensor%d'
    sonar = [0] * 16
    for i in range(16):
        _,h = sim.simxGetObjectHandle(clientID, str % (i+1),
                                       sim.simx_opmode_blocking)
        sonar[i] = h
        sim.simxReadProximitySensor(clientID, h, sim.simx_opmode_streaming)

    # Camera handles
    _,cam = sim.simxGetObjectHandle(clientID, 'Pioneer_p3dx_camera',
                                        sim.simx_opmode_oneshot_wait)
    sim.simxGetVisionSensorImage(clientID, cam, 0, sim.simx_opmode_streaming)
    sim.simxReadVisionSensor(clientID, cam, sim.simx_opmode_streaming)

    return [lmh, rmh], sonar, cam

# --------------------------------------------------------------------------

def setSpeed(clientID, hRobot, lspeed, rspeed):
    sim.simxSetJointTargetVelocity(clientID, hRobot[0][0], lspeed,
                                    sim.simx_opmode_oneshot)
    sim.simxSetJointTargetVelocity(clientID, hRobot[0][1], rspeed,
                                    sim.simx_opmode_oneshot)

# --------------------------------------------------------------------------

def getSonar(clientID, hRobot):
    r = [1.0] * 16
    for i in range(16):
        handle = hRobot[1][i]
        e,s,p,_,_ = sim.simxReadProximitySensor(clientID, handle,
                                                 sim.simx_opmode_buffer)
        if e == sim.simx_return_ok and s:
            r[i] = math.sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2])

    return r

# --------------------------------------------------------------------------

def startFuzzy():
    ballDist = ctrl.Antecedent(np.arange(0.0, 1.1, 0.001), 'ballDist')
    ballDesp=ctrl.Antecedent(np.arange(0.0, 1.1, 0.1), 'ballDesp')
    turnL=ctrl.Consequent(np.arange(0.0, 1.1, 0.1), 'turnL')
    turnR=ctrl.Consequent(np.arange(0.0, 1.1, 0.1), 'turnR')
    velocidad = ctrl.Consequent(np.arange(0, 4.0, 0.1), 'velocidad')

    ballDesp['left']=fuzz.trapmf(ballDesp.universe, [0.0, 0.0, 0.2, 0.5])
    ballDesp['center']=fuzz.trimf(ballDesp.universe, [0.35, 0.5, 0.65])
    ballDesp['right']=fuzz.trapmf(ballDesp.universe, [0.5, 0.8, 1.0, 1.0])

    turnL['backward']=fuzz.trapmf(turnL.universe, [0.0, 0.0, 0.1, 0.5])
    turnL['static']= fuzz.trimf(turnL.universe, [0.4, 0.5, 0.6])
    turnL['forward']= fuzz.trapmf(turnL.universe, [0.5, 0.6, 1.0, 1.0])

    turnR['backward']=fuzz.trapmf(turnR.universe, [0.0, 0.0, 0.1, 0.5])
    turnR['static']= fuzz.trimf(turnR.universe, [0.4, 0.5, 0.6])
    turnR['forward']= fuzz.trapmf(turnR.universe, [0.5, 0.9, 1.0, 1.0])
    
    # Definir conjuntos distancia y velicidad

    ballDist['far'] = fuzz.trapmf(ballDist.universe, [0.0, 0.0, 0.1, 0.4])
    ballDist['normal'] = fuzz.trimf(ballDist.universe, [0.3, 0.4, 0.5])
    ballDist['close'] = fuzz.trapmf(ballDist.universe, [0.4, 0.9, 1.0, 1.0])

    velocidad['slow'] = fuzz.trapmf(velocidad.universe, [0.0, 0.0, 1, 2.0])
    velocidad['normal'] = fuzz.trimf(velocidad.universe, [1, 2.0, 3])
    velocidad['fast'] = fuzz.trapmf(velocidad.universe, [2.0, 3, 4.0, 4.0])

    # Redifinir reglas

    # Reglas movimiento caso lejos
    rule1 = ctrl.Rule(ballDist['far'] & ballDesp['left'], (turnL['backward'], turnR['forward'], velocidad['fast']))
    rule2 = ctrl.Rule(ballDist['far'] & ballDesp['center'], (turnL['forward'], turnR['forward'], velocidad['fast']))
    rule3 = ctrl.Rule(ballDist['far'] & ballDesp['right'], (turnL['forward'], turnR['backward'], velocidad['fast']))

    # Reglas movimiento caso normal
    rule4 = ctrl.Rule(ballDist['normal'] & ballDesp['left'], (turnL['backward'], turnR['forward'], velocidad['normal']))
    rule5 = ctrl.Rule(ballDist['normal'] & ballDesp['center'], (turnL['forward'], turnR['forward'], velocidad['normal']))
    rule6 = ctrl.Rule(ballDist['normal'] & ballDesp['right'], (turnL['forward'], turnR['backward'], velocidad['normal']))

    # Reglas movimiento caso cerca
    rule7 = ctrl.Rule(ballDist['close'] & ballDesp['left'], (turnL['backward'], turnR['forward'], velocidad['slow']))
    rule8 = ctrl.Rule(ballDist['close'] & ballDesp['center'], (turnL['forward'], turnR['forward'], velocidad['slow']))
    rule9 = ctrl.Rule(ballDist['close'] & ballDesp['right'], (turnL['forward'], turnR['backward'], velocidad['slow']))

    turnCtrl= ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    turn= ctrl.ControlSystemSimulation(turnCtrl)
    return turn


# --------------------------------------------------------------------------

def getImageBlob(clientID, hRobot):
    area = 0
    blobs = 0
    coord = []

    rc,ds,pk = sim.simxReadVisionSensor(clientID, hRobot[2],
                                         sim.simx_opmode_buffer)
    rc1, res, image=sim.simxGetVisionSensorImage(clientID, hRobot[2],
                                                0, sim.simx_opmode_buffer)

    if rc == sim.simx_return_ok and pk[1][0]:
        image=np.asarray(image)%255
        image=np.uint8(np.reshape(image, (256, 256, 3)))
        imageG=cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        ret, imageG = cv.threshold(imageG, 1, 255, cv.THRESH_BINARY)
        contours, h=cv.findContours(imageG, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        area=cv.contourArea(contours[0])
        cv.drawContours(image, contours, -1, (0, 100, 0), 3)
        cv.waitKey(35)
        blobs = int(pk[1][0])
        offset = int(pk[1][1])
        for i in range(blobs):
            coord.append(pk[1][4+offset*i])
            coord.append(pk[1][5+offset*i])

    return blobs, coord, area

# --------------------------------------------------------------------------

def avoid(sonar):
    if (sonar[3] < 0.3) or (sonar[4] < 0.3):
        lspeed, rspeed = +1, -1
    elif sonar[1] < 0.2:
        lspeed, rspeed = +1, -1
    elif sonar[5] < 0.2:
        lspeed, rspeed = -1, +1
    else:
        lspeed, rspeed = +0.0, +0.0

    return lspeed, rspeed

# --------------------------------------------------------------------------

#-----------------------------------------------------------------------------

def calcula_direccion_perdida(turn):
    lspeed, rspeed = +0.4, -0.4

    if hasattr(turn, 'out'):
        out = turn.out()
        last_lspeed, last_rspeed = out['turnL'], out['turnR']
        if last_lspeed < last_rspeed:
            lspeed, rspeed = -0.4, +0.4

    return lspeed, rspeed

def seguirBola(coord, turn, area):
    print('Area before: ', area)
    area = area/45000.0 # 
    print('Area: ', area)
    lspeed, rspeed=calcula_direccion_perdida(turn) # Empieza a girar por donde perdió la bola por última vez

    if(len(coord)>0 and area>=0):
        turn.input['ballDesp'] = coord[0]
        turn.input['ballDist'] = area 
        turn.compute()
        out=turn.output

        # He modificado las velocidades en función de la distancia a la bola
        print('Velocidad: ', out['velocidad'])
        lspeed = out['velocidad'] * out['turnL']
        rspeed = out['velocidad'] * out['turnR']
    return lspeed, rspeed





#---------------------------------------------------------------------------

def main():
    print('### Program started')

    print('### Number of arguments:', len(sys.argv), 'arguments.')
    print('### Argument List:', str(sys.argv))

    sim.simxFinish(-1) # just in case, close all opened connections

    port = int(sys.argv[1])
    clientID = sim.simxStart('127.0.0.1', port, True, True, 2000, 5)

    if clientID == -1:
        print('### Failed connecting to remote API server')

    else:
        print('### Connected to remote API server')
        hRobot = getRobotHandles(clientID)

        turn=startFuzzy()

        while sim.simxGetConnectionId(clientID) != -1:
            # Perception
            sonar = getSonar(clientID, hRobot)
            # print '### s', sonar

            blobs, coord, area = getImageBlob(clientID, hRobot)

            print('coord: ', coord)

            # Planning
            ##lspeed, rspeed = avoid(sonar)
            lspeed, rspeed = seguirBola(coord, turn, area)
            print('lspeed: ', lspeed)
            print('rspeed: ', rspeed)
            # Action
            setSpeed(clientID, hRobot, lspeed, rspeed)
            time.sleep(0.05)

        print('### Finishing...')
        sim.simxFinish(clientID)

    print('### Program ended')

# --------------------------------------------------------------------------

if __name__ == '__main__':
    main()
