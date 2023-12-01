import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
import pickle
import scipy.spatial as ss
import numpy as np
import random
import time
from particle_sys import ParticlesControl

visualize = True
useConstantForce = False

Total_Sample_NUM = 5

unitAll = 0.15
floorHeight = -10.5


'''
Use these codes to setup your experiment objects
'''
#chair1 setting
length = 3.0
height1 = 3.0
height2 = 6.0
width = 3.0
half_W = width / 2.0
half_L = length / 2.0
weight = 1.5

#[x,y,z,weight]
#chair1
descriptorPoints = [
        #1
        [-half_W, floorHeight + height1 , -half_L, 1.0],
        #2 
        [-half_W, floorHeight + height1 , half_L, 1.0], 
        #3
        [half_W, floorHeight + height1 , -half_L, 1.0], 
        #4
        [half_W, floorHeight + height2 , -half_L, 1.0], 
        #5
        [half_W, floorHeight + height1 , half_L, 1.0], 
        #6
        [-half_W, floorHeight + height2 , -half_L, 1.0], 
        #7
        [0.0, floorHeight + height1, 0.0, 1.0], 
        #8
        [0.0, floorHeight + (0.5*height1 + 0.5*height2), -half_L, 1.0], 
        #9
        [-half_W, floorHeight, half_L, 1.0], 
        #10
        [half_W, floorHeight, half_L, 1.0], 
        #11
        [-half_W, floorHeight, -half_L, 1.0], 
        #12
        [half_W, floorHeight, -half_L, 1.0]
    ]
shouldDraw = [(1,6), (4,6), (3,4), (1,3), (1,2), (2,5), (3,5), (1,11), (3,12), (2,9), (5,10)]  #small must be first
for i in range(len(shouldDraw)):
    shouldDraw[i] = (shouldDraw[i][0]-1, shouldDraw[i][1]-1)


#table setting
'''
length = 12.0
height1 = 7.0
height2 = 3.0
width = 10.0
half_W = width / 2.0
half_L = length / 2.0
weight = 1.0

#[x,y,z,weight]
#table
descriptorPoints = [
        #1
        [0.0, floorHeight + height1 + height2, -half_L, 1.0],
        #2 
        [half_W, floorHeight + height1, -half_L, 0.5], 
        #3
        [half_W, floorHeight + height1 + height2, 0.0, 1.0], 
        #4
        [0.0, floorHeight + height1 +height2, half_L, 1.0], 
        #5
        [-half_W, floorHeight + height1 +height2, 0.0, 1.0], 
        #6
        [half_W, floorHeight + height1, half_L, 0.5], 
        #7
        [-half_W, floorHeight + height1, half_L, 0.5], 
        #8
        [-half_W, floorHeight, half_L, 0.2], 
        #9
        [half_W, floorHeight, half_L, 0.2], 
        #10
        [-half_W, floorHeight, -half_L, 0.2], 
        #11
        [half_W, floorHeight, -half_L, 0.2], 
        #12
        [-half_W, floorHeight+height1, -half_L, 0.5]
    ]
shouldDraw = [(1,5), (4,5), (3,4), (1,3), (10,12), (2,11), (6,9), (7,8)]  #small must be first
for i in range(len(shouldDraw)):
    shouldDraw[i] = (shouldDraw[i][0]-1, shouldDraw[i][1]-1)
'''


            


def updating(theControl):
    if visualize:
        glBegin(GL_LINES)
        for constraint in theControl.constraints:
            if not constraint.visibale:
                continue
            glColor(1.0, 1.0, 1.0)
            selected1 = theControl.particles[constraint.index0]
            selected2 = theControl.particles[constraint.index1]
            glVertex3fv((selected1.x, selected1.y, selected1.z))
            glVertex3fv((selected2.x, selected2.y, selected2.z))
        glEnd()
        #glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER)
        glBegin(GL_QUADS)
        glColor(0.0, 0.4, 0.5)
        glVertex3f(-4, -1.51, -5)
        glVertex3f(-4, -1.51, 5)
        glVertex3f(4, -1.51, 5)
        
        glVertex3f(4, -1.51, -5)
        glEnd()
    else:
        for constraint in theControl.constraints:
            if not constraint.visibale:
                continue
            selected1 = theControl.particles[constraint.index0]
            selected2 = theControl.particles[constraint.index1]
    theControl.update()



def resetControl():
    global descriptorPoints, shouldDraw
    control = ParticlesControl(unitAll, floorHeight)

    for P in descriptorPoints:
        control.addParticle(P[0], P[1], P[2], P[3])
    
    for i in range(0, 11):
        for j in range(i+1, 12):
            if (i, j) in shouldDraw:
                control.addConstraint(i, j, True)
            else:
                control.addConstraint(i, j, False)
    return control

def main():
    global descriptorPoints
    if visualize:
        pygame.init()
        display = (800,600)
        pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (display[0]/display[1]), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)

    vectors = []
    for i in range(12):
        vectors.append(np.array(descriptorPoints[i][:3]) * unitAll)
    origin = np.mean(np.array(vectors), axis = 0)
    for i in range(12):
        vectors[i] = vectors[i] - origin
        
    rx, ry = (0,0)
    tx, ty = (0,0)
    count = -1
    zpos = 10
    rotate = move = False
    DataCount = 0
    AllData = []
    operate = []
    curFrame = []
    np.random.seed(0)
    startF = 200

    constantF = 100

    horiChoice = np.random.normal(0, 50, 100000)
    upChoice = np.random.normal(300, 50, 500)
    
    manul = []
    while DataCount < Total_Sample_NUM:
        if count < 0:
            #         s  l  r  u  b ll  lr lu lb300
            #time.sleep(1)
            print(f"\n\nProcessing Step: {DataCount}")
            selectedPRecod = [0 for i in range(12)]
            manul = [0 for i in range(5)]
            curFrame = []
            control = resetControl()
            selectedP = random.randint(0, 11)
            '''
            selectedP = 10
            if DataCount % 2 == 0:
                selectedP = 1
            '''
            choice = random.randint(0, 4)
            #choice = 2 
            #selectedP = 5
            #print(selectedP)
            #print(choice)
            #x,z,-x,-z,y
            print("select Point")
            print(selectedP)
            print("select mode")
            print(choice)

            manul[choice] += 1
            selectedPRecod[selectedP] += 1
            if not useConstantForce:
                def random_hori_choice(adjust=0.0):
                    return (horiChoice[random.randint(0, 999)] * 1.0 + adjust) / weight

                def random_val(a, b):
                    return random.uniform(a, b) / weight
                selected_particle = control.particles[selectedP]
                mapping = {
                    0: [random_hori_choice(startF), random_val(0, 100), random_val(-20, 20)],
                    1: [random_val(-20, 20), random_val(0, 100), random_hori_choice(startF)],
                    2: [random_hori_choice(-startF), random_val(0, 100), random_val(-20, 20)],
                    3: [random_val(-20, 20), random_val(0, 100), random_hori_choice(-startF)],
                    'default': [random_val(-20, 20), (upChoice[random.randint(0, 500)] * 1.0 + 100) / weight, random_val(-20, 20)]
                }

                ax1, ay1, az1 = mapping.get(choice, mapping['default'])

                selected_particle.ax1 = ax1
                selected_particle.ay1 = ay1
                selected_particle.az1 = az1

            else:
                selected_particle = control.particles[selectedP]

                def adjusted_constantF(sign=1.0):
                    return sign * constantF * 0.7 / weight

                mapping = {
                    0: [adjusted_constantF(), 35 / weight, 0],
                    1: [0, 65 / weight, adjusted_constantF()],
                    2: [adjusted_constantF(-1.0), 65 / weight, 0],
                    3: [0, 35 / weight, adjusted_constantF(-1.0)],
                    'default': [0, constantF * 1.4 / weight, 0]
                }

                ax1, ay1, az1 = mapping.get(choice, mapping['default'])

                selected_particle.ax1 = ax1
                selected_particle.ay1 = ay1
                selected_particle.az1 = az1

            
            DataCount += 1
            operate = [control.particles[selectedP].ax1, control.particles[selectedP].ay1, control.particles[selectedP].az1]
            #print("Current data num:  " + str(DataCount))

        count += 1
        if count > 30:
            control.particles[selectedP].ax1 = 0.0
            control.particles[selectedP].ay1 = 0.0
            control.particles[selectedP].az1 = 0.0
            
        else:
            control.particles[selectedP].selected = False
            control.mouse = False
        
        if count == 70:
            currentVectors = []
            for pCount in range(12):
                currentVectors.append(np.array([control.particles[pCount].x, control.particles[pCount].y, control.particles[pCount].z]))
            originCur = np.mean(np.array(currentVectors), axis = 0)
            trans = originCur - origin
            for i in range(12):
                currentVectors[i] = currentVectors[i] - originCur
            rotat = ss.transform.Rotation.align_vectors(np.array(currentVectors), np.array(vectors))[0]
            curFrame = [rotat.as_euler('xyz').tolist(), trans.tolist()]

            AllData.append([selectedPRecod, operate, manul, curFrame])
            AllData.append([selectedPRecod, operate, manul, curFrame])

            print("Observation:")
            print(trans.tolist())
            #print(rotat.as_euler('xyz').tolist())
            print(rotat.as_euler('xyz', degrees=True).tolist())
            if visualize:
                time.sleep(1.5)
            count = -1
            
        if visualize:
            for e in pygame.event.get():
                if e.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if e.type == KEYDOWN and e.key == K_w:
                    zpos = max(1, zpos-1)
                elif e.type == KEYDOWN and e.key == K_s:
                    zpos += 1
                elif e.type == MOUSEBUTTONDOWN:
                    if e.button == 1: rotate = True
                    elif e.button == 3: move = True
                elif e.type == MOUSEBUTTONUP:
                    if e.button == 1: rotate = False
                    elif e.button == 3: move = False
                elif e.type == MOUSEMOTION:
                    i, j = e.rel
                    if rotate:
                        rx += i
                        ry += j
                    if move:
                        tx += i
                        ty -= j
            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
            glEnable(GL_DEPTH_TEST)
        updating(control)
        if visualize:
            glLoadIdentity()
            
            glTranslatef(0.0,0.0, -zpos)
            glRotate(ry, 1, 0, 0)
            glRotate(rx, 0, 1, 0)
            

            pygame.display.flip()
            pygame.time.wait(20)
    with open('./save_data.pkl', 'wb') as fp:
        pickle.dump(AllData, fp)

if __name__ == "__main__":
    main()
