from matplotlib import animation
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import pickle


fig = plt.figure()
ax = fig.add_subplot(111 , projection="3d")

startFrame = 0



connections = [(0, 1), (1, 2), (2, 3), (3, 4), 
                             (2, 5), (5, 6), (6, 7), (7, 8), 
                             (2, 9), (9, 10), (10, 11), (11, 12),
                             (0, 13), (13, 14), (14, 15), 
                             (0, 16), (16, 17), (17, 18)]

def updateHumanObj(frame, *fargs):
    global startFrame
    print("current frame")
    print(frame)
    print()
    ax.clear()
    bodyData, objPoint, scat = fargs
    z_points = bodyData[frame, :, 2] #* -1.0
    x_points = bodyData[frame, :, 0]
    y_points = bodyData[frame, :, 1]
    for connect in connections:
        a,b = connect
        ax.plot([x_points[a], x_points[b]],[y_points[a],y_points[b]],[z_points[a],z_points[b]], color="b")
    ax.scatter3D(x_points, y_points, z_points, color="r")

    thisObjPoint = objPoint[frame].reshape((12,3))
    z_points = thisObjPoint[ :, 2] #* -1.0
    x_points = thisObjPoint[ :, 0] 
    y_points = thisObjPoint[ :, 1]
    ax.scatter3D(x_points, y_points, z_points, color="g")
    #'''
    ax.plot([0.0, 0.0],[-0.7,1.2,],[0.0,0.0], color="b")
    ax.plot([-1.2,1.2],[0.0,0.0,],[0.0,0.0], color="r")
    ax.plot([0.0, 0.0],[0.0,0.0,],[-1.2,1.2], color="g")
    
    return ax


def visualize(skeledonData, objPoint, startFrame = 0):
    global fig, ax
    bodyData = skeledonData
    lenFrame = skeledonData.shape[0]
    
    bodyData = bodyData.reshape((lenFrame, 21, 3))
    ax.yaxis.set_label_position("top")
    ax.view_init(elev=117., azim=-88.)
    scat = ax.scatter(bodyData[0,:,0], bodyData[0,:,1], bodyData[0,:,2], c='r', marker = 'o',alpha=0.5, s=100)
    #time.sleep(.01)

    ani = animation.FuncAnimation(fig, updateHumanObj, frames= range(lenFrame), interval = 50, repeat_delay=100,
                                  fargs=(bodyData, objPoint, scat))
    plt.show()
    

pathName = "./data/actor5_table_push_001/actor5_table_push_001.data"

with open(pathName, 'rb') as f:
    dataList = pickle.load(f)[0]
    
    
    startFrame = 0
    endFrame = len(dataList[0]) - 1   # len(dataList[0]) - 1
    #endFrame = startFrame + 2
    
    videoLength = endFrame - startFrame
    skeledonData = dataList[0][startFrame:endFrame:1][:]
    videoLength = len(skeledonData)
    skeledonData = np.array(skeledonData).reshape((videoLength, 21, 3))
    
    objData = dataList[3][startFrame:endFrame:1][:]
    objData = np.array(objData).reshape((videoLength, 12, 3))
    
    visualize(skeledonData, objData, 0)