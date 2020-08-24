import modern_robotics as mr 
import numpy as np 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation

class Robot:
    def __init__(self,M,Slist,Vs):
        self.M = M 
        self.Slist = Slist
        self.Vs = Vs
    class Joint:
        def __init__(self,x,y,z):
            self.x = x
            self.y = y
            self.z = z 

#-------------------------------------Animation Functions-------------------------------------------------------------
def BuildFrame(ax,X,Y,Z,size):
    x = [[X, X+size],[Y,Y],[Z,Z]]
    y = [[X,X],[Y, Y+size],[Z,Z]]
    z = [[X,X],[Y,Y],[Z, Z+size]]
    ax.plot3D(x[0],x[1],x[2], 'red')
    ax.plot3D(y[0],y[1],y[2], 'green')
    ax.plot3D(z[0],z[1],z[2], 'blue')

def PlotRobot(thetalist):
    
    xline, yline, zline = FindJointPositions(thetalist)
    

    ax.plot3D(xline, yline, zline, 'black')
    BuildFrame(ax,0,0,0,100)
    ax.scatter(xline[:6],yline[:6],zline[:6], s = 30)
    
    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect('key_release_event',
                                 lambda event: [exit(0) if event.key == 'escape' else None])

    ax.scatter(xline[6],yline[6],zline[6], s = 20,color = 'red')

def GerarGrafico(minx,maxx,miny,maxy,minz,maxz):
    global fig
    global ax
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlim3d(minx,maxx)
    ax.set_ylim3d(miny,maxy)
    ax.set_zlim3d(minz,maxz)

def animarTrajetoria(Ti,Tf,iterations):
    GerarGrafico(-500,500,-500,500,0,600)
    traj = Trajectory(Ti,Tf,iterations)
    def animate(i):
        ax.clear()
        ax.set_xlim3d(-500,500)
        ax.set_ylim3d(-500,500)
        ax.set_zlim3d(0,600)
        PlotRobot(traj[i])
    ani = animation.FuncAnimation(fig,animate,iterations)
    plt.show()
#-------------------------------------Kinematics Functions-------------------------------------------------------------
def FindJointPositions(thetalist):
    M2 = np.array([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,1,145],
                   [0,0,0,1]])

    M3 = np.array([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,1,351],
                   [0,0,0,1]])

    M4 = np.array([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,1,468],
                   [0,0,0,1]])

    M5 = np.array([[1,0,0,0],
                   [0,1,0,0],
                   [0,0,1,468],
                   [0,0,0,1]])

    M6 = np.array([[1,0,0,0],
                   [0,1,0,117],
                   [0,0,1,468],
                   [0,0,0,1]])     
    
    M7 = np.array([[1,0,0,0],
                   [0,1,0,117],
                   [0,0,1,541],
                   [0,0,0,1]])

    Slist = Robot.Slist

    TJ1 = [[1,0,0,0],
          [0,1,0,0],
          [0,0,1,0],
          [0,0,0,1]] 
    TJ2 = mr.FKinSpace(M2,Slist[:2].T,thetalist[:2])
    TJ3 = mr.FKinSpace(M3,Slist[:3].T,thetalist[:3])
    TJ4 = mr.FKinSpace(M4,Slist[:4].T,thetalist[:4])
    TJ5 = mr.FKinSpace(M5,Slist[:5].T,thetalist[:5])
    TJ6 = mr.FKinSpace(M6,Slist[:6].T,thetalist[:6])
    TEE = mr.FKinSpace(M7,Slist[:6].T,thetalist[:6])

    J1 = Robot.Joint(TJ1[0][3], TJ1[1][3], TJ1[2][3])
    J2 = Robot.Joint(TJ2[0][3], TJ2[1][3], TJ2[2][3])
    J3 = Robot.Joint(TJ3[0][3], TJ3[1][3], TJ3[2][3])
    J4 = Robot.Joint(TJ4[0][3], TJ4[1][3], TJ4[2][3])
    J5 = Robot.Joint(TJ5[0][3], TJ5[1][3], TJ5[2][3])
    J6 = Robot.Joint(TJ6[0][3], TJ6[1][3], TJ6[2][3])
    EE = Robot.Joint(TEE[0][3], TEE[1][3], TEE[2][3])
    
    Xlist = [J1.x,J2.x,J3.x,J4.x,J5.x,J6.x,EE.x]
    Ylist = [J1.y,J2.y,J3.y,J4.y,J5.y,J6.y,EE.y]
    Zlist = [J1.z,J2.z,J3.z,J4.z,J5.z,J6.z,EE.z]


    return Xlist, Ylist, Zlist

def JointsVelocity(Slist,thetalist,Vs):
    J = mr.JacobianSpace(Slist,thetalist)
    thetaVel = np.matmul(Vs,J.T)
    return thetaVel

def JointLenght(J1X,J1Y,J2X,J2Y):
    Lenght = np.sqrt((J1X-J2X)**2 + (J1Y-J2Y)**2)
    return Lenght

def BestTheta0(X,Y,Z,d1,d2):
    Theta1 = np.arctan2(Y,X)
    Theta3 = -np.cos((X**2+Y**2+Z**2-(d1**2)*(d2**2)) / (2*d1*d2))
    r = np.sqrt(d1**2 + d2**2 - 2*d2*d2*np.cos(np.pi - Theta3))
    Theta2 = np.pi/2 - (np.arcsin(Z/r) - np.arctan(d2*np.sin(Theta3)/(d1 + d2*np.cos(Theta3))))
    return [Theta1,Theta2,Theta3,0,0,0]

def Trajectory(Ti,Tf,iterations):
    thetalist0 = BestTheta0(Ti[0][3],Ti[1][3],Ti[2][3],206,189)
    AngleList = []
    VelocityList = []
    traj = mr.CartesianTrajectory(Ti,Tf,10,iterations,3)
    for i in range(iterations):
        if i == 0:
            Ik = mr.IKinSpace(Robot.Slist.T,Robot.M,traj[i],thetalist0, eomg = 0.00001, ev = 0.000001)
            Vel = JointsVelocity(Robot.Slist,Ik,Robot.Vs)
            VelocityList.append(Vel)
            AngleList.append(Ik[0])
        else:
            Ik = mr.IKinSpace(Robot.Slist.T,Robot.M,traj[i],AngleList[i-1], eomg = 0.000001, ev = 0.000001)
            Vel = JointsVelocity(Robot.Slist,Ik,Robot.Vs)
            VelocityList.append(Vel)
            AngleList.append(Ik[0])
    return AngleList

def InverseKinematics(Slist,M,T):
    Ik = mr.IKinSpace(Slist.T,M,T,BestTheta0(Ti[0][3],Ti[1][3],Ti[2][3],206,289),eomg=0.000001,ev=0.000001)
    #Pack angles in a 360 degree range
    for i in range(6):
        if Ik[0][i] > 0:
            while Ik[0][i] > np.pi*2:
                Ik[0][i] = Ik[0][i] - np.pi*2  
        if Ik[0][i] < 0:  
            while Ik[0][i] < -np.pi*2:
                Ik[0][i] = Ik[0][i] + np.pi*2 
    return Ik[0]
#-----------------------------------------Especificações do Robo UR5-------------------------------------------------------------

#M Matrix:
Robot.M = np.array([[1,0,0,0],
                    [0,1,0,117],
                    [0,0,1,541],
                    [0,0,0,1]])

#S list:
Robot.Slist = np.array([[0,0,1,0,0,0],
                        [0,1,0,-145,0,0],
                        [0,1,0,-351,0,0],
                        [0,0,1,0,0,0],
                        [0,1,0,-468,0,0],
                        [0,0,1,117,0,0]])

#Vs Velocity Matrix
Robot.Vs = np.array([100,100,100,100,100,100])

#--------------------------------------------------------Main Program--------------------------------------------------------------

#Desired Configuration:
Ti = np.array([[-1,0,0,0],
               [0,1,0,200],
               [0,0,-1,0],
               [0,0,0,1]])

Tf = np.array([[-1,0,0,100],
               [0,1,0,200],
               [0,0,-1,300],
               [0,0,0,1]])
    


Fk = mr.FKinSpace(Robot.M,Robot.Slist.T,[np.deg2rad(90),np.deg2rad(0),np.deg2rad(90),np.deg2rad(90),np.deg2rad(90),np.deg2rad(0)])
Ik = mr.IKinSpace(Robot.Slist.T,Robot.M,Ti,BestTheta0(Ti[0][3],Ti[1][3],Ti[2][3],206,239),eomg=0.0001,ev=0.0001)

animarTrajetoria(Ti,Tf,10)

for i in range(6):
    if Ik[0][i] > 0:
        while Ik[0][i] > np.pi*2:
            Ik[0][i] = Ik[0][i] - np.pi*2  
    if Ik[0][i] < 0:  
        while Ik[0][i] < -np.pi*2:
            Ik[0][i] = Ik[0][i] + np.pi*2 

for i in range(6):        
    Ik[0][i] = round(np.rad2deg(Ik[0][i]),0)
    Ik[0][i] = float(Ik[0][i])











