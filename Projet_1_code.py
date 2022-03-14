# Projet_1:Double pendule

import numpy as np
import matplotlib.pyplot as plt
import  matplotlib.animation as animation

#from IPython.display import HTML
#plt.rc('figure', figsize=(12,9))

# Constantes du système
g = 9.81
m1 = 1
m2 = 1
l1 = 1
l2 = 1
tmax = 15
k = 0.01

#conditions initialles
th0_1 = 1
th0_2 = np.pi/2
p0_1 = 0
p0_2 = 0

#définision F(pour euler avant donc F[i])
def thstar(th1, th2, p1, p2):
    c1 = (m2*l2**2*p1-m2*l1*l2*np.cos(th1-th2)*p2)/(m2*l1**2*l2**2*(m1+m2*(np.sin(th1-th2))**2))
    c2 = ((m1+m2)*l1**2*p2-m2*l1*l2*np.cos(th1-th2)*p1)/(m2*l1**2*l2**2*(m1+m2*(np.sin(th1-th2))**2))
    return c1, c2

def pstar(th1, th2, dth1, dth2):
    c1 = -(m1+m2)*g*l1*np.sin(th1)-m2*l1*l2*dth1*dth2*np.sin(th1-th2)
    c2 = m2*l1*l2*dth1*dth2*np.sin(th1-th2)-m2*g*l2*np.sin(th2)
    return c1, c2    


# Schéma d'Euler explicite
def euler(th0_1, th0_2, p0_1, p0_2, k, tmax):
    
    N = int(tmax/k)
    t = np.linspace(0, tmax, N+1)
    sol_th = np.zeros((2,N+1))
    sol_p = np.zeros((2,N+1))
    sol_th[0,0] = th0_1
    sol_th[1,0] = th0_2
    sol_p[0,0] = p0_1
    sol_p[1,0] = p0_2

    for i in range(N):
        F1, F2 = thstar(sol_th[0,i], sol_th[1,i], sol_p[0,i], sol_p[1,i])
        sol_th[0,i+1] = sol_th[0,i] + k*F1
        sol_th[1,i+1] = sol_th[1,i] + k*F2
        F3, F4 = pstar(sol_th[0,i], sol_th[1,i], F1, F2)
        sol_p[0,i+1] = sol_p[0,i] + k*F3
        sol_p[1,i+1] = sol_p[1,i] + k*F4
    return t, sol_th, sol_p

t, sol_th, sol_p = euler(th0_1, th0_2, p0_1, p0_2, k, tmax)
t, sol_th1 ,sol_p1 = euler(th0_1+0.1,th0_2+0.1, p0_1+0.1,p0_2+0.1,k,tmax)



#position des pendules
x1 = l1*np.sin(sol_th[0,:])
y1 = -l1*np.cos(sol_th[0,:])
x2 = x1 + l2*np.sin(sol_th[1,:])
y2 = y1 - l2*np.cos(sol_th[1,:])

x11 = l1*np.sin(sol_th1[0,:])
y11 = -l1*np.cos(sol_th1[0,:])
x21 = x1 + l2*np.sin(sol_th1[1,:])
y21 = y1 - l2*np.cos(sol_th1[1,:])

#graph traj des pendules
fig, ax1=plt.subplots()
ax1.plot(x1, y1, label="traj m1")
ax1.plot(x2, y2, label="traj m2")
#ax1.plot(x11, y11, label= "traj m11")
#ax1.plot(x21, y21, label = "traj m21")
plt.title("Trajectoir des penules")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

#graph th1 th2
fig, ax=plt.subplots()
ax.plot(t,sol_th[0,:], label = "theta 1")
ax.plot(t,sol_th[1,:], label = "theta 2")
plt.legend()
plt.show()




#Energie

def Energie_calc(p,g,m,h):
    return((p*p)/(2*m)- m*g*h)

E1 = Energie_calc(sol_p[0,:],g,m1,y1)
E2 = Energie_calc(sol_p[1,:],g,m2,y2)
E = E1 +E2
fig, ax2=plt.subplots()
#ax2.plot(t, E1, label = 'E1')
#ax2.plot(t,E2, label = 'E2')
ax2.plot(t, E, label = 'E1+E2')
plt.legend()
plt.show()




# animation
fig, ax = plt.subplots()
def visualisation(t, th):
    x1 = l1*np.sin(sol_th[0,:])
    y1 = -l1*np.cos(sol_th[0,:])
    x2 = x1 + l2*np.sin(sol_th[1,:])
    y2 = y1 - l2*np.cos(sol_th[1,:])
    ax.set_xlim(-(l1+l2), l1+l2)
    ax.set_ylim(-(l1+l2), l1+l2)
    line, = plt.plot([0,x1[0], x2[0]], [0, y1[0], y2[0]], linewidth=1, marker="o", color="red")
    plt.title("position initiale du pendule")
    plt.ylabel("y")
    plt.xlabel("x")
    def barre(i):
        line.set_data([0,x1[i], x2[i]], [0, y1[i], y2[i]])
        return line,
    
    ani = animation.FuncAnimation(fig, barre, frames=range(np.size(x1)), blit=True, interval=50)
    return ani

ani = visualisation(t, sol_th)
