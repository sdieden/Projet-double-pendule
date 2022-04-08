# Projet_1:Double pendule

import numpy as np
import matplotlib.pyplot as plt
import  matplotlib.animation as animation
from collections import deque

from IPython.display import HTML, Image
from IPython import display


# Constantes du système
g = 9.81
m1 = 1
m2 = 1
l1 = 1
l2 = 1
tmax = 10
k = 0.01
L = l1+l2
history_len = 500  # how many trajectory points to display


#conditions initialles
th0_1 = np.pi/32
th0_2 = np.pi/32
p0_1 = 0
p0_2 = 0

#définition F
def thstar(th1, th2, p1, p2):
    c1 = (m2*l2**2*p1-m2*l1*l2*np.cos(th1-th2)*p2)/(m2*l1**2*l2**2*(m1+m2*(np.sin(th1-th2))**2))
    c2 = ((m1+m2)*l1**2*p2-m2*l1*l2*np.cos(th1-th2)*p1)/(m2*l1**2*l2**2*(m1+m2*(np.sin(th1-th2))**2))
    return c1, c2

def pstar(th1, th2, dth1, dth2):
    c1 = -(m1+m2)*g*l1*np.sin(th1)-m2*l1*l2*dth1*dth2*np.sin(th1-th2)
    c2 = m2*l1*l2*dth1*dth2*np.sin(th1-th2)-m2*g*l2*np.sin(th2)
    return c1, c2    


#Def Euler avant 
def Euler(th0_1, th0_2, p0_1, p0_2, k, tmax):
    
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

#Def Range-Kuta 4
def RK4(th0_1, th0_2, p0_1, p0_2, k, tmax):
    
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
        
        k11,k21 = thstar(sol_th[0,i], sol_th[1,i], sol_p[0,i], sol_p[1,i])
        k12,k22 = thstar(sol_th[0,i]+(k11/2), sol_th[1,i]+(k21/2), sol_p[0,i]+(k11/2), sol_p[1,i]+(k21/2))
        k13,k23 = thstar(sol_th[0,i]+(k12/2), sol_th[1,i]+(k22/2), sol_p[0,i]+(k12/2), sol_p[1,i]+(k22/2))
        k14,k24 = thstar(sol_th[0,i]+k12, sol_th[1,i]+k22, sol_p[0,i]+k12, sol_p[1,i]+k22)
        
        H11, H21 = k*k11, k*k21
        H12, H22 = k*k12, k*k22
        H13, H23 = k*k13, k*k23
        H14, H24 = k*k14, k*k24
        
        sol_th[0,i+1] = sol_th[0,i] + (H11+(2*H12)+(2*H13)+H14)/6
        sol_th[1,i+1] = sol_th[1,i] + (H21+(2*H22)+(2*H23)+H24)/6
        
        Q11,Q21 = pstar(sol_th[0,i], sol_th[1,i], F1, F2)
        Q12,Q22 = pstar(sol_th[0,i]+(Q11/2), sol_th[1,i]+(Q21/2), F1+(Q11/2), F2+(Q21/2))
        Q13,Q23 = pstar(sol_th[0,i]+(Q12/2), sol_th[1,i]+(Q22/2), F1+(Q12/2), F2+(Q22/2))
        Q14,Q24 = pstar(sol_th[0,i]+Q12, sol_th[1,i]+Q22, F1+Q12, F2+Q22)
        
        q11, q21 = k*Q11, k*Q21
        q12, q22 = k*Q12, k*Q22
        q13, q23 = k*Q13, k*Q23
        q14, q24 = k*Q14, k*Q24
        
        sol_p[0,i+1] = sol_p[0,i] + k*(q11+(2*q12)+(2*q13)+q14)/6
        sol_p[1,i+1] = sol_p[1,i] + k*(q21+(2*q22)+(2*q23)+q24)/6
    return t, sol_th, sol_p



t, sol_th, sol_p = Euler(th0_1, th0_2, p0_1, p0_2, k, tmax)
t, sol_th1 ,sol_p1 = Euler(th0_1+0.01,th0_2+0.01, p0_1+0.01,p0_2+0.01,k,tmax)

### Ne marche pas avec RK4
# t, sol_th, sol_p = RK4(th0_1, th0_2, p0_1, p0_2, k, tmax)
# t, sol_th1 ,sol_p1 = RK4(th0_1+0.01,th0_2+0.01, p0_1+0.01,p0_2+0.01,k,tmax)

#position des pendules
x1 = l1*np.sin(sol_th[0,:])
y1 = -l1*np.cos(sol_th[0,:])
x2 = x1 + l2*np.sin(sol_th[1,:])
y2 = y1 - l2*np.cos(sol_th[1,:])

x11 = l1*np.sin(sol_th1[0,:])
y11 = -l1*np.cos(sol_th1[0,:])
x21 = x1 + l2*np.sin(sol_th1[1,:])
y21 = y1 - l2*np.cos(sol_th1[1,:])

pos1 = x1 + y1
pos2 = x2 + y2

#Vitesses des masses:
def derivee(x, t):
    d = []
    for i in range(0, len(x)-1):
        dt = t[i+1]-t[i]
        dx = x[i+1]-x[i]
        d.append(dx/dt)
    return d    

v1 = derivee(pos1, t)
v2 = derivee(pos2, t)


fig = plt.figure(figsize=(5, 4))
ax = fig.add_subplot(autoscale_on=False, xlim=(-L, L), ylim=(-L, 1.))
ax.set_aspect('equal')
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
trace, = ax.plot([], [], '.-', lw=1, ms=2)
time_template = 'time = %.1fs'
time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
history_x, history_y = deque(maxlen=history_len), deque(maxlen=history_len)

# animation
def animate(i):
    thisx = [0, x1[i], x2[i]]
    thisy = [0, y1[i], y2[i]]

    if i == 0 : 
        history_x.clear()
        history_y.clear()
    history_x.appendleft(thisx[2])
    history_y.appendleft(thisy[2])

    line.set_data(thisx, thisy)
    trace.set_data(history_x, history_y)
    time_text.set_text(time_template % (i*k))
    return line, trace, time_text

ani = animation.FuncAnimation(
    fig, animate,int(tmax/k), interval=k*1000, blit=True) #add len(y) pr éviter bug -> déterminer y c'est quoi ici
plt.show()
video = ani.to_html5_video()
# html = display.HTML(video)
# display.display(html)
# plt.close()

# Writer=animation.writers['ffmpeg']
# writer=Writer(fps=10, metadata=dict(artist='Me'),bitrate=1800)
# ani.save('animdp.mp4',writer=writer)



# fig, ax = plt.subplots()
# def visualisation(t, th):
#     x1 = l1*np.sin(sol_th[0,:])
#     y1 = -l1*np.cos(sol_th[0,:])
#     x2 = x1 + l2*np.sin(sol_th[1,:])
#     y2 = y1 - l2*np.cos(sol_th[1,:])
#     ax.set_xlim(-(l1+l2), l1+l2)
#     ax.set_ylim(-(l1+l2), l1+l2)
#     line, = plt.plot([0,x1[0], x2[0]], [0, y1[0], y2[0]], linewidth=1, marker="o", color="red")
#     plt.title("position initiale du pendule")
#     plt.ylabel("y")
#     plt.xlabel("x")
#     def barre(i):
#         line.set_data([0,x1[i], x2[i]], [0, y1[i], y2[i]])
#         return line,
    
#     ani = animation.FuncAnimation(fig, barre, frames=range(np.size(x1)), blit=True, interval=50)
#     return ani

# ani = visualisation(t, sol_th)


#HTML(ani.to_html5_video())
