import numpy as np
from matplotlib import pyplot as plt
from utils import rk4

#parameters for lead and ego vehicle
c1 = 0.001
c2 = 0.01
c3 = 0.2

m_ego = 2000
m_lead = 2000

mu = 0.7
g = 9.81

#dynamics for lead and ego vehicles
def f(t,x,u):
    u_ego = u[0]
    u_lead = u[1]

    x_ego = x[0]
    v_ego = x[1]
    x_lead = x[2]
    v_lead = x[3]

    Fr_ego = c1 * np.sign(v_ego) + c2 * v_ego + c3 * v_ego ** 2
    Fr_lead = c1 * np.sign(v_lead) + c2 * v_lead + c3 * v_lead ** 2

    return np.array([v_ego,
            1/m_ego * (-Fr_ego + u_ego),
            v_lead,
            1 / m_lead * (-Fr_lead + u_lead)])


#simulation parameters
T = 30
dt = 0.1

tvec = np.arange(0,T+dt,dt)
N = len(tvec)
X = np.zeros((N,4))
X[0,:] = [0,25,50,25]
U = np.zeros((N-1,2))

for i in range(len(tvec)-1):
    u_ego = np.minimum(np.maximum(-m_ego/5*(X[i, 1] - 30), -mu * m_ego * g), mu*m_ego*g)
    u_lead = np.minimum(np.maximum(-m_lead/5*(X[i, 3] - 20), -mu * m_lead * g), mu * m_lead * g)

    U[i,:] = [u_ego,u_lead]

    X[i+1,:] = rk4(f,tvec[i],X[i,:],U[i,:],dt)

#plot results
fig,ax = plt.subplots(nrows=3,ncols=1)

ax[0].plot(tvec,X[:,2]-X[:,0])
ax[0].set_ylabel('xlead-xego [m]')

ax[1].plot(tvec,X[:,1])
ax[1].plot(tvec,X[:,3])
ax[1].set_ylabel('v [m/s]')

ax[2].plot(tvec[:-1],U[:,0]/m_ego)
ax[2].plot(tvec[:-1],U[:,1]/m_lead)
ax[2].plot(tvec,mu*g*np.ones(N),'r--')
ax[2].plot(tvec,-mu*g*np.ones(N),'r--')
ax[2].set_xlabel('t [s]')
ax[2].set_ylabel('u/m [m/s^2]')

fig.tight_layout()
plt.show()




