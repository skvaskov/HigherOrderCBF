import numpy as np
from matplotlib import pyplot as plt
from utils import rk4
import osqp
from scipy import sparse

#parameters for lead and ego vehicle
c0 = 0.1
c1 = 5
c2 = 0.25

m_ego = 1650
m_lead = 1650

ca = 0.4
cd = 0.4
g = 9.81

#dynamics for lead and ego vehicles
def f(t,x,u):
    u_ego = u[0]
    v_lead = u[1]

    x_ego = x[0]
    v_ego = x[1]
    x_lead = x[2]
    #v_lead = x[3]

    Fr_ego = c0 * np.sign(v_ego) + c1 * v_ego + c2 * v_ego ** 2
    #Fr_lead = c1 * np.sign(v_lead) + c2 * v_lead + c3 * v_lead ** 2
    return np.array([v_ego,
            1/m_ego * (-Fr_ego + u_ego),
            v_lead])

def get_state_dependent_matrices_for_qp(x,v_des,epsilon):
    v_ego = x[1]
    Fr_ego = c0 * np.sign(v_ego) + c1 * v_ego + c2 * v_ego ** 2
    q = np.array([-2 * Fr_ego / m_ego ** 2, 0])

    #lyapunov control
    LgVacc = 2*(v_ego-v_des) / m_ego
    LfVacc = -2*(v_ego-v_des) / m_ego * Fr_ego
    Vacc = (v_ego-v_des)**2

    A = sparse.csc_matrix([[1,0],[LgVacc, -1]])
    u = np.array([ca*m_ego*g,-LfVacc-epsilon*Vacc])
    return q, A, u

#setup OSQP
pacc = 1
prob = osqp.OSQP()
P = sparse.csc_matrix([[2/m_ego**2,0],[0,2*pacc]])
l = np.concatenate(([-cd*m_ego*g],-np.inf * np.ones(1)))
x0 = [0,20,100]
v_d = 24
epsilon = 10
q,A,u = get_state_dependent_matrices_for_qp(x0,v_d,epsilon)

# Setup workspace
prob.setup(P, q, A, l, u, warm_start=True)

#simulation parameters
T = 30
dt = 0.1
v_lead = 13.89
tvec = np.arange(0,T+dt,dt)
N = len(tvec)
X = np.zeros((N,3))
X[0,:] = x0
U = np.zeros((N-1,2))

for i in range(len(tvec)-1):
    q_new, A_new, u_new = get_state_dependent_matrices_for_qp(X[i,:], v_d, epsilon)
    prob.update(q=q_new, u=u_new)
    prob.update(Ax=A_new.data)

    res = prob.solve()
    u_ego = res.x[0]

    U[i,:] = [u_ego,v_lead]

    X[i+1,:] = rk4(f,tvec[i],X[i,:],U[i,:],dt)

#plot results
fig,ax = plt.subplots(nrows=3,ncols=1)

ax[0].plot(tvec,X[:,2]-X[:,0])
ax[0].set_ylabel('xlead-xego [m]')

ax[1].plot(tvec,X[:,1])
ax[1].set_ylabel('v [m/s]')

ax[2].plot(tvec[:-1],U[:,0]/m_ego)
ax[2].plot(tvec,ca*g*np.ones(N),'r--')
ax[2].plot(tvec,-cd*g*np.ones(N),'r--')
ax[2].set_xlabel('t [s]')
ax[2].set_ylabel('u/m [m/s^2]')

fig.tight_layout()
plt.show()




