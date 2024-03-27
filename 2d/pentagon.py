#Boundary points
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import numpy as np
import torch
import torch.nn as nn
import math
from scipy.linalg import pinv
from numpy.linalg import lstsq
from scipy.fftpack import fftshift,fftn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import scipy.special as sps
torch.set_default_dtype(torch.float64)

r = 0.5
Boundary_points = []
Boundary_points_1 = []
k = 1002
Boundary_len = k-2
p = (k-2)//5 + 1
print(p)
r_0 = 0.35*np.sqrt(2)
for j in range(5):
    for i in range(p - 1):
        Boundary_points_1.append([0.5-r_0 * np.sin(2*j*np.pi/5) + (r_0 * np.sin(2*j*np.pi/5) - r_0 * np.sin(2*(j + 1)*np.pi/5))*(i + 1)/p, 0.5+r_0 * np.cos(2*j*np.pi/5) + (-r_0 * np.cos(2*j*np.pi/5) + r_0 * np.cos(2*(j + 1)*np.pi/5))*(i + 1)/p])
j = 0
for i in range(2):
    Boundary_points_1.append([0.5-r_0 * np.sin(2*j*np.pi/5) + (r_0 * np.sin(2*j*np.pi/5) - r_0 * np.sin(2*(j + 1)*np.pi/5))*i/p, 0.5+r_0 * np.cos(2*j*np.pi/5) + (-r_0 * np.cos(2*j*np.pi/5) + r_0 * np.cos(2*(j + 1)*np.pi/5))*i/p])
Boundary_points_1 = np.array(Boundary_points_1)
plt.plot(Boundary_points_1[:,0], Boundary_points_1[:,1])
for j in range(5):
    for i in range(p):
        Boundary_points.append([0.5-r_0 * np.sin(2*j*np.pi/5) + (r_0 * np.sin(2*j*np.pi/5) - r_0 * np.sin(2*(j + 1)*np.pi/5))*i/p, 0.5+r_0 * np.cos(2*j*np.pi/5) + (-r_0 * np.cos(2*j*np.pi/5) + r_0 * np.cos(2*(j + 1)*np.pi/5))*i/p])
Boundary_points = np.array(Boundary_points)
def automatic_normal_vector(x):
    len_ = x.shape[0] - 2
    c = []
    for i in range(len_):
        if (i+1) % p == 0:
            continue
        a = x[i+2] - x[i]
        b = np.zeros(2)
        b[0] = a[1]/(a[1]**2 + a[0]**2)**(1/2)
        b[1] = -a[0]/(a[1]**2 + a[0]**2)**(1/2)
        c.append(b)
    
    a = x[len_+1] - x[len_-1]
    b = np.zeros(2)
    b[0] = a[1]/(a[1]**2 + a[0]**2)**(1/2)
    b[1] = -a[0]/(a[1]**2 + a[0]**2)**(1/2)
    c.append(b)
    
    return np.array(c)
normal_vector_1 = automatic_normal_vector(Boundary_points)
#plt.figure()
#plt.scatter(normal_vector_1[:,0], normal_vector_1[:,1])
#print(normal_vector_1.shape)
#print(normal_vector_1)

def set_seed(x):
    np.random.seed(x)
    torch.manual_seed(x)
    torch.cuda.manual_seed_all(x)
    torch.backends.cudnn.deterministic = True


rand_mag = 1.0
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.uniform_(m.weight, a = -rand_mag, b = rand_mag)
        nn.init.uniform_(m.bias, a = -rand_mag, b = rand_mag)
        #nn.init.normal_(m.weight, mean=0, std=1)
        #nn.init.normal_(m.bias, mean=0, std=1)


class RFM_rep(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers, M, x_max, x_min, t_max, t_min):
        super(RFM_rep, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = M
        self.hidden_layers  = hidden_layers
        self.x_max = x_max
        self.x_min = x_min
        self.t_max = t_max
        self.t_min = t_min
        self.M = M
        self.a = torch.tensor([2.0/(x_max - x_min),2.0/(t_max - t_min)]).to(device)
        self.x_0 = torch.tensor([(x_max + x_min)/2,(t_max + t_min)/2]).to(device)
        self.hidden_layer = nn.Sequential(nn.Linear(self.in_features, self.hidden_features, bias=True),nn.Tanh())
        #print([x_min,x_max],[t_min,t_max])

    def forward(self,x):
#        dx = (x[:,:,:1] - self.x_min) / (self.x_max - self.x_min)
#        dt = (x[:,:,1:] - self.t_min) / (self.t_max - self.t_min)
#        
#        dx0 = dx <= -1/4
#        dx1 = (dx <= 1/4)  & (dx > -1/4)
#        dx2 = (dx <= 3/4)  & (dx > 1/4)
#        dx3 = (dx <= 5/4)  & (dx > 3/4)
#        dx4 = dx > 5/4
#        
#        dt0 = dt <= -1/4
#        dt1 = (dt <= 1/4)  & (dt > -1/4)
#        dt2 = (dt <= 3/4)  & (dt > 1/4)
#        dt3 = (dt <= 5/4)  & (dt > 3/4)
#        dt4 = dt > 5/4
#        
#        yx0 = 0.0
#        yx1 = (1 + torch.sin(2*np.pi*dx) ) / 2
#        yx2 = 1.0
#        yx3 = (1 - torch.sin(2*np.pi*(dx-1)) ) / 2
#        yx4 = 0.0
#        
#        yt0 = 0.0
#        yt1 = (1 + torch.sin(2*np.pi*dt) ) / 2
#        yt2 = 1.0
#        yt3 = (1 - torch.sin(2*np.pi*(dt-1)) ) / 2
#        yt4 = 0.0
        
        
        y = self.a * (x - self.x_0)
        y = self.hidden_layer(y)
        
        dx = (x[:,:,:1] - (self.x_min + self.x_max)/2) / ((self.x_max - self.x_min)/2)
        dt = (x[:,:,1:] - (self.t_min + self.t_max)/2) / ((self.t_max - self.t_min)/2)
        
        dx0 = dx <= -5/4
        dx1 = (dx <= -3/4)  & (dx > -5/4)
        dx2 = (dx <= 3/4)  & (dx > -3/4)
        dx3 = (dx <= 5/4)  & (dx > 3/4)
        dx4 = dx > 5/4
        
        dt0 = dt <= -5/4
        dt1 = (dt <= -3/4)  & (dt > -5/4)
        dt2 = (dt <= 3/4)  & (dt > -3/4)
        dt3 = (dt <= 5/4)  & (dt > 3/4)
        dt4 = dt > 5/4
        
        yx0 = 0.0
        yx1 = (1 + torch.sin(2*np.pi*dx) ) / 2
        yx2 = 1.0
        yx3 = (1 - torch.sin(2*np.pi*dx) ) / 2
        yx4 = 0.0
        
        yt0 = 0.0
        yt1 = (1 + torch.sin(2*np.pi*dt) ) / 2
        yt2 = 1.0
        yt3 = (1 - torch.sin(2*np.pi*dt) ) / 2
        yt4 = 0.0
        
        #print(y.shape,dx0.shape,yx0)
        if self.x_min == 0:
            y = y*(dx0*yx0+(dx1+dx2)*yx2+dx3*yx3+dx4*yx4)
        elif self.x_max == L:
            y = y*(dx0*yx0+dx1*yx1+(dx2+dx3)*yx2+dx4*yx4)
        else:
            y = y*(dx0*yx0+dx1*yx1+dx2*yx2+dx3*yx3+dx4*yx4)
        
        if self.t_min == 0:
            y = y*(dt0*yt0+(dt1+dt2)*yt2+dt3*yt3+dt4*yt4)
        elif self.t_max == tf:
            y = y*(dt0*yt0+dt1*yt1+(dt2+dt3)*yt2+dt4*yt4)
        else:
            y = y*(dt0*yt0+dt1*yt1+dt2*yt2+dt3*yt3+dt4*yt4)
        
        return y

#一些预置参数
L = 1.0
tf = 1.0

epsilon = 0
k_1 = 10
#k_1 = 1
lamb = k_1**2
l = 1
n = 1
delta = 0.0001*1/k_1
robin_parameter = 1 * k_1
#解函数实部

def anal_u(x,y):
    x = x + l
    y = y + l
    h1 = sps.hankel1(n, k_1*(x**2 + y**2)**(1/2)) 
    u = np.real(h1) * np.cos(n * np.arctan2(y,x)) -  np.imag(h1) * np.sin(n * np.arctan2(y,x))
    return u

def anal_dudx(x,y):
    x = x + l
    y = y + l
    h1 = np.cos(np.arctan2(y,x)) * (n * sps.hankel1(n, k_1*(x**2 + y**2)**(1/2))/(x**2 + y**2)**(1/2) - k_1 * sps.hankel1(n + 1, k_1*(x**2 + y**2)**(1/2))) - np.array(complex(0,1)) * n /(x**2 + y**2)**(1/2) * np.sin(np.arctan2(y,x))*sps.hankel1(n, k_1*(x**2 + y**2)**(1/2))
    h2 = np.sin(np.arctan2(y,x)) * (n * sps.hankel1(n, k_1*(x**2 + y**2)**(1/2))/(x**2 + y**2)**(1/2) - k_1 * sps.hankel1(n + 1, k_1*(x**2 + y**2)**(1/2))) + np.array(complex(0,1)) * n /(x**2 + y**2)**(1/2) * np.cos(np.arctan2(y,x))*sps.hankel1(n, k_1*(x**2 + y**2)**(1/2))
    dudx = np.real(h1) * np.cos(n * np.arctan2(y,x)) -  np.imag(h1) * np.sin(n * np.arctan2(y,x))
    dudy = np.real(h2) * np.cos(n * np.arctan2(y,x)) -  np.imag(h2) * np.sin(n * np.arctan2(y,x))
    return np.concatenate((np.expand_dims(dudx, axis = 1), np.expand_dims(dudy, axis = 1)),axis = 1)

def anal_d2udx2_plus_d2udy2(x,y):
    x = x + l
    y = y + l
    #print(np.array(complex(np.cos(n * np.arctan2(y,x)), np.sin(n * np.arctan2(y,x)))))
    u = np.real(k_1**2 * (sps.hankel1(n, k_1*(x**2 + y**2)**(1/2)) + sps.hankel1(n+2, k_1*(x**2 + y**2)**(1/2))) - 2*k_1*(n+1)/((x**2 + y**2)**(1/2)) * sps.hankel1(n+1, k_1*(x**2 + y**2)**(1/2))) * np.cos(n * np.arctan2(y,x)) \
        - np.imag(k_1**2 * (sps.hankel1(n, k_1*(x**2 + y**2)**(1/2)) + sps.hankel1(n+2, k_1*(x**2 + y**2)**(1/2))) - 2*k_1*(n+1)/((x**2 + y**2)**(1/2)) * sps.hankel1(n+1, k_1*(x**2 + y**2)**(1/2))) *  np.sin(n * np.arctan2(y,x))
    return u
'''
def anal_d2udx2_plus_d2udy2(x,y):
    u = (anal_u(x + 2 * delta, y) - 16 * anal_u(x + delta, y) + 30 * anal_u(x, y) - 16 * anal_u(x - delta, y) + anal_u(x - 2*delta, y))/(12*delta**2) + (anal_u(x, y + 2 * delta) - 16 * anal_u(x, y + delta) + 30 * anal_u(x, y) - 16 * anal_u(x, y - delta) + anal_u(x, y - 2 * delta))/(12*delta**2)  
    return k_1**2 * anal_u(x,y) + u
'''

def anal_v(x,y):
    x = x + l
    y = y + l
    h1 = sps.hankel1(n, k_1*(x**2 + y**2)**(1/2))
    u = np.real(h1) * np.sin(n * np.arctan2(y,x)) +  np.imag(h1) * np.cos(n * np.arctan2(y,x))
    return u

#精确解
def anal_dvdx(x,y):
    x = x + l
    y = y + l
    h1 = np.cos(np.arctan2(y,x)) * (n * sps.hankel1(n, k_1*(x**2 + y**2)**(1/2))/(x**2 + y**2)**(1/2) - k_1 * sps.hankel1(n + 1, k_1*(x**2 + y**2)**(1/2))) - np.array(complex(0,1)) * n /(x**2 + y**2)**(1/2) * np.sin(np.arctan2(y,x))*sps.hankel1(n, k_1*(x**2 + y**2)**(1/2))
    h2 = np.sin(np.arctan2(y,x)) * (n * sps.hankel1(n, k_1*(x**2 + y**2)**(1/2))/(x**2 + y**2)**(1/2) - k_1 * sps.hankel1(n + 1, k_1*(x**2 + y**2)**(1/2))) + np.array(complex(0,1)) * n /(x**2 + y**2)**(1/2) * np.cos(np.arctan2(y,x))*sps.hankel1(n, k_1*(x**2 + y**2)**(1/2))
    dudx = np.real(h1) * np.sin(n * np.arctan2(y,x)) +  np.imag(h1) * np.cos(n * np.arctan2(y,x))
    dudy = np.real(h2) * np.sin(n * np.arctan2(y,x)) +  np.imag(h2) * np.cos(n * np.arctan2(y,x))
    return np.concatenate((np.expand_dims(dudx, axis = 1), np.expand_dims(dudy, axis = 1)),axis = 1)

def anal_d2vdx2_plus_d2vdy2(x,y):
    x = x + l
    y = y + l
    u = np.real(k_1**2 * (sps.hankel1(n, k_1*(x**2 + y**2)**(1/2)) + sps.hankel1(n+2, k_1*(x**2 + y**2)**(1/2))) - 2*k_1*(n+1)/((x**2 + y**2)**(1/2)) * sps.hankel1(n+1, k_1*(x**2 + y**2)**(1/2))) * np.sin(n * np.arctan2(y,x)) \
        + np.imag(k_1**2 * (sps.hankel1(n, k_1*(x**2 + y**2)**(1/2)) + sps.hankel1(n+2, k_1*(x**2 + y**2)**(1/2))) - 2*k_1*(n+1)/((x**2 + y**2)**(1/2)) * sps.hankel1(n+1, k_1*(x**2 + y**2)**(1/2))) *  np.cos(n * np.arctan2(y,x))
    return u

vanal_u = anal_u
vanal_f = anal_d2udx2_plus_d2udy2

vanal_v = anal_v
vanal_g = anal_d2vdx2_plus_d2vdy2

#先共用一套基函数
def pre_define(Nx,Ny,M,Qx,Qy):#x负责分块，y负责数据
    models_u = []
    points = []
    for k in range(Nx):
        model_for_x = []
        point_for_x = []
        x_min = L/Nx * k
        x_max = L/Nx * (k+1)
        x_devide = np.linspace(x_min, x_max, Qx + 1)
        for n in range(Ny):
            t_min = tf/Ny * n
            t_max = tf/Ny * (n+1)
            model = RFM_rep(in_features = 2, out_features = 1, hidden_layers = 1, M = M, x_min = x_min, 
                              x_max = x_max, t_min = t_min, t_max = t_max).to(device)
            model = model.apply(weights_init)
            model = model.double()
            for param in model.parameters():
                param.requires_grad = False
            model_for_x.append(model)
            t_devide = np.linspace(t_min, t_max, Qy + 1)
            grid = np.array(list(itertools.product(x_devide,t_devide))).reshape(Qx+1,Qy+1,2)
            point_for_x.append(torch.tensor(grid,requires_grad=True).to(device))
        models_u.append(model_for_x)
        points.append(point_for_x)
    return(models_u, points)

def cal_matrix(models_u,models_v,points,Nx,Ny,M,Qx,Qy):
# matrix define (Aw=b)
    #实虚分开
    A_1 = np.zeros([Nx*Ny*Qx*Qy,Nx*Ny*M]) # u_t - c*u_x = 0
    B_1 = np.zeros([Nx*Ny*Qx*Qy,Nx*Ny*M])
    C_1 = np.zeros([Nx*Ny*Qx*Qy,Nx*Ny*M])
    f_1 = np.zeros([Nx*Ny*Qx*Qy,1])
    g_1 = np.zeros([Nx*Ny*Qx*Qy,1])

    A_2 = np.zeros([Nx*Ny*Boundary_len,Nx*Ny*M]) # u(x,0) = h(x)
    B_2 = np.zeros([Nx*Ny*Boundary_len,Nx*Ny*M])
    f_2 = np.zeros([Nx*Ny*Boundary_len,1])
    g_2 = np.zeros([Nx*Ny*Boundary_len,1])

    A_3 = np.zeros([Nx*Ny*Boundary_len,Nx*Ny*M]) # u(x,0) = h(x)
    B_3 = np.zeros([Nx*Ny*Boundary_len,Nx*Ny*M])

    A_4 = np.zeros([Nx*Ny*Boundary_len,Nx*Ny*M]) # u(x,0) = h(x)
    B_4 = np.zeros([Nx*Ny*Boundary_len,Nx*Ny*M])
    f_3 = np.zeros([Nx*Ny*Boundary_len,1])
    g_3 = np.zeros([Nx*Ny*Boundary_len,1])

    A_5 = np.zeros([Nx*Ny*Boundary_len,Nx*Ny*M]) # u(x,0) = h(x)
    B_5 = np.zeros([Nx*Ny*Boundary_len,Nx*Ny*M])

    #np.set_printoptions(threshold=np.inf)
    #r_bound = np.expand_dims(np.sqrt((Boundary_points_1[:Boundary_len,0] + l)**2 + (Boundary_points_1[:Boundary_len,1] + l)**2), axis = 1)
    #x_bound = np.expand_dims((Boundary_points_1[:Boundary_len,0] + l), axis = 1)
    #y_bound = np.expand_dims((Boundary_points_1[:Boundary_len,1] + l), axis = 1)
    #normal_vector_1 = np.concatenate((x_bound/r_bound, y_bound/r_bound), axis = 1)
    #r_bound = 5
    for k in range(Nx):
        for n in range(Ny):
            print(k,n)
            # u_t - c*u_x = 0
            in_ = points[k][n].cpu().detach().numpy()
            #print(in_[:Qx,0,0])
            #out = models[k][n](points[k][n])
            #values = out.cpu().detach().numpy()
            #f_2[(k*Ny + n)*Boundary_len : (k*Ny + n)*Boundary_len+Boundary_len,:] = vanal_u(Boundary_points_1[:,0],Boundary_points_1[:,1])[:Boundary_len_k].reshape((-1,1))
            #g_2[(k*Ny + n)*Boundary_len : (k*Ny + n)*Boundary_len+Boundary_len,:] = vanal_v(Boundary_points_1[:,0],Boundary_points_1[:,1])[:Boundary_len_k].reshape((-1,1))
            f_1[(k*Ny + n)*Qx*Qy : (k*Ny + n + 1)*Qx*Qy,:] = anal_d2udx2_plus_d2udy2(in_[:Qx,:Qy,0], in_[:Qx,:Qy,1]).reshape((-1,1))
            g_1[(k*Ny + n)*Qx*Qy : (k*Ny + n + 1)*Qx*Qy,:] = anal_d2vdx2_plus_d2vdy2(in_[:Qx,:Qy,0], in_[:Qx,:Qy,1]).reshape((-1,1))
            #print(f_1)
            #stop
            #f_2[(k*Ny + n)*Boundary_len : (k*Ny + n)*Boundary_len+Boundary_len,:] = np.sum(anal_dudx_1(Boundary_points_1[:Boundary_len,0],Boundary_points_1[:Boundary_len,1]) * normal_vector_1, axis = 1).reshape(-1,1) - robin_parameter * vanal_v(Boundary_points_1[:,0],Boundary_points_1[:,1])[:Boundary_len].reshape((-1,1))
            #g_2[(k*Ny + n)*Boundary_len : (k*Ny + n)*Boundary_len+Boundary_len,:] = np.sum(anal_dvdx_1(Boundary_points_1[:Boundary_len,0],Boundary_points_1[:Boundary_len,1]) * normal_vector_1, axis = 1).reshape(-1,1) + robin_parameter * vanal_u(Boundary_points_1[:,0],Boundary_points_1[:,1])[:Boundary_len].reshape((-1,1))
            #print(f_2)
            f_2[(k*Ny + n)*Boundary_len : (k*Ny + n)*Boundary_len+Boundary_len,:] = np.sum(anal_dudx(Boundary_points_1[:Boundary_len,0],Boundary_points_1[:Boundary_len,1]) * normal_vector_1, axis = 1).reshape(-1,1) - robin_parameter * vanal_v(Boundary_points_1[:,0],Boundary_points_1[:,1])[:Boundary_len].reshape((-1,1))
            g_2[(k*Ny + n)*Boundary_len : (k*Ny + n)*Boundary_len+Boundary_len,:] = np.sum(anal_dvdx(Boundary_points_1[:Boundary_len,0],Boundary_points_1[:Boundary_len,1]) * normal_vector_1, axis = 1).reshape(-1,1) + robin_parameter * vanal_u(Boundary_points_1[:,0],Boundary_points_1[:,1])[:Boundary_len].reshape((-1,1))
            #print(f_2)
            #stop
            for k_ in range(Nx):
                for n_ in range(Ny):
                    # u_t - c*u_x = 0
                    out = models_u[k_][n_](points[k][n])
                    #Boundary_1
                    Boundary_point_1_1 = torch.tensor(Boundary_points_1,requires_grad=True).to(device).unsqueeze(dim = 1)
                    out_boundarys_1 = models_u[k_][n_](Boundary_point_1_1)
                    out_boundary_1 = out_boundarys_1.cpu().detach().numpy()
                    values = out.cpu().detach().numpy()
                    M_begin = k_*Ny*M + n_*M
                    #print(values.shape)
                    grads = []
                    g_boundary_grads = []
                    g_boundary_grads_1 = []
                    grads_2_xx = []
                    grads_2_yy = []

                    for i in range(M):
                        g_1_1 = torch.autograd.grad(outputs=out[:,:,i], inputs=points[k][n],
                                              grad_outputs=torch.ones_like(out[:,:,i]),
                                              create_graph = True, retain_graph = True)[0]
                        grads.append(g_1_1.squeeze().cpu().detach().numpy())
                        g_boundary_x_1 = torch.autograd.grad(outputs=out_boundarys_1[:,:,i], inputs=Boundary_point_1_1,
                                          grad_outputs=torch.ones_like(out_boundarys_1[:,:,0]),
                                          create_graph = True, retain_graph = True)[0]
                        g_boundary_grads_1.append(g_boundary_x_1.squeeze().cpu().detach().numpy())
                        #print(len(g_boundary_grads))
                        g_2_x = torch.autograd.grad(outputs=g_1_1[:,:,0], inputs=points[k][n],
                                          grad_outputs=torch.ones_like(out[:,:,i]),
                                          create_graph = True, retain_graph = True)[0]
                        g_2_y = torch.autograd.grad(outputs=g_1_1[:,:,1], inputs=points[k][n],
                                          grad_outputs=torch.ones_like(out[:,:,i]),
                                          create_graph = True, retain_graph = True)[0]
                        grads_2_xx.append(g_2_x[:,:,0].squeeze().cpu().detach().numpy())
                        grads_2_yy.append(g_2_y[:,:,1].squeeze().cpu().detach().numpy())
                    grads = np.array(grads).swapaxes(0,3)
                    g_boundary_grads_1 = np.array(g_boundary_grads_1)[:,:Boundary_len,:].swapaxes(0,2)
                    normal_boundary_grads_1 = g_boundary_grads_1 * np.expand_dims(normal_vector_1.T, axis = 2)
                    grads_x = grads[0,:,:,:]
                    grads_y = grads[1,:,:,:]
                    #print(grads_x.shape)
                    value = values[:Qx,:Qy, :]
                    value = value.reshape(-1,M)
                    grads_2_xx = np.array(grads_2_xx)
                    grads_2_xx_1 = grads_2_xx.swapaxes(0,2).swapaxes(0,1)
                    #print(grads_2_xx.shape)
                    grads_2_xx = grads_2_xx[:,:Qx,:Qy]
                    #print(grads_2_xx.shape)
                    #print(grads_2_xx.transpose(1,2,0).shape)
                    grads_2_xx = grads_2_xx.transpose(1,2,0).reshape(-1,M)
                    #print(grads_2_xx.shape)
                    grads_2_yy = np.array(grads_2_yy)
                    grads_2_yy_1 = grads_2_yy.swapaxes(0,2).swapaxes(0,1)
                    grads_2_yy = grads_2_yy[:,:Qx,:Qy]
                    grads_2_yy = grads_2_yy.transpose(1,2,0).reshape(-1,M)
                    #values
                    A_1[k*Ny*Qx*Qy + n*Qx*Qy : k*Ny*Qx*Qy + n*Qx*Qy + Qx*Qy, M_begin : M_begin + M] = grads_2_xx + grads_2_yy + lamb * value
                    #A_2[(k*Ny + n)*Boundary_len : (k*Ny + n +1)*Boundary_len, M_begin : M_begin + M] = out_boundary_1[:Boundary_len,0,:]
                    A_2[(k*Ny + n)*Boundary_len : (k*Ny + n +1)*Boundary_len, M_begin : M_begin + M] = np.sum(normal_boundary_grads_1 ,axis = 0) 
                    A_3[(k*Ny + n)*Boundary_len : (k*Ny + n +1)*Boundary_len, M_begin : M_begin + M] = robin_parameter * out_boundary_1[:Boundary_len,0,:]
            for k_ in range(Nx):
                for n_ in range(Ny):
                    # u_t - c*u_x = 0
                    out = models_v[k_][n_](points[k][n])
                    #r_bound = np.sqrt((Boundary_points_1[:Boundary_len,0] + l)**2 + (Boundary_points_1[:Boundary_len,1] + l)**2)
                    #Boundary_1
                    Boundary_point_1_1 = torch.tensor(Boundary_points_1,requires_grad=True).to(device).unsqueeze(dim = 1)
                    out_boundarys_1 = models_v[k_][n_](Boundary_point_1_1)
                    out_boundary_1 = out_boundarys_1.cpu().detach().numpy()
                    values = out.cpu().detach().numpy()
                    M_begin = k_*Ny*M + n_*M
                    #print(values.shape)
                    grads = []
                    g_boundary_grads = []
                    g_boundary_grads_1 = []
                    grads_2_xx = []
                    grads_2_yy = []
                    for i in range(M):
                        g_1_1 = torch.autograd.grad(outputs=out[:,:,i], inputs=points[k][n],
                                              grad_outputs=torch.ones_like(out[:,:,i]),
                                              create_graph = True, retain_graph = True)[0]
                        grads.append(g_1_1.squeeze().cpu().detach().numpy())
                        g_boundary_x_1 = torch.autograd.grad(outputs=out_boundarys_1[:,:,i], inputs=Boundary_point_1_1,
                                          grad_outputs=torch.ones_like(out_boundarys_1[:,:,0]),
                                          create_graph = True, retain_graph = True)[0]
                        g_boundary_grads_1.append(g_boundary_x_1.squeeze().cpu().detach().numpy())
                        #print(len(g_boundary_grads))
                        g_2_x = torch.autograd.grad(outputs=g_1_1[:,:,0], inputs=points[k][n],
                                          grad_outputs=torch.ones_like(out[:,:,i]),
                                          create_graph = True, retain_graph = True)[0]
                        g_2_y = torch.autograd.grad(outputs=g_1_1[:,:,1], inputs=points[k][n],
                                          grad_outputs=torch.ones_like(out[:,:,i]),
                                          create_graph = True, retain_graph = True)[0]
                        grads_2_xx.append(g_2_x[:,:,0].squeeze().cpu().detach().numpy())
                        grads_2_yy.append(g_2_y[:,:,1].squeeze().cpu().detach().numpy())
                    grads = np.array(grads).swapaxes(0,3)
                    g_boundary_grads_1 = np.array(g_boundary_grads_1)[:,:Boundary_len,:].swapaxes(0,2)
                    normal_boundary_grads_1 = g_boundary_grads_1 * np.expand_dims(normal_vector_1.T, axis = 2)
                    grads_x = grads[0,:,:,:]
                    grads_y = grads[1,:,:,:]
                    #print(grads_x.shape)
                    value = values[:Qx,:Qy, :]
                    value = value.reshape(-1,M)
                    #print(values.shape)
                    #print(values.shape,grads.shape)
                    #grads = grads[:,:Qx,:Qy,:].reshape(M,-1,2)
                    grads_2_xx = np.array(grads_2_xx)
                    grads_2_xx_1 = grads_2_xx.swapaxes(0,2).swapaxes(0,1)
                    #print(grads_2_xx.shape)
                    grads_2_xx = grads_2_xx[:,:Qx,:Qy]
                    #print(grads_2_xx.shape)
                    #print(grads_2_xx.transpose(1,2,0).shape)
                    grads_2_xx = grads_2_xx.transpose(1,2,0).reshape(-1,M)
                    #print(grads_2_xx.shape)
                    grads_2_yy = np.array(grads_2_yy)
                    grads_2_yy_1 = grads_2_yy.swapaxes(0,2).swapaxes(0,1)
                    grads_2_yy = grads_2_yy[:,:Qx,:Qy]
                    grads_2_yy = grads_2_yy.transpose(1,2,0).reshape(-1,M)
                    #values
                    B_1[k*Ny*Qx*Qy + n*Qx*Qy : k*Ny*Qx*Qy + n*Qx*Qy + Qx*Qy, M_begin : M_begin + M] = grads_2_xx + grads_2_yy + lamb * value
                    #B_3[(k*Ny + n)*Boundary_len : (k*Ny + n +1)*Boundary_len, M_begin : M_begin + M] = out_boundary_1[:Boundary_len,0,:]
                    B_2[(k*Ny + n)*Boundary_len : (k*Ny + n +1)*Boundary_len, M_begin : M_begin + M] = -robin_parameter * out_boundary_1[:Boundary_len,0,:]
                    B_3[(k*Ny + n)*Boundary_len : (k*Ny + n +1)*Boundary_len, M_begin : M_begin + M] = np.sum(normal_boundary_grads_1 ,axis = 0)  
    D_1 = np.concatenate((A_1,C_1),axis = 1) 
    D_2 = np.concatenate((C_1,B_1),axis = 1)
    D_3 = np.concatenate((A_2,B_2),axis = 1)
    D_4 = np.concatenate((A_3,B_3),axis = 1)
    A = np.concatenate((D_1,D_2,D_3,D_4),axis = 0)
    f = np.concatenate((f_1,g_1,f_2,g_2),axis = 0)
    #A = np.concatenate((D_1,D_2,D_3,D_4,D_5,D_6,D_7,D_8,D_9,D_10,D_11,D_12),axis = 0)
    #f =  np.concatenate((f_1,f_2,f_3,f_4,f_5,g_1,g_2,g_3,g_4,g_5,f_6,g_6),axis = 0)
    #A = np.concatenate((D_1,D_6,D_11,D_12),axis = 0)
    #f =  np.concatenate((f_1,g_1,f_6,g_6),axis = 0)
    #A = np.concatenate((D_1,D_2,D_3,D_4,D_5,D_6,D_7,D_8,D_9,D_10),axis = 0)
    #f =  np.concatenate((f_1,f_2,f_3,f_4,f_5,g_1,g_2,g_3,g_4,g_5),axis = 0)                            
    #A = np.concatenate((B_1,B_2,B_3,B_4,B_5),axis = 0)
    #f =  np.concatenate((g_1,g_2,g_3,g_4,g_5),axis = 0)
    print(f.shape)
    return(A,f)
'''
def grid_cir(x):
    a = np.ones((x.shape[0],x.shape[1]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if (x[i,j,1]-0.5) ** 2 + (x[i,j,0]-0.5) ** 2 <= 0.04 or (x[i,j,1]-0.5) ** 2 + (x[i,j,0]-0.5) ** 2 >= 0.25:
                a[i,j] = 0.0
    return a
'''
class Line:
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
 
def intersection(l1, l2):
    #快速排斥实验
    if (max(l1.x1, l1.x2) < min(l2.x1, l2.x2) or
        max(l1.y1, l1.y2) < min(l2.y1, l2.y2) or
        max(l2.x1, l2.x2) < min(l1.x1, l1.x2) or
        max(l2.y1, l2.y2) < min(l1.y1, l1.y2)):
        return False
    #跨立实验
    if (((l1.x1-l2.x1)*(l2.y2-l2.y1)-(l1.y1-l2.y1)*(l2.x2-l2.x1))*((l1.x2-l2.x1)*(l2.y2-l2.y1)-(l1.y2-l2.y1)*(l2.x2-l2.x1)) > 0 or
        ((l2.x1-l1.x1)*(l1.y2-l1.y1)-(l2.y1-l1.y1)*(l1.x2-l1.x1))*((l2.x2-l1.x1)*(l1.y2-l1.y1)-(l2.y2-l1.y1)*(l1.x2-l1.x1)) > 0):
        return False
    return True

def grid_cir(M):
    # 生成二维网格
    X = M[:,:,0]
    Y = M[:,:,1]
    grid = np.ones(X.shape)

    # 构造正五边形
    radius = 0.35*np.sqrt(2)
    outer_angle = np.pi * 2 / 5
    inner_angle = np.pi / 5
    cx, cy = 0.5, 0.5
    vertices = []
    for i in range(5):
        angle = i * outer_angle + np.pi/10
        vertices.append((cx + radius * np.cos(angle), cy + radius * np.sin(angle)))
    # 判断网格是否在五边形内
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            for k in range(5):
                p1 = vertices[k]
                p2 = vertices[(k+1)%5]
                l1 = Line(p1[0],p1[1],p2[0],p2[1])
                l2 = Line(cx,cy,X[i][j],Y[i][j])
                if intersection(l1, l2):
                    grid[i][j] = 0
    return grid

def test(models,Nx,Ny,M,Qx,Qy,w,plot = False):
    epsilon = []
    true_values = []
    numerical_values = []
    test_Qx = int(300/Nx)
    test_Qy = int(300/Ny)
    L = 1.0
    tf = 1.0
    x_devide = np.linspace(0, L, 300 + 1)[:300]
    t_devide = np.linspace(0, tf, 300 + 1)[:300]
    circle = np.array(list(itertools.product(x_devide,t_devide))).reshape(300,300,2)
    grid_circle = grid_cir(circle)
    for k in range(Nx):
        epsilon_x = []
        true_value_x = []
        numerical_value_x = []
        for n in range(Ny):
            numerical_value = None
            # forward and grad
            x_min = L/Nx * k
            x_max = L/Nx * (k+1)
            x_devide = np.linspace(x_min, x_max, test_Qx + 1)[:test_Qx]
            t_min = tf/Ny * n
            t_max = tf/Ny * (n+1)
            t_devide = np.linspace(t_min, t_max, test_Qy + 1)[:test_Qy]
            grid = np.array(list(itertools.product(x_devide,t_devide))).reshape(test_Qx,test_Qy,2)
            test_point = torch.tensor(grid,requires_grad=True).to(device)
            in_ = test_point.cpu().detach().numpy()
            true_value = anal_u(in_[:,:,0],in_[:,:,1])
            for k_ in range(Nx):
                for n_ in range(Ny):
                        out = models[k_][n_](test_point)
                        values = out.cpu().detach().numpy()
                        if numerical_value is None:
                            numerical_value = np.dot(values, w[k_*Ny*M + n_*M : k_*Ny*M + n_*M + M,:]).reshape(test_Qx,test_Qy)
                        else:
                            numerical_value = numerical_value + np.dot(values, w[k_*Ny*M + n_*M : k_*Ny*M + n_*M + M,:]).reshape(test_Qx,test_Qy)
            e = np.abs(true_value - numerical_value)
            true_value_x.append(true_value)
            numerical_value_x.append(numerical_value)
            epsilon_x.append(e)
        epsilon_x = np.concatenate(epsilon_x, axis=1)
        epsilon.append(epsilon_x)
        true_value_x = np.concatenate(true_value_x, axis=1)
        true_values.append(true_value_x)
        numerical_value_x = np.concatenate(numerical_value_x, axis=1)
        numerical_values.append(numerical_value_x)
    epsilon = np.concatenate(epsilon, axis=0)
    epsilon = epsilon * grid_circle
    true_values = np.concatenate(true_values, axis=0)
    numerical_values = np.concatenate(numerical_values, axis=0)
    e = epsilon.reshape((-1,1))
    print('********************* ERROR *********************')
    print('Nx=%s,Ny=%s,M=%s,Qx=%s,Qy=%s'%(Nx,Ny,M,Qx,Qy))
    print('L_inf=',e.max(),'L_2=',math.sqrt(sum(e*e)/len(e)))
    print("边值条件误差")
    print(max(epsilon[0,:]),max(epsilon[-1,:]),max(epsilon[:,0]),max(epsilon[:,-1]))
    #np.save('./epsilon_psi2.npy',epsilon)
    if True:
        L,tf=1,1
        x = np.linspace(0, L, 301)[:300]
        y = np.linspace(0, tf, 301)[:300]
        x,y = np.meshgrid(x,y)
        plt.figure(figsize=[12, 10])
        plt.axis('equal')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.tick_params(labelsize=15)
        font2 = {
        'weight' : 'normal',
        'size'   : 20,
        }
        plt.xlabel('x',font2)
        plt.ylabel('y',font2)
        plt.pcolor(x,y,epsilon.T, cmap='jet')
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=15)
        cbar.ax.yaxis.get_offset_text().set_fontsize(15)
        cbar.update_ticks()

        plt.figure(figsize=[12, 10])
        plt.axis('equal')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.tick_params(labelsize=15)
        font2 = {
        'weight' : 'normal',
        'size'   : 20,
        }
        plt.xlabel('x',font2)
        plt.ylabel('y',font2)
        plt.pcolor(x,y,true_values.T, cmap='jet')
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=15)
        cbar.ax.yaxis.get_offset_text().set_fontsize(15)
        cbar.update_ticks()

    return(e.max(),math.sqrt(sum(e*e)/len(e)))

def test1(models,Nx,Ny,M,Qx,Qy,w,plot = False):
    epsilon = []
    true_values = []
    numerical_values = []
    test_Qx = int(300/Nx)
    test_Qy = int(300/Ny)
    L = 1.0
    tf = 1.0
    x_devide = np.linspace(0, L, 300 + 1)[:300]
    t_devide = np.linspace(0, tf, 300 + 1)[:300]
    circle = np.array(list(itertools.product(x_devide,t_devide))).reshape(300,300,2)
    grid_circle = grid_cir(circle)
    for k in range(Nx):
        epsilon_x = []
        true_value_x = []
        numerical_value_x = []
        for n in range(Ny):
            numerical_value = None
            # forward and grad
            x_min = L/Nx * k
            x_max = L/Nx * (k+1)
            x_devide = np.linspace(x_min, x_max, test_Qx + 1)[:test_Qx]
            t_min = tf/Ny * n
            t_max = tf/Ny * (n+1)
            t_devide = np.linspace(t_min, t_max, test_Qy + 1)[:test_Qy]
            grid = np.array(list(itertools.product(x_devide,t_devide))).reshape(test_Qx,test_Qy,2)
            test_point = torch.tensor(grid,requires_grad=True).to(device)
            in_ = test_point.cpu().detach().numpy()
            true_value = anal_v(in_[:,:,0],in_[:,:,1])
            for k_ in range(Nx):
                for n_ in range(Ny):
                        out = models[k_][n_](test_point)
                        values = out.cpu().detach().numpy()
                        if numerical_value is None:
                            numerical_value = np.dot(values, w[k_*Ny*M + n_*M : k_*Ny*M + n_*M + M,:]).reshape(test_Qx,test_Qy)
                        else:
                            numerical_value = numerical_value + np.dot(values, w[k_*Ny*M + n_*M : k_*Ny*M + n_*M + M,:]).reshape(test_Qx,test_Qy)
            e = np.abs(true_value - numerical_value)
            true_value_x.append(true_value)
            numerical_value_x.append(numerical_value)
            epsilon_x.append(e)
        epsilon_x = np.concatenate(epsilon_x, axis=1)
        epsilon.append(epsilon_x)
        true_value_x = np.concatenate(true_value_x, axis=1)
        true_values.append(true_value_x)
        numerical_value_x = np.concatenate(numerical_value_x, axis=1)
        numerical_values.append(numerical_value_x)
    epsilon = np.concatenate(epsilon, axis=0)
    epsilon = epsilon * grid_circle
    true_values = np.concatenate(true_values, axis=0)
    numerical_values = np.concatenate(numerical_values, axis=0)
    e = epsilon.reshape((-1,1))
    print('********************* ERROR *********************')
    print('Nx=%s,Ny=%s,M=%s,Qx=%s,Qy=%s'%(Nx,Ny,M,Qx,Qy))
    print('L_inf=',e.max(),'L_2=',math.sqrt(sum(e*e)/len(e)))
    print("边值条件误差")
    print(max(epsilon[0,:]),max(epsilon[-1,:]),max(epsilon[:,0]),max(epsilon[:,-1]))
    #np.save('./epsilon_psi2.npy',epsilon)
    if True:
        L,tf=1,1
        x = np.linspace(0, L, 301)[:300]
        y = np.linspace(0, tf, 301)[:300]
        x,y = np.meshgrid(x,y)
        plt.figure(figsize=[12, 10])
        plt.axis('equal')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.tick_params(labelsize=15)
        font2 = {
        'weight' : 'normal',
        'size'   : 20,
        }
        plt.xlabel('x',font2)
        plt.ylabel('y',font2)
        plt.pcolor(x,y,epsilon.T, cmap='jet')
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=15)
        cbar.ax.yaxis.get_offset_text().set_fontsize(15)
        cbar.update_ticks()

        plt.figure(figsize=[12, 10])
        plt.axis('equal')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.tick_params(labelsize=15)
        font2 = {
        'weight' : 'normal',
        'size'   : 20,
        }
        plt.xlabel('x',font2)
        plt.ylabel('y',font2)
        plt.pcolor(x,y,true_values.T, cmap='jet')
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=15)
        cbar.ax.yaxis.get_offset_text().set_fontsize(15)
        cbar.update_ticks()
    return(e.max(),math.sqrt(sum(e*e)/len(e)))

def main(Nx,Ny,M,Qx,Qy,plot = False,moore = False):
    # prepare models and collocation pointss
    models_u,points = pre_define(Nx,Ny,M,Qx,Qy)
    set_seed(0)
    models_v,points = pre_define(Nx,Ny,M,Qx,Qy)
    
    # matrix define (Aw=b)
    A,f = cal_matrix(models_u, models_v,points,Nx,Ny,M,Qx,Qy)
    
    max_value = 10.0
    for i in range(len(A)):
        if np.abs(A[i,:].max())==0:
            print("error line : ",i)
            continue
        ratio = max_value/np.abs(A[i,:]).max()
        A[i,:] = A[i,:]*ratio
        f[i] = f[i]*ratio
    length = A.shape[1]//2
    print(length)
    # solve
    # solve
    if moore:
        inv_coeff_mat = pinv(A)  # moore-penrose inverse, shape: (n_units,n_colloc+2)
        w = np.matmul(inv_coeff_mat, f)
    else:
        a = lstsq(A,f)
        w = a[0]
        #print(w)
        #print(w.shape)
        print(a[1])
    # test
    test(models_u,Nx,Ny,M,Qx,Qy,np.expand_dims(w[:length,0],1),plot)
    test1(models_v,Nx,Ny,M,Qx,Qy,np.expand_dims(w[length:,0],1),plot)    
    return 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    set_seed(100)
    result = []
    main(2,2,400,25,25,True)
    #main(2,2,100,25,25,True)
    #main(2,2,2000,80,80,True)