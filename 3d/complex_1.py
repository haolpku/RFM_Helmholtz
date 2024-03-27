import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import numpy as np
import torch
import torch.nn as nn
import math
from scipy.linalg import lstsq,pinv
from scipy.fftpack import fftshift,fftn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools
import scipy.special as sps
torch.set_default_dtype(torch.float64)

# 边界上的点数量
theta_points = 40
phi_points = 20

# 角度的取值范围
theta_values = np.linspace(0, 2. * np.pi, theta_points)
phi_values = np.linspace(0, np.pi, phi_points)

# 初始化边界点和单位法向量的列表
boundary_points = []
normal_vectors = []

# 对于每个θ和φ，计算边界点和单位法向量
for theta in theta_values:
    for phi in phi_values:
        # 计算r的值
        r = 0.4 + 0.1 * np.cos(5 * theta) * np.sin(5 * phi)

        # 计算边界点的笛卡尔坐标
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        boundary_points.append([x, y, z])

        # 计算r关于θ和φ的偏导数
        dr_dtheta = -np.sin(5 * theta) * np.sin(5 * phi)
        dr_dphi = np.cos(5 * phi) * np.cos(5 * theta)

        # 计算单位法向量
        normal_vector = np.array([-dr_dtheta * np.sin(phi) * np.cos(theta) - r * np.cos(phi) * np.cos(theta),
                                  -dr_dtheta * np.sin(phi) * np.sin(theta) - r * np.cos(phi) * np.sin(theta),
                                  -dr_dphi])
        normal_vector = normal_vector / np.linalg.norm(normal_vector)
        normal_vectors.append(normal_vector)

# 将列表转换为NumPy数组
Boundary_points_1 = np.array(boundary_points) + 0.5
normal_vector_1 = np.array(normal_vectors)
Boundary_len = Boundary_points_1.shape[0]

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
    def __init__(self, in_features, out_features, hidden_layers, M, x_max, x_min, y_max, y_min, z_max, z_min):
        super(RFM_rep, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = M
        self.hidden_layers  = hidden_layers
        self.x_max = x_max
        self.x_min = x_min
        self.y_max = y_max
        self.y_min = y_min
        self.z_max = z_max
        self.z_min = z_min
        self.M = M
        self.a = torch.tensor([2.0/(x_max - x_min),2.0/(y_max - y_min),2.0/(z_max - z_min)]).to(device)
        self.x_0 = torch.tensor([(x_max + x_min)/2,(y_max + y_min)/2,(z_max + z_min)/2]).to(device)
        self.hidden_layer = nn.Sequential(nn.Linear(self.in_features, self.hidden_features, bias=True),nn.Tanh())

    def forward(self,x):
        y = self.a * (x - self.x_0)
        y = self.hidden_layer(y)

        dx = (x[:,:,:,0].unsqueeze(dim = 3) - (self.x_min + self.x_max)/2) / ((self.x_max - self.x_min)/2)
        dy = (x[:,:,:,1].unsqueeze(dim = 3) - (self.y_min + self.y_max)/2) / ((self.y_max - self.y_min)/2)
        dz = (x[:,:,:,2].unsqueeze(dim = 3) - (self.z_min + self.z_max)/2) / ((self.z_max - self.z_min)/2)

        dx0 = dx <= -5/4
        dx1 = (dx <= -3/4)  & (dx > -5/4)
        dx2 = (dx <= 3/4)  & (dx > -3/4)
        dx3 = (dx <= 5/4)  & (dx > 3/4)
        dx4 = dx > 5/4

        dy0 = dy <= -5/4
        dy1 = (dy <= -3/4)  & (dy > -5/4)
        dy2 = (dy <= 3/4)  & (dy > -3/4)
        dy3 = (dy <= 5/4)  & (dy > 3/4)
        dy4 = dy > 5/4

        dz0 = dz <= -5/4
        dz1 = (dz <= -3/4)  & (dz > -5/4)
        dz2 = (dz <= 3/4)  & (dz > -3/4)
        dz3 = (dz <= 5/4)  & (dz > 3/4)
        dz4 = dz > 5/4

        yx0 = 0.0
        yx1 = (1 + torch.sin(2*np.pi*dx) ) / 2
        yx2 = 1.0
        yx3 = (1 - torch.sin(2*np.pi*dx) ) / 2
        yx4 = 0.0

        yy0 = 0.0
        yy1 = (1 + torch.sin(2*np.pi*dy) ) / 2
        yy2 = 1.0
        yy3 = (1 - torch.sin(2*np.pi*dy) ) / 2
        yy4 = 0.0

        yz0 = 0.0
        yz1 = (1 + torch.sin(2*np.pi*dz) ) / 2
        yz2 = 1.0
        yz3 = (1 - torch.sin(2*np.pi*dz) ) / 2
        yz4 = 0.0

        if self.x_min == 0:
            y = y*(dx0*yx0+(dx1+dx2)*yx2+dx3*yx3+dx4*yx4)
        elif self.x_max == X_edge:
            y = y*(dx0*yx0+dx1*yx1+(dx2+dx3)*yx2+dx4*yx4)
        else:
            y = y*(dx0*yx0+dx1*yx1+dx2*yx2+dx3*yx3+dx4*yx4)

        if self.y_min == 0:
            y = y*(dy0*yy0+(dy1+dy2)*yy2+dy3*yy3+dy4*yy4)
        elif self.y_max == Y_edge:
            y = y*(dy0*yy0+dy1*yy1+(dy2+dy3)*yy2+dy4*yy4)
        else:
            y = y*(dy0*yy0+dy1*yy1+dy2*yy2+dy3*yy3+dy4*yy4)

        if self.z_min == 0:
            y = y*(dz0*yz0+(dz1+dz2)*yz2+dz3*yz3+dz4*yz4)
        elif self.z_max == Z_edge:
            y = y*(dz0*yz0+dz1*yz1+(dz2+dz3)*yz2+dz4*yz4)
        else:
            y = y*(dz0*yz0+dz1*yz1+dz2*yz2+dz3*yz3+dz4*yz4)
        return y


#一些预置参数
X_edge = 1.0
Y_edge = 1.0
Z_edge = 1.0

A = 1.0
a = np.pi
b = 2*np.pi

B = 1.0
c = 2*np.pi
d = 4*np.pi
epsilon = 0
k_1 = 1
omega = k_1 * np.sqrt(2)
lamb = k_1**2
l_1 = 1
n = 0
r = 0.5
delta = 0.00001
robin_parameter = 1 * k_1
#解函数实部
def anal_u(x,y,z):
    x = x + l_1
    y = y + l_1
    z = z + l_1
    r = np.sqrt(x**2 + y**2 + z**2)
    u = np.cos(k_1 * r) / r
    return u

def anal_dudx(x,y,z):
    x = x + l_1
    y = y + l_1
    z = z + l_1
    r = np.sqrt(x**2 + y**2 + z**2)
    drdx = x/r
    drdy = y/r
    drdz = z/r
    dudr = -k_1 * np.sin(k_1 * r) / r - np.cos(k_1 * r) / r**2
    return np.concatenate((np.expand_dims(dudr * drdx, axis = 1), np.expand_dims(dudr * drdy, axis = 1), np.expand_dims(dudr * drdz, axis = 1)), axis = 1)

# 同理，对于 v 的函数也可以进行类似的计算
def anal_v(x,y,z):
    x = x + l_1
    y = y + l_1
    z = z + l_1
    r = np.sqrt(x**2 + y**2 + z**2)
    v = np.sin(k_1 * r) / r
    return v

def anal_dvdx(x,y,z):
    x = x + l_1
    y = y + l_1
    z = z + l_1
    r = np.sqrt(x**2 + y**2 + z**2)
    drdx = x/r
    drdy = y/r
    drdz = z/r
    dvdr = k_1 * np.cos(k_1 * r) / r - np.sin(k_1 * r) / r**2
    return np.concatenate((np.expand_dims(dvdr * drdx, axis = 1), np.expand_dims(dvdr * drdy, axis = 1), np.expand_dims(dvdr * drdz, axis = 1)), axis = 1)
vanal_u = anal_u
vanal_v = anal_v

#先共用一套基函数
def pre_define(Nx,Ny,Nz,M,Qx,Qy,Qz):
    models = []
    points = []
    for k in range(Nx):
        model_for_x = []
        point_for_x = []
        x_min = X_edge/Nx * k
        x_max = X_edge/Nx * (k+1)
        x_devide = np.linspace(x_min, x_max, Qx + 1)
        for n in range(Ny):
            y_min = Y_edge/Ny * n
            y_max = Y_edge/Ny * (n+1)
            y_devide = np.linspace(y_min, y_max, Qy + 1)
            model_for_x_y = []
            point_for_x_y = []
            for l in range(Nz):
                z_min = Z_edge/Nz * l
                z_max = Z_edge/Nz * (l+1)
                z_devide = np.linspace(z_min, z_max, Qz + 1)
                model = RFM_rep(in_features = 3, out_features = 1, hidden_layers = 1, M = M, x_min = x_min,
                              x_max = x_max, y_min = y_min, y_max = y_max, z_min = z_min, z_max = z_max).to(device)
                model = model.apply(weights_init)
                model = model.double()
                for param in model.parameters():
                    param.requires_grad = False
                model_for_x_y.append(model)
                grid = np.array(list(itertools.product(x_devide,y_devide,z_devide))).reshape(Qx+1,Qy+1,Qz+1,3)
                point_for_x_y.append(torch.tensor(grid,requires_grad=True).to(device))
            model_for_x.append(model_for_x_y)
            point_for_x.append(point_for_x_y)
        models.append(model_for_x)
        points.append(point_for_x)
    return(models,points)

def cal_matrix(models_u, models_v, points,Nx,Ny,Nz,M,Qx,Qy,Qz):
# matrix define (Aw=b)
    #实虚分开
    A_1 = np.zeros([Nx*Ny*Nz*Qx*Qy*Qz,Nx*Ny*Nz*M]) # u_t - c*u_x = 0
    B_1 = np.zeros([Nx*Ny*Nz*Qx*Qy*Qz,Nx*Ny*Nz*M])
    C_1 = np.zeros([Nx*Ny*Nz*Qx*Qy*Qz,Nx*Ny*Nz*M])
    f_1 = np.zeros([Nx*Ny*Nz*Qx*Qy*Qz,1])
    g_1 = np.zeros([Nx*Ny*Nz*Qx*Qy*Qz,1])

    A_2 = np.zeros([Nx*Ny*Nz*Boundary_len,Nx*Ny*Nz*M]) # u(x,0) = h(x)
    B_2 = np.zeros([Nx*Ny*Nz*Boundary_len,Nx*Ny*Nz*M])
    f_2 = np.zeros([Nx*Ny*Nz*Boundary_len,1])
    g_2 = np.zeros([Nx*Ny*Nz*Boundary_len,1])

    A_3 = np.zeros([Nx*Ny*Nz*Boundary_len,Nx*Ny*Nz*M]) # u(x,0) = h(x)
    B_3 = np.zeros([Nx*Ny*Nz*Boundary_len,Nx*Ny*Nz*M])

    A_4 = np.zeros([Nx*Ny*Nz*Boundary_len,Nx*Ny*Nz*M]) # u(x,0) = h(x)
    B_4 = np.zeros([Nx*Ny*Nz*Boundary_len,Nx*Ny*Nz*M])
    f_3 = np.zeros([Nx*Ny*Nz*Boundary_len,1])
    g_3 = np.zeros([Nx*Ny*Nz*Boundary_len,1])

    A_5 = np.zeros([Nx*Ny*Nz*Boundary_len,Nx*Ny*Nz*M]) # u(x,0) = h(x)
    B_5 = np.zeros([Nx*Ny*Nz*Boundary_len,Nx*Ny*Nz*M])

    #np.set_printoptions(threshold=np.inf)
    #r_bound = np.expand_dims(np.sqrt((Boundary_points_1[:Boundary_len,0] + l_1)**2 + (Boundary_points_1[:Boundary_len,1] + l_1)**2 + (Boundary_points_1[:Boundary_len,2] + l_1)**2), axis = 1)
    #x_bound = np.expand_dims((Boundary_points_1[:Boundary_len,0] + l_1), axis = 1)
    #y_bound = np.expand_dims((Boundary_points_1[:Boundary_len,1] + l_1), axis = 1)
    #z_bound = np.expand_dims((Boundary_points_1[:Boundary_len,2] + l_1), axis = 1)
    #normal_vector_1 = np.concatenate((x_bound/r_bound, y_bound/r_bound, z_bound/r_bound), axis = 1)
    #print(r_bound.shape)(1250, 1)
    #print(normal_vector_1.shape)(1250, 3)
    #stop
    for k in range(Nx):
        for n in range(Ny):
            for l in range(Nz):
                print(k,n,l)
                # u_t - c*u_x = 0
                in_ = points[k][n][l].cpu().detach().numpy()
                #print(in_[:Qx,0,0])
                #out = models[k][n](points[k][n])
                #values = out.cpu().detach().numpy()
                #f_1[(k*Ny*Nz + n*Nz + l)*Qx*Qy : (k*Ny*Nz + n*Nz + l + 1)*Qx*Qy,:] = anal_d2udx2_plus_d2udy2(in_[:Qx,:Qy,0], in_[:Qx,:Qy,1], in_[:Qx,:Qy,2]).reshape((-1,1))
                #g_1[(k*Ny*Nz + n*Nz + l)*Qx*Qy : (k*Ny*Nz + n*Nz + l + 1)*Qx*Qy,:] = anal_d2vdx2_plus_d2vdy2(in_[:Qx,:Qy,0], in_[:Qx,:Qy,1], in_[:Qx,:Qy,2]).reshape((-1,1))
                #print(f_1)
                #stop
                f_2[(k*Ny*Nz + n*Nz + l)*Boundary_len : (k*Ny*Nz + n*Nz + l)*Boundary_len+Boundary_len,:] =  np.sum(anal_dudx(Boundary_points_1[:Boundary_len,0],Boundary_points_1[:Boundary_len,1],Boundary_points_1[:Boundary_len,2]) * normal_vector_1, axis = 1).reshape(-1,1)\
                    - robin_parameter * vanal_v(Boundary_points_1[:,0],Boundary_points_1[:,1],Boundary_points_1[:,2])[:Boundary_len].reshape((-1,1))
                g_2[(k*Ny*Nz + n*Nz + l)*Boundary_len : (k*Ny*Nz + n*Nz + l)*Boundary_len+Boundary_len,:] =  np.sum(anal_dvdx(Boundary_points_1[:Boundary_len,0],Boundary_points_1[:Boundary_len,1],Boundary_points_1[:Boundary_len,2]) * normal_vector_1, axis = 1).reshape(-1,1)\
                    + robin_parameter * vanal_u(Boundary_points_1[:,0],Boundary_points_1[:,1],Boundary_points_1[:,2])[:Boundary_len].reshape((-1,1))
                for k_ in range(Nx):
                    for n_ in range(Ny):
                        for l_ in range(Nz):
                            out = models_u[k_][n_][l_](points[k][n][l])
                            Boundary_point_1 = torch.tensor(Boundary_points_1,requires_grad=True).to(device).unsqueeze(dim = 1).unsqueeze(dim = 2)
                            out_boundarys_1 = models_u[k_][n_][l_](Boundary_point_1)
                            out_boundary_1 = out_boundarys_1.cpu().detach().numpy()
                            out_boundary_1 = np.squeeze(out_boundary_1, axis = 2)
                            out_boundary_1 = np.squeeze(out_boundary_1, axis = 1)
                            values = out.cpu().detach().numpy()
                            M_begin = k_*Ny*Nz*M + n_*Nz*M + l_*M
                            grads = []
                            g_boundary_grads = []
                            g_boundary_grads_1 = []
                            grads_2_xx = []
                            grads_2_yy = []
                            grads_2_zz = []
                            for i in range(M):
                                g_1_1 = torch.autograd.grad(outputs=out[:,:,:,i], inputs=points[k][n][l],
                                                      grad_outputs=torch.ones_like(out[:,:,:,i]),
                                                      create_graph = True, retain_graph = True)[0]
                                grads.append(g_1_1.squeeze().cpu().detach().numpy())
                                g_boundary_x_1 = torch.autograd.grad(outputs=out_boundarys_1[:,:,:,i], inputs=Boundary_point_1,
                                              grad_outputs=torch.ones_like(out_boundarys_1[:,:,:,0]),
                                              create_graph = True, retain_graph = True)[0]
                                g_boundary_grads_1.append(g_boundary_x_1.squeeze().cpu().detach().numpy())
                                g_2_x = torch.autograd.grad(outputs=g_1_1[:,:,:,0], inputs=points[k][n][l],
                                                  grad_outputs=torch.ones_like(out[:,:,:,i]),
                                                  create_graph = True, retain_graph = True)[0]
                                g_2_y = torch.autograd.grad(outputs=g_1_1[:,:,:,1], inputs=points[k][n][l],
                                                  grad_outputs=torch.ones_like(out[:,:,:,i]),
                                                  create_graph = True, retain_graph = True)[0]
                                g_2_z = torch.autograd.grad(outputs=g_1_1[:,:,:,2], inputs=points[k][n][l],
                                                  grad_outputs=torch.ones_like(out[:,:,:,i]),
                                                  create_graph = True, retain_graph = True)[0]
                                grads_2_xx.append(g_2_x[:,:,:,0].squeeze().cpu().detach().numpy())
                                grads_2_yy.append(g_2_y[:,:,:,1].squeeze().cpu().detach().numpy())
                                grads_2_zz.append(g_2_z[:,:,:,2].squeeze().cpu().detach().numpy())
                            #print(np.array(g_boundary_grads_1).shape)(100,1250,3)
                            #stop
                            g_boundary_grads_1 = np.array(g_boundary_grads_1)[:,:Boundary_len,:].swapaxes(1,2)
                            normal_boundary_grads_1 = g_boundary_grads_1 * np.expand_dims(normal_vector_1.T, axis = 0)
                            normal_boundary_grads_1 = np.sum(normal_boundary_grads_1, axis = 1).swapaxes(0,1)
                            grads = np.array(grads).swapaxes(0,4)
                            value = values[:Qx,:Qy,:Qz, :]
                            value = value.reshape(-1,M)
                            grads_2_xx = np.array(grads_2_xx)
                            grads_2_xx = grads_2_xx[:,:Qx,:Qy,:Qz]
                            grads_2_xx = grads_2_xx.transpose(1,2,3,0).reshape(-1,M)
                            grads_2_yy = np.array(grads_2_yy)
                            grads_2_yy = grads_2_yy[:,:Qx,:Qy,:Qz]
                            grads_2_yy = grads_2_yy.transpose(1,2,3,0).reshape(-1,M)
                            grads_2_zz = np.array(grads_2_zz)
                            grads_2_zz = grads_2_zz[:,:Qx,:Qy,:Qz]
                            grads_2_zz = grads_2_zz.transpose(1,2,3,0).reshape(-1,M)
                            A_1[k*Ny*Nz*Qx*Qy*Qz + n*Nz*Qx*Qy*Qz +l*Qx*Qy*Qz: k*Ny*Nz*Qx*Qy*Qz + n*Nz*Qx*Qy*Qz +(l+1)*Qx*Qy*Qz, M_begin : M_begin + M] = grads_2_xx + grads_2_yy + grads_2_zz + lamb*value
                            A_4[(k*Ny*Nz+n*Nz+l)*Boundary_len:(k*Ny*Nz+n*Nz+l+1)*Boundary_len, M_begin : M_begin + M] = normal_boundary_grads_1
                            A_5[(k*Ny*Nz+n*Nz+l)*Boundary_len:(k*Ny*Nz+n*Nz+l+1)*Boundary_len, M_begin : M_begin + M] = robin_parameter * out_boundary_1
                for k_ in range(Nx):
                    for n_ in range(Ny):
                        for l_ in range(Nz):
                            out = models_v[k_][n_][l_](points[k][n][l])
                            Boundary_point_1 = torch.tensor(Boundary_points_1,requires_grad=True).to(device).unsqueeze(dim = 1).unsqueeze(dim = 2)
                            out_boundarys_1 = models_v[k_][n_][l_](Boundary_point_1)
                            out_boundary_1 = out_boundarys_1.cpu().detach().numpy()
                            out_boundary_1 = np.squeeze(out_boundary_1, axis = 2)
                            out_boundary_1 = np.squeeze(out_boundary_1, axis = 1)
                            values = out.cpu().detach().numpy()
                            M_begin = k_*Ny*Nz*M + n_*Nz*M + l_*M
                            grads = []
                            g_boundary_grads = []
                            g_boundary_grads_1 = []
                            grads_2_xx = []
                            grads_2_yy = []
                            grads_2_zz = []
                            for i in range(M):
                                g_1_1 = torch.autograd.grad(outputs=out[:,:,:,i], inputs=points[k][n][l],
                                                      grad_outputs=torch.ones_like(out[:,:,:,i]),
                                                      create_graph = True, retain_graph = True)[0]
                                grads.append(g_1_1.squeeze().cpu().detach().numpy())
                                g_boundary_x_1 = torch.autograd.grad(outputs=out_boundarys_1[:,:,:,i], inputs=Boundary_point_1,
                                              grad_outputs=torch.ones_like(out_boundarys_1[:,:,:,0]),
                                              create_graph = True, retain_graph = True)[0]
                                g_boundary_grads_1.append(g_boundary_x_1.squeeze().cpu().detach().numpy())
                                g_2_x = torch.autograd.grad(outputs=g_1_1[:,:,:,0], inputs=points[k][n][l],
                                                  grad_outputs=torch.ones_like(out[:,:,:,i]),
                                                  create_graph = True, retain_graph = True)[0]
                                g_2_y = torch.autograd.grad(outputs=g_1_1[:,:,:,1], inputs=points[k][n][l],
                                                  grad_outputs=torch.ones_like(out[:,:,:,i]),
                                                  create_graph = True, retain_graph = True)[0]
                                g_2_z = torch.autograd.grad(outputs=g_1_1[:,:,:,2], inputs=points[k][n][l],
                                                  grad_outputs=torch.ones_like(out[:,:,:,i]),
                                                  create_graph = True, retain_graph = True)[0]
                                grads_2_xx.append(g_2_x[:,:,:,0].squeeze().cpu().detach().numpy())
                                grads_2_yy.append(g_2_y[:,:,:,1].squeeze().cpu().detach().numpy())
                                grads_2_zz.append(g_2_z[:,:,:,2].squeeze().cpu().detach().numpy())
                            #print(np.array(g_boundary_grads_1).shape)(100,1250,3)
                            #stop
                            g_boundary_grads_1 = np.array(g_boundary_grads_1)[:,:Boundary_len,:].swapaxes(1,2)
                            normal_boundary_grads_1 = g_boundary_grads_1 * np.expand_dims(normal_vector_1.T, axis = 0)
                            normal_boundary_grads_1 = np.sum(normal_boundary_grads_1, axis = 1).swapaxes(0,1)
                            grads = np.array(grads).swapaxes(0,4)
                            value = values[:Qx,:Qy,:Qz, :]
                            value = value.reshape(-1,M)
                            grads_2_xx = np.array(grads_2_xx)
                            grads_2_xx = grads_2_xx[:,:Qx,:Qy,:Qz]
                            grads_2_xx = grads_2_xx.transpose(1,2,3,0).reshape(-1,M)
                            grads_2_yy = np.array(grads_2_yy)
                            grads_2_yy = grads_2_yy[:,:Qx,:Qy,:Qz]
                            grads_2_yy = grads_2_yy.transpose(1,2,3,0).reshape(-1,M)
                            grads_2_zz = np.array(grads_2_zz)
                            grads_2_zz = grads_2_zz[:,:Qx,:Qy,:Qz]
                            grads_2_zz = grads_2_zz.transpose(1,2,3,0).reshape(-1,M)
                            B_1[k*Ny*Nz*Qx*Qy*Qz + n*Nz*Qx*Qy*Qz +l*Qx*Qy*Qz: k*Ny*Nz*Qx*Qy*Qz + n*Nz*Qx*Qy*Qz +(l+1)*Qx*Qy*Qz, M_begin : M_begin + M] = grads_2_xx + grads_2_yy + grads_2_zz + lamb*value
                            B_4[(k*Ny*Nz+n*Nz+l)*Boundary_len:(k*Ny*Nz+n*Nz+l+1)*Boundary_len, M_begin : M_begin + M] = - robin_parameter * out_boundary_1
                            B_5[(k*Ny*Nz+n*Nz+l)*Boundary_len:(k*Ny*Nz+n*Nz+l+1)*Boundary_len, M_begin : M_begin + M] = normal_boundary_grads_1
    D_1 = np.concatenate((A_1,C_1),axis = 1)
    D_2 = np.concatenate((C_1,B_1),axis = 1)
    D_5 = np.concatenate((A_4,B_4),axis = 1)
    D_6 = np.concatenate((A_5,B_5),axis = 1)
    A = np.concatenate((D_1,D_2,D_5,D_6),axis = 0)
    f = np.concatenate((f_1,g_1,f_2,g_2),axis = 0)
    print(f.shape)
    return(A,f)

def grid_cir(x):
    a = np.ones((x.shape[0], x.shape[1], x.shape[2]))
    eps = 1e-8  # small positive value to avoid division by zero
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for l in range(x.shape[2]):
                # Cartesian to spherical coordinates
                r = np.sqrt((x[i,j,l,2]-0.5) ** 2 + (x[i,j,l,1]-0.5) ** 2 + (x[i,j,l,0]-0.5) ** 2) + eps
                theta = np.arccos((x[i,j,l,2]-0.5) / r)
                phi = np.arctan2((x[i,j,l,1]-0.5), (x[i,j,l,0]-0.5))

                if r - eps >= 0.4 + 0.1 * np.cos(5 * theta) * np.sin(5 * phi):
                    a[i,j,l] = 0.0
    return a

'''
def grid_cir(x):
    a = np.ones((x.shape[0], x.shape[1], x.shape[2]))
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for l in range(x.shape[2]):
                if (x[i,j,l,2]-0.5) ** 2 + (x[i,j,l,1]-0.5) ** 2 + (x[i,j,l,0]-0.5) ** 2 <= 0.04 or (x[i,j,l,2]-0.5) ** 2 + (x[i,j,l,1]-0.5) ** 2 + (x[i,j,l,0]-0.5) ** 2 >= 0.25:
                    a[i,j,l] = 0.0
    return a
'''
def test(models_u,Nx,Ny,Nz,M,Qx,Qy,Qz,w,plot = False):
    epsilon = []
    true_values = []
    numerical_values = []
    test_Qx = int(100/Nx)
    test_Qy = int(100/Ny)
    test_Qz = int(100/Nz)
    x_devide = np.linspace(0, 1, 100 + 1)[:100]
    y_devide = np.linspace(0, 1, 100 + 1)[:100]
    z_devide = np.linspace(0, 1, 100 + 1)[:100]
    circle = np.array(list(itertools.product(x_devide,y_devide,z_devide))).reshape(100,100,100,3)
    grid_circle = grid_cir(circle)
    for k in range(Nx):
        epsilon_x = []
        true_value_x = []
        numerical_value_x = []
        x_min = X_edge/Nx * k
        x_max = X_edge/Nx * (k+1)
        x_devide = np.linspace(x_min, x_max, test_Qx + 1)[:test_Qx]
        for n in range(Ny):
            # forward and grad
            y_min = Y_edge/Ny * n
            y_max = Y_edge/Ny * (n+1)
            y_devide = np.linspace(y_min, y_max, test_Qy + 1)[:test_Qy]
            epsilon_x_y = []
            true_value_x_y = []
            numerical_value_x_y = []
            for l in range(Nz):
                numerical_value = None
                z_min = Z_edge/Nz * n
                z_max = Z_edge/Nz * (n+1)
                z_devide = np.linspace(z_min, z_max, test_Qz + 1)[:test_Qz]
                grid = np.array(list(itertools.product(x_devide,y_devide,z_devide))).reshape(test_Qx,test_Qy,test_Qz,3)
                test_point = torch.tensor(grid,requires_grad=True).to(device)
                in_ = test_point.cpu().detach().numpy()
                true_value = vanal_u(in_[:,:,:,0], in_[:,:,:,1], in_[:,:,:,2])
                for k_ in range(Nx):
                    for n_ in range(Ny):
                        for l_ in range(Nz):
                            out = models_u[k_][n_][l_](test_point)
                            values = out.cpu().detach().numpy()
                            if numerical_value is None:
                                numerical_value = np.dot(values, w[k_*Ny*Nz*M + n_*Nz*M + l_*M: k_*Ny*Nz*M + n_*Nz*M + (l_+1)*M,:]).reshape(test_Qx,test_Qy,test_Qz)
                            else:
                                numerical_value = numerical_value + np.dot(values, w[k_*Ny*Nz*M + n_*Nz*M + l_*M: k_*Ny*Nz*M + n_*Nz*M + (l_+1)*M,:]).reshape(test_Qx,test_Qy,test_Qz)
                e = np.abs(true_value - numerical_value)
                true_value_x_y.append(true_value)
                numerical_value_x_y.append(numerical_value)
                epsilon_x_y.append(e)
            epsilon_x_y = np.concatenate(epsilon_x_y, axis=2)
            epsilon_x.append(epsilon_x_y)
            true_value_x_y = np.concatenate(true_value_x_y, axis=2)
            true_value_x.append(true_value_x_y)
            numerical_value_x_y = np.concatenate(numerical_value_x_y, axis=2)
            numerical_value_x.append(numerical_value_x_y)
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
    true = true_values.reshape((-1,1))
    print('********************* ERROR *********************')
    print('Nx=%s,Ny=%s,Nz=%s,M=%s,Qx=%s,Qy=%s,Qz=%s'%(Nx,Ny,Nz,M,Qx,Qy,Qz))
    print('L_inf=',e.max()/abs(true).max(),'L_2=',math.sqrt(sum(e*e)/len(e))/math.sqrt(sum(true*true)/len(true)))
    #np.save('./epsilon_psi2.npy',epsilon)
    return e, true

def test1(models_v,Nx,Ny,Nz,M,Qx,Qy,Qz,w,plot = False):
    epsilon = []
    true_values = []
    numerical_values = []
    test_Qx = int(100/Nx)
    test_Qy = int(100/Ny)
    test_Qz = int(100/Nz)
    x_devide = np.linspace(0, 1, 100 + 1)[:100]
    y_devide = np.linspace(0, 1, 100 + 1)[:100]
    z_devide = np.linspace(0, 1, 100 + 1)[:100]
    circle = np.array(list(itertools.product(x_devide,y_devide,z_devide))).reshape(100,100,100,3)
    grid_circle = grid_cir(circle)
    for k in range(Nx):
        epsilon_x = []
        true_value_x = []
        numerical_value_x = []
        x_min = X_edge/Nx * k
        x_max = X_edge/Nx * (k+1)
        x_devide = np.linspace(x_min, x_max, test_Qx + 1)[:test_Qx]
        for n in range(Ny):
            # forward and grad
            y_min = Y_edge/Ny * n
            y_max = Y_edge/Ny * (n+1)
            y_devide = np.linspace(y_min, y_max, test_Qy + 1)[:test_Qy]
            epsilon_x_y = []
            true_value_x_y = []
            numerical_value_x_y = []
            for l in range(Nz):
                numerical_value = None
                z_min = Z_edge/Nz * n
                z_max = Z_edge/Nz * (n+1)
                z_devide = np.linspace(z_min, z_max, test_Qz + 1)[:test_Qz]
                grid = np.array(list(itertools.product(x_devide,y_devide,z_devide))).reshape(test_Qx,test_Qy,test_Qz,3)
                test_point = torch.tensor(grid,requires_grad=True).to(device)
                in_ = test_point.cpu().detach().numpy()
                true_value = vanal_v(in_[:,:,:,0], in_[:,:,:,1], in_[:,:,:,2])
                for k_ in range(Nx):
                    for n_ in range(Ny):
                        for l_ in range(Nz):
                            out = models_v[k_][n_][l_](test_point)
                            values = out.cpu().detach().numpy()
                            if numerical_value is None:
                                numerical_value = np.dot(values, w[k_*Ny*Nz*M + n_*Nz*M + l_*M: k_*Ny*Nz*M + n_*Nz*M + (l_+1)*M,:]).reshape(test_Qx,test_Qy,test_Qz)
                            else:
                                numerical_value = numerical_value + np.dot(values, w[k_*Ny*Nz*M + n_*Nz*M + l_*M: k_*Ny*Nz*M + n_*Nz*M + (l_+1)*M,:]).reshape(test_Qx,test_Qy,test_Qz)
                e = np.abs(true_value - numerical_value)
                true_value_x_y.append(true_value)
                numerical_value_x_y.append(numerical_value)
                epsilon_x_y.append(e)
            epsilon_x_y = np.concatenate(epsilon_x_y, axis=2)
            epsilon_x.append(epsilon_x_y)
            true_value_x_y = np.concatenate(true_value_x_y, axis=2)
            true_value_x.append(true_value_x_y)
            numerical_value_x_y = np.concatenate(numerical_value_x_y, axis=2)
            numerical_value_x.append(numerical_value_x_y)
        epsilon_x = np.concatenate(epsilon_x, axis=1)
        epsilon.append(epsilon_x)
        true_value_x = np.concatenate(true_value_x, axis=1)
        true_values.append(true_value_x)
        numerical_value_x = np.concatenate(numerical_value_x, axis=1)
        numerical_values.append(numerical_value_x)
    epsilon = np.concatenate(epsilon, axis=0)
    epsilon = epsilon * grid_circle
    true_values = np.concatenate(true_values, axis=0) * grid_circle
    numerical_values = np.concatenate(numerical_values, axis=0) * grid_circle
    e = epsilon.reshape((-1,1))
    true = true_values.reshape((-1,1))
    print('********************* ERROR *********************')
    print('Nx=%s,Ny=%s,Nz=%s,M=%s,Qx=%s,Qy=%s,Qz=%s'%(Nx,Ny,Nz,M,Qx,Qy,Qz))
    print('L_inf=',e.max()/abs(true).max(),'L_2=',math.sqrt(sum(e*e)/len(e))/math.sqrt(sum(true*true)/len(true)))
    #np.save('./epsilon_psi2.npy',epsilon)
    return e, true

def main(Nx,Ny,Nz,M,Qx,Qy,Qz,plot = False,moore = False):
    # prepare models and collocation pointss
    models_u,points = pre_define(Nx,Ny,Nz,M,Qx,Qy,Qz)
    #set_seed(0)
    #models_v,points = pre_define(Nx,Ny,Nz,M,Qx,Qy,Qz)
    models_v = models_u
    # matrix define (Aw=b)
    A,f = cal_matrix(models_u, models_v,points,Nx,Ny,Nz,M,Qx,Qy,Qz)

    max_value = 10.0
    for i in range(len(A)):
        if np.abs(A[i,:].max())==0:
            print("error line : ",i)
            continue
        ratio = max_value/np.abs(A[i,:]).max()
        A[i,:] = A[i,:]*ratio
        f[i] = f[i]*ratio
    A = torch.tensor(A).to(device)
    f = torch.tensor(f).to(device)
    length = A.shape[1]//2
    print(length)
    # solve
    if moore:
        inv_coeff_mat = torch.tensor(pinv(A)).to(device)  # moore-penrose inverse, shape: (n_units,n_colloc+2)
        w = torch.linalg.lstsq(inv_coeff_mat, f)
    else:
        a = torch.linalg.lstsq(A,f)
        w = a[0].cpu().detach().numpy()
        #print(w)
        #print(w.shape)
        print(a[1].cpu().detach().numpy())
    # test
    e_real, true_real = test(models_u,Nx,Ny,Nz,M,Qx,Qy,Qz,np.expand_dims(w[:length,0],1),plot)
    e_imag, true_imag = test1(models_v,Nx,Ny,Nz,M,Qx,Qy,Qz,np.expand_dims(w[length:,0],1),plot)
    e_1 = np.sqrt(e_real**2 + e_imag**2)
    true_1 = np.sqrt(true_real**2 + true_imag**2)
    print('********************* ERROR *********************')
    print('Nx=%s,Ny=%s,Nz=%s,M=%s,Qx=%s,Qy=%s,Qz=%s'%(Nx,Ny,Nz,M,Qx,Qy,Qz))
    print('L_inf=',e_1.max()/true_1.max(),'L_2=',math.sqrt(sum(e_1*e_1)/len(e_1))/math.sqrt(sum(true_1*true_1)/len(true_1)))
    return 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    set_seed(100)
    result = []
    main(2,2,2,100,10,10,10,True)
    #main(2,2,2,400,10,10,10,True)