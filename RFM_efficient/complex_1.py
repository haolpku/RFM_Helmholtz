#用手算RFM矩阵
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import math
import itertools
import scipy.special as sps
torch.set_default_dtype(torch.float64)
#first get args
parser = argparse.ArgumentParser()

parser.add_argument('--basis', type=int, default=500)
parser.add_argument('--k', type=float, default=20)
parser.add_argument('--multi', type=float, default=1.2)
parser.add_argument('--Boundary_len', type=int, default=1000)

args = parser.parse_args()

#Boundary points

# 生成均匀分布的θ值，从0到2π，包括2π。
Boundary_len = args.Boundary_len
theta = np.linspace(0, 2*np.pi, Boundary_len)
# 计算每个θ对应的r值。
r = 0.4 + 0.1 * np.cos(5*theta)

# 将极坐标转换为笛卡尔坐标。
x = r * np.cos(theta) + 0.5
y = r * np.sin(theta) + 0.5

# 组成一个1000x2的numpy数组
points = np.vstack((x, y)).T
Boundary_points_1 = points

# 计算极坐标的切向量
tangent_r = -0.2 * 5 * np.sin(5 * theta)

# 根据切向量计算法向量的极坐标
normal_theta = theta + np.pi/2
normal_r = np.where(theta <= np.pi, np.abs(tangent_r), -np.abs(tangent_r))

# 将极坐标转换为笛卡尔坐标
normal_x = normal_r * np.cos(normal_theta)
normal_y = normal_r * np.sin(normal_theta)

# 归一化处理
norms = np.sqrt(normal_x**2 + normal_y**2)
norms = np.where(norms == 0, 1, norms)
normal_x /= norms
normal_y /= norms

normal_vector_1 = np.vstack((normal_x, normal_y)).T
print(normal_x.shape)

def set_seed(x):
    np.random.seed(x)
    torch.manual_seed(x)
    torch.cuda.manual_seed_all(x)
    torch.backends.cudnn.deterministic = True

#一些预置参数
L = 1.0
tf = 1.0
epsilon = 0
k_1 = args.k
lamb = k_1**2
l = 1
n = 0
delta = 0.0001*1/k_1
robin_parameter = 1 * k_1

rand_mag = args.multi * k_1
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.uniform_(m.weight, a = -rand_mag, b = rand_mag)
        nn.init.uniform_(m.bias, a = -rand_mag, b = rand_mag)


class RFM_rep(nn.Module):
    def __init__(self, in_features, M):
        super(RFM_rep, self).__init__()
        self.hidden_layer = nn.Linear(in_features, M, bias=True) 
        self.output_layer = nn.Linear(M, 1, bias=False)
    def forward_hidden_only(self,y):
        y = self.hidden_layer(y)
        y = torch.sin(y)
        return y
    
    def forward(self,y):
        y = self.hidden_layer(y)
        y = torch.sin(y)
        y = self.output_layer(y)
        return y


#解函数实部
def anal_u(x,y):
    x = x + l
    y = y + l
    h1 = sps.hankel1(n, k_1*(x**2 + y**2)**(1/2)) 
    u = np.real(h1) * np.cos(n * np.arctan2(y,x)) -  np.imag(h1) * np.sin(n * np.arctan2(y,x))
    return u

def anal_d2udx2_plus_d2udy2(x,y):
    x = x + l
    y = y + l
    #print(np.array(complex(np.cos(n * np.arctan2(y,x)), np.sin(n * np.arctan2(y,x)))))
    u = np.real(k_1**2 * (sps.hankel1(n, k_1*(x**2 + y**2)**(1/2)) + sps.hankel1(n+2, k_1*(x**2 + y**2)**(1/2))) - 2*k_1*(n+1)/((x**2 + y**2)**(1/2)) * sps.hankel1(n+1, k_1*(x**2 + y**2)**(1/2))) * np.cos(n * np.arctan2(y,x)) \
        - np.imag(k_1**2 * (sps.hankel1(n, k_1*(x**2 + y**2)**(1/2)) + sps.hankel1(n+2, k_1*(x**2 + y**2)**(1/2))) - 2*k_1*(n+1)/((x**2 + y**2)**(1/2)) * sps.hankel1(n+1, k_1*(x**2 + y**2)**(1/2))) *  np.sin(n * np.arctan2(y,x))
    return u

#先共用一套基函数
def pre_define(Nx,Ny,M,Qx,Qy):#x负责分块，y负责数据
    model = RFM_rep(in_features = 2, M = M)
    model = model.to(dtype=torch.float64)
    model.hidden_layer.apply(weights_init)
    x_devide = np.linspace(0, 1, Qx + 1)
    t_devide = np.linspace(0, 1, Qy + 1)
    grid = np.array(list(itertools.product(x_devide,t_devide))).reshape(Qx+1,Qy+1,2)
    point = torch.tensor(grid, dtype = torch.float64, requires_grad=True)
    return(model, point)

def cal_matrix(models_u,points,Nx,Ny,M,Qx,Qy):
# matrix define (Aw=b)
    #实虚分开
    A_1 = torch.zeros([Nx*Ny*Qx*Qy,Nx*Ny*M]) # u_t - c*u_x = 0
    f_1 = np.zeros([Nx*Ny*Qx*Qy,1])
    A_2 = torch.zeros([Nx*Ny*Boundary_len,Nx*Ny*M]) # u(x,0) = h(x)
    f_2 = np.zeros([Nx*Ny*Boundary_len,1])
    # u_t - c*u_x = 0
    in_ = points.detach().numpy()
    f_1[0 : Qx*Qy,:] = anal_d2udx2_plus_d2udy2(in_[:Qx,:Qy,0], in_[:Qx,:Qy,1]).reshape((-1,1))
    f_2[0 : Boundary_len,:] = anal_u(Boundary_points_1[:,0],Boundary_points_1[:,1])[:Boundary_len].reshape((-1,1))
    del in_
    # u_t - c*u_x = 0
    input_1_u = points[:Qx,:Qy,:]
    out = models_u.forward_hidden_only(input_1_u)
    #Boundary_1
    Boundary_point_1_1 = torch.tensor(Boundary_points_1[:Boundary_len,:], dtype = torch.float64, requires_grad=True).unsqueeze(dim = 1)
    out_boundarys_1 = models_u.forward_hidden_only(Boundary_point_1_1)
    weight = models_u.hidden_layer.weight.data
    weight_1 = torch.sum(weight ** 2, dim=1, keepdim=True)
    value = out[:Qx,:Qy, :]
    value = value.reshape(-1,M)
    del points
    del Boundary_point_1_1
    out_boundary_1 = out_boundarys_1.reshape(-1,M)
    A_1[0 : Qx*Qy, 0 : M] = value * (-weight_1.T + lamb)
    A_2[0 : Boundary_len, 0 : M] = out_boundary_1
    A = torch.cat((A_1, A_2),dim = 0)
    f = np.concatenate((f_1, f_2),axis = 0)
    f = torch.tensor(f, dtype = torch.float64)
    return(A,f)

def grid_cir(M):
    # 生成二维网格
    X = M[:,:,0]
    Y = M[:,:,1]
    grid = np.zeros(X.shape)

    # 遍历所有点
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            # 将笛卡尔坐标转换为极坐标
            r = np.sqrt((X[i,j]-0.5)**2 + (Y[i,j]-0.5)**2)
            theta = np.arctan2((Y[i,j]-0.5), (X[i,j]-0.5))
            # 计算闭曲线在θ处的r值
            r_boundary = 0.4 + 0.1 * np.cos(5*theta)
            # 判断点是否在闭曲线内部
            if r <= r_boundary:
                grid[i,j] = 1
    return grid

def test(models,Nx,Ny,M,Qx,Qy,w,plot = False):
    epsilon = []
    true_values = []
    numerical_values = []
    test_Qx = int(100/Nx)
    test_Qy = int(100/Ny)
    L = 1.0
    tf = 1.0
    x_devide = np.linspace(0, L, 100 + 1)[:100]
    t_devide = np.linspace(0, tf, 100 + 1)[:100]
    circle = np.array(list(itertools.product(x_devide,t_devide))).reshape(100,100,2)
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
            test_point = torch.tensor(grid, dtype = torch.float64, requires_grad=True)
            in_ = test_point.cpu().detach().numpy()
            true_value = anal_u(in_[:,:,0],in_[:,:,1])
            for k_ in range(Nx):
                for n_ in range(Ny):
                        out = models.forward_hidden_only(test_point)
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
    true_values = np.concatenate(true_values, axis=0) * grid_circle
    numerical_values = np.concatenate(numerical_values, axis=0)
    e = epsilon.reshape((-1,1))
    true = true_values.reshape((-1,1))
    print('********************* ERROR *********************')
    print('Nx=%s,Ny=%s,M=%s,Qx=%s,Qy=%s'%(Nx,Ny,M,Qx,Qy))
    print('L_inf=',e.max()/true.max(),'L_2=',math.sqrt(sum(e*e)/len(e))/math.sqrt(sum(true*true)/len(true)))
    return 0

def linear_least_squares_LBFGS(A, b, max_iter = 100):
    # 确保A和b是PyTorch张量
    A = torch.tensor(A, dtype=torch.float64)
    b = torch.tensor(b, dtype=torch.float64)

    # 初始化参数向量x（初始猜测）
    x = torch.zeros(A.size(1), 1, requires_grad=True)

    # 定义优化器
    optimizer = torch.optim.LBFGS([x], max_iter=max_iter, line_search_fn='strong_wolfe')

    losses = []
    # 定义最小二乘目标函数
    def closure():
        optimizer.zero_grad()
        loss = torch.norm(A.mm(x) - b) ** 2
        loss.backward()
        losses.append(loss.item())
        return loss

    # 运行优化器
    optimizer.step(closure)

    # 绘制损失曲线
    '''
    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.show()
    print(losses)
    '''
    # 返回最优解
    return x

def main(Nx,Ny,M,Qx,Qy,plot = False,moore = False):
    models_u,points = pre_define(Nx,Ny,M,Qx,Qy)
    # test
    A,f = cal_matrix(models_u,points,Nx,Ny,M,Qx,Qy)
    del points
    #stop
    A_numpy = A.cpu().detach().numpy()
    max_value = 10.0
    for i in range(len(A)):
        ratio = max_value/np.abs(A_numpy[i,:]).max()
        A[i,:] = A[i,:]*ratio
        f[i] = f[i]*ratio
    A_t = torch.mm(A.t(), A).to(device)
    b = torch.mm(A.t(), f).to(device)
    a = torch.linalg.lstsq(A_t, b)
    #print(a.shape)
    #stop
    A = A.cpu()
    f = f.cpu()
    w_best = a[0]
    test(models_u,Nx,Ny,M,Qx,Qy,w_best.cpu().detach().numpy(),plot)
    return 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if __name__ == '__main__':
    torch.cuda.empty_cache()
    set_seed(100)
    result = []
    main(1,1,args.basis,int(0.5 * args.k),int(0.5 * args.k),True)
