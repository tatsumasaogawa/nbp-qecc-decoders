# https://github.com/samn33/qlazy/blob/master/example/py/SurfaceCode/error_correction.py

from qlazy import QState
import numpy as np
import math
import torch

N = 3

Lattice = [{'edge': 0, 'faces':[0, 6], 'vertices':[0, 1]},
           {'edge': 1, 'faces':[1, 7], 'vertices':[1, 2]},
           {'edge': 2, 'faces':[2, 8], 'vertices':[0, 2]},
           {'edge': 3, 'faces':[0, 2], 'vertices':[0, 3]},
           {'edge': 4, 'faces':[0, 1], 'vertices':[1, 4]},
           {'edge': 5, 'faces':[1, 2], 'vertices':[2, 5]},
           {'edge': 6, 'faces':[0, 3], 'vertices':[3, 4]},
           {'edge': 7, 'faces':[1, 4], 'vertices':[4, 5]},
           {'edge': 8, 'faces':[2, 5], 'vertices':[3, 5]},
           {'edge': 9, 'faces':[3, 5], 'vertices':[3, 6]},
           {'edge':10, 'faces':[3, 4], 'vertices':[4, 7]},
           {'edge':11, 'faces':[4, 5], 'vertices':[5, 8]},
           {'edge':12, 'faces':[3, 6], 'vertices':[6, 7]},
           {'edge':13, 'faces':[4, 7], 'vertices':[7, 8]},
           {'edge':14, 'faces':[5, 8], 'vertices':[6, 8]},
           {'edge':15, 'faces':[6, 8], 'vertices':[0, 6]},
           {'edge':16, 'faces':[6, 7], 'vertices':[1, 7]},
           {'edge':17, 'faces':[7, 8], 'vertices':[2, 8]}]

F_OPERATORS = [{'face':0, 'edges':[ 0,  3,  4,  6]},
               {'face':1, 'edges':[ 1,  4,  5,  7]},
               {'face':2, 'edges':[ 2,  3,  5,  8]},
               {'face':3, 'edges':[ 6,  9, 10, 12]},
               {'face':4, 'edges':[ 7, 10, 11, 13]},
               {'face':5, 'edges':[ 8,  9, 11, 14]},
               {'face':6, 'edges':[ 0, 12, 15, 16]},
               {'face':7, 'edges':[ 1, 13, 16, 17]},
               {'face':8, 'edges':[ 2, 14, 15, 17]}]

V_OPERATORS = [{'vertex':0, 'edges':[ 0,  2,  3, 15]},
               {'vertex':1, 'edges':[ 0,  1,  4, 16]},
               {'vertex':2, 'edges':[ 1,  2,  5, 17]},
               {'vertex':3, 'edges':[ 3,  6,  8,  9]},
               {'vertex':4, 'edges':[ 4,  6,  7, 10]},
               {'vertex':5, 'edges':[ 5,  7,  8, 11]},
               {'vertex':6, 'edges':[ 9, 12, 14, 15]},
               {'vertex':7, 'edges':[10, 12, 13, 16]},
               {'vertex':8, 'edges':[11, 13, 14, 17]}]

LZ_OPERATORS = [{'logical_qid':0, 'edges':[0, 1,  2]},
                {'logical_qid':1, 'edges':[3, 9, 15]}]

LX_OPERATORS = [{'logical_qid':0, 'edges':[0, 6, 12]},
                {'logical_qid':1, 'edges':[3, 4,  5]}]

H = [[0] * 4 * N ** 2 for _ in range(2 * N ** 2)]
for fop in F_OPERATORS:
    i = fop['face']
    qid = fop['edges']
    for j in qid:
        H[i][j] = 1

for vop in V_OPERATORS:
    i = vop['vertex'] + N ** 2
    qid = vop['edges']
    for j in qid:
        H[i][j + 2 * N ** 2] = 1

print(np.array(H))

H_v = [[0] * 4 * N ** 2 for _ in range(2 * N ** 2)]
# H_v の左半分を計算
for i in range(2 * N ** 2):
    for j in range(2 * N ** 2):
        if H[i][j + 2 * N ** 2] == 0:
            H_v[i][j] = 1
        else:
            H_v[i][j] = 0
# 右半分を計算
for i in range(2 * N ** 2):
    for j in range(2 * N ** 2, 4 * N ** 2):
        if H[i][j - 2 * N ** 2] == 0:
            H_v[i][j] = 1
        else:
            H_v[i][j] = 0

H_tensor = torch.tensor(H, dtype=torch.float32)
print(H_tensor.dtype)
H_v_tensor = torch.tensor(H_v, dtype=torch.float32)


class MyQState(QState):
    
    def Lz(self, q):

        [self.z(i) for i in LZ_OPERATORS[q]['edges']]
        return self
        
    def Lx(self, q):

        [self.x(i) for i in LX_OPERATORS[q]['edges']]
        return self

def make_logical_zero():

    qs = MyQState(19)  # data:18 + ancilla:1

    mvals = [0, 0, 0, 0, 0, 0, 0, 0, 0] # measured values of 9 plaquette operators
    for vop in V_OPERATORS: # measure and get measured values of 9 star operators
        qid = vop['edges']
        qs.h(18).cx(18,qid[0]).cx(18,qid[1]).cx(18,qid[2]).cx(18,qid[3]).h(18)
        mvals.append(int(qs.m(qid=[18]).last))
        qs.reset(qid=[18])
        
    return qs, mvals

def measure_syndrome(qs, mvals):

    syn = []
    for fop in F_OPERATORS:  # plaquette operators
        qid = fop['edges']
        qs.h(18).cz(18,qid[0]).cz(18,qid[1]).cz(18,qid[2]).cz(18,qid[3]).h(18)
        syn.append(int(qs.m(qid=[18]).last))
        qs.reset(qid=[18])
        
    for vop in V_OPERATORS:  # start operators
        qid = vop['edges']
        qs.h(18).cx(18,qid[0]).cx(18,qid[1]).cx(18,qid[2]).cx(18,qid[3]).h(18)
        syn.append(int(qs.m(qid=[18]).last))
        qs.reset(qid=[18])
        
    for i in range(len(syn)): syn[i] = syn[i]^mvals[i]

    return syn

def get_error_chain(syn):

    face_id = [i for i,v in enumerate(syn) if i < 9 and v == 1]
    vertex_id = [i-9 for i,v in enumerate(syn) if i >= 9 and v == 1]

    e_chn = []
    if face_id != []:  # chain type: X
        for lat in Lattice:
            if lat['faces'][0] == face_id[0] and lat['faces'][1] == face_id[1]:
                e_chn.append({'type':'X', 'qid':[lat['edge']]})
                break

    if vertex_id != []: # chain type: Z
        for lat in Lattice:
            if lat['vertices'][0] == vertex_id[0] and lat['vertices'][1] == vertex_id[1]:
                e_chn.append({'type':'Z', 'qid':[lat['edge']]})
                break

    return e_chn

def error_correction(qs, e_chn):

    for c in e_chn:
        if c['type'] == 'X': [qs.x(i) for i in c['qid']]
        if c['type'] == 'Z': [qs.z(i) for i in c['qid']]

if __name__ == '__main__':

    print("* initial state: logical |11>")
    qs_ini, mval_list = make_logical_zero()  # logical |00>
    qs_ini.Lx(0).Lx(1)  # logical |00> -> |11>
    qs_fin = qs_ini.clone()  # for evaluating later

    print("* add noise")
    qs_fin.x(7)  # bit flip error at #7
    # qs_fin.z(7).x(7)  # bit and phase flip error at #7
    e_v = torch.tensor([0] * 4 * N ** 2, dtype=torch.float32)
    e_v[7] = 1.

    syndrome = measure_syndrome(qs_fin, mval_list)
    syndrome_tensor = torch.tensor(syndrome)
    print('syndrome_tensor.dtype:', syndrome_tensor.dtype)

    err_chain = get_error_chain(syndrome)
    print("* syndrome measurement:", syndrome)
    print("* error chain:", err_chain)

    error_correction(qs_fin, err_chain)
    print("* fidelity after error correction:", qs_fin.fidelity(qs_ini))


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pyldpc.code import make_ldpc
from pyldpc.encoder import encode_random_message, encode
from pyldpc.decoder import get_message

np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)


class BeliefPropagation(nn.Module):
    def __init__(self, num_nodes, num_edges):
        super(BeliefPropagation, self).__init__()
        
        # 学習可能なパラメータの定義
        self.b = nn.Parameter(torch.rand(num_nodes))
        self.w = nn.Parameter(torch.rand(num_edges, num_nodes))

        # b, w を 1 に固定した場合 (BP と等価)
        # self.b = torch.ones(num_nodes)
        # self.w = torch.ones(num_edges, num_nodes)
    
    def forward(self, l_v, h, s_c, iterations):
        num_nodes = l_v.size(0)
        num_edges = s_c.size(0)
        
        # メッセージ初期化
        mu_v_to_c = torch.zeros(num_nodes, num_edges, dtype=torch.float32)
        mu_c_to_v = torch.zeros(num_edges, num_nodes, dtype=torch.float32)
        
        # メッセージ伝播ループ
        for t in range(iterations):
            # メッセージ伝播 v -> c
            # print(self.w)
            # print(mu_c_to_v)
            # print('mu * w')
            # print(torch.matmul(mu_c_to_v, self.w))
            # print('l_v * b_v')
            # print(l_v.unsqueeze(1).size())
            # print(self.b_v)
            # print(l_v.unsqueeze(1) * self.b_v)
            # print(torch.cat([self.b_v.unsqueeze(1), self.b_v.unsqueeze(1)], dim=1).size())
            mu_v_to_c_new  = torch.zeros(num_nodes, num_edges, dtype=torch.float32)
            
            for v in range(num_nodes):
                for c in range(num_edges):
                    mu_v_to_c_new[v, c] = self.update_message_v_to_c(v, c, l_v, h, mu_c_to_v, self.w, self.b)
            
            # メッセージ伝播 c -> v
            # print(s_c.unsqueeze(1))
            # print(torch.ones(num_nodes, num_edges, dtype=torch.float32))
            mu_c_to_v_new = torch.zeros(num_edges, num_nodes, dtype=torch.float32)

            for c in range(num_edges):
                # print(c)
                for v in range(num_nodes):
                    mu_c_to_v_new[c, v] = self.update_message_c_to_v(c, v, h, s_c, mu_v_to_c)
            
            mu_v_to_c = mu_v_to_c_new
            mu_c_to_v = mu_c_to_v_new
        
        # 最終的なビリーフ計算
        # print(l_v)
        # print(self.b_v)
        mu_v = torch.zeros(num_nodes, dtype=torch.float32)

        for v in range(num_nodes):
            mu_v[v] = self.merginalization(v, l_v, h, mu_c_to_v, self.w, self.b)
        
        sigma_mu_v = self.hf_sigmoid(mu_v)
        
        return sigma_mu_v
    
    def update_message_v_to_c(self, v, c, l_v, h, mu_c_to_v, w, b):
        num_edges = h.size(0)

        message = l_v[v] * b[v]
        h_row = h[:, v]

        for c_prime in range(num_edges):
            if h_row[c_prime] == 1 and c_prime != c:
                # print(c)
                # print('c_prime')
                # print(c_prime)
                # print('v')
                # print(v)
                message += mu_c_to_v[c_prime, v] * w[c_prime, v]

        return message

    def update_message_c_to_v(self, c, v, h, s_c, mu_v_to_c):
        num_nodes = h.size(1)

        message = 1
        h_line = h[c, :]

        # print(h[c, :])
        for v_prime in range(num_nodes):
            if h_line[v_prime] == 1 and v_prime != v:
                message *= torch.tanh(mu_v_to_c[v_prime, c] / 2)
        
        message = (-1) ** s_c[c] * 2 * torch.atanh(message)

        return message
    
    def merginalization(self, v, l_v, h, mu_c_to_v, w, b):
        num_edges = h.size(0)

        message = l_v[v] * b[v]
        h_row = h[:, v]
        # print('h_row')
        # print(h_row)

        for c in range(num_edges):
            if h_row[c] == 1:
                message += mu_c_to_v[c, v] * w[c, v]

        return message
    
    def hf_sigmoid(self, x):
        return 1 / (torch.exp(x) + 1)


# 使用例
seed = np.random.RandomState(0)

k, n = H_tensor.shape
print("Number of coded bits:", n)

iterations = 5
l_v = torch.tensor([3.74899243611] * n)

model = BeliefPropagation(num_nodes=n, num_edges=k)

# sigma_mu_v = model(l_v=l_v, h=H_tensor, s_c=syndrome_tensor, iterations=iterations)
# print(sigma_mu_v)
# exit()

optimizer = optim.Adam(model.parameters(), lr=0.01)  # 最適化手法

O = torch.zeros((n // 2, n // 2))
I = torch.eye(n // 2)

OI = torch.cat((O, I), 1)
IO = torch.cat((I, O), 1)

M = torch.cat((OI, IO), 0)

print('M:')
print(M)
print(M.dtype)

# print('Error pattern:', e_v_tensor.to(torch.int32).numpy())
# sigma_mu_v = model(l_v=l_v, h=H_tensor, s_c=s_c_tensor, iterations=iterations)
# print('sigma_mu_v:')
# print(sigma_mu_v.numpy())
# exit()

print('H_tensor')
print(H_tensor)

print('H_v_tensor')
print(H_v_tensor)

for epoch in range(100):
    optimizer.zero_grad()
    # print(l_v[2])
    sigma_mu_v = model(l_v=l_v, h=H_tensor, s_c=syndrome_tensor, iterations=iterations)

    # print('H_v * M')
    # print(torch.matmul(H_v_tensor, M).shape)

    # print('x')
    # print(torch.matmul(torch.matmul(H_v_tensor, M), e_v + sigma_mu_v).shape)

    # print(e_v)
    # print(sigma_mu_v)

    loss = torch.sum(torch.sin((torch.pi / 2) * (torch.matmul(torch.matmul(H_tensor, M), e_v + sigma_mu_v))))
    # criterion = nn.BCEWithLogitsLoss()  # 二値クロスエントロピー損失関数
    # loss = criterion(sigma_mu_v, e_v)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")
        print(sigma_mu_v)


# 学習後のパラメータの値
# print("Learned parameters:")
# print("b_v:", model.b.data)
# print("w:", model.w.data)

print('Final loss:', loss.item())
print('Error pattern:', e_v_tensor.to(torch.int32).numpy())

decoded_message = np.where(sigma_mu_v < 0.5, 0, 1)
print("Decoded message:", decoded_message)