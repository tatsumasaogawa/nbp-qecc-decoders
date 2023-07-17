# https://github.com/samn33/qlazy/blob/master/example/py/SurfaceCode/error_correction.py

from qlazy import QState
import numpy as np

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
    qs_fin.x(0)  # bit flip error at #7
    # qs_fin.z(7).x(7)  # bit and phase flip error at #7

    syndrome = measure_syndrome(qs_fin, mval_list)
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
from pyldpc.encoder import encode_random_message
from pyldpc.decoder import get_message

np.set_printoptions(suppress=True)


class TannerGraph:
    def __init__(self, variable_nodes, check_nodes):
        self.variable_nodes = variable_nodes
        self.check_nodes = check_nodes

    def belief_propagation(self, syndrome, max_iterations=50):
        for i in range(max_iterations):
            for check_node in self.check_nodes:
                for variable_node in check_node.connected_variable_nodes:
                    self.update_variable_node(variable_node, check_node)

            for variable_node in self.variable_nodes:
                for check_node in variable_node.connected_check_nodes:
                    self.update_check_node(check_node, variable_node, syndrome)
            
            if self.check_convergence() == True:
                print(i)
                return
            
            # ma_check_node_message = 0 # 正規化
            # for check_node in self.check_nodes:
            #     ma_check_node_message = max(ma_check_node_message, abs(check_node.message_new))
            # if ma_check_node_message == 0:
            #     ma_check_node_message = 1
            
            print(i)
            for check_node in self.check_nodes:
                # check_node.message_new /= ma_check_node_message
                check_node.messages_old = check_node.messages_new
                print('c' + str(check_node.index))
                print(check_node.messages_old)
            
            # ma_variable_node_message = 0 # 正規化
            # for variable_node in self.variable_nodes:
            #     ma_variable_node_message = max(ma_variable_node_message, abs(variable_node.message_new))
            # if ma_variable_node_message == 0:
            #     ma_variable_node_message = 1
            
            for variable_node in self.variable_nodes:
                # variable_node.message_new /= ma_variable_node_message
                variable_node.messages_old = variable_node.messages_new
                print('v' + str(variable_node.index))
                print(variable_node.messages_old)
            print()

    def update_variable_node(self, variable_node, check_node):
        messages = []
        for neighbor_check_node in variable_node.connected_check_nodes:
            if neighbor_check_node.index != check_node.index:
                messages.append(neighbor_check_node.messages_old[str(neighbor_check_node.index) + ',' + str(variable_node.index)])
        # variable_node.message = sum(messages) % 2
        variable_node.messages_new[str(variable_node.index) + ',' + str(check_node.index)] = log_error_rate + sum(messages)        

    def update_check_node(self, check_node, variable_node, syndrome):
        messages = []
        for neighbor_variable_node in check_node.connected_variable_nodes:
            if neighbor_variable_node.index != variable_node.index:
                messages.append(np.tanh(neighbor_variable_node.messages_old[str(neighbor_variable_node.index) + ',' + str(check_node.index)] / 2))
        # check_node.message = sum(messages) % 2
        check_node.messages_new[str(check_node.index) + ',' + str(variable_node.index)] = (-1) ** syndrome[check_node.index] * 2 * np.arctanh(np.prod(messages, axis=0))
        # print(messages)

    def decode(self):
        decoded_message = []
        probabilities = []
        for variable_node in self.variable_nodes:
            messages = []
            for neighbor_check_node in variable_node.connected_check_nodes:
                messages.append(neighbor_check_node.messages_new[str(neighbor_check_node.index) + ',' + str(variable_node.index)])
            variable_node.messages_new[str(variable_node.index)] = log_error_rate + sum(messages)

            probability = self.hf_sigmoid(variable_node.messages_new[str(variable_node.index)])
            probabilities.append(probability)

            decoded_bit = 0 if variable_node.messages_new[str(variable_node.index)] > 0 else 1
            decoded_message.append(decoded_bit)
        return decoded_message, probabilities

    def hf_sigmoid(self, x):
        return 1 / (np.exp(x) + 1)
    
    def check_convergence(self):
        for variable_node in self.variable_nodes:
            messages_v_to_c_old_list = sorted(variable_node.messages_old.values())
            messages_v_to_c_new_list = sorted(variable_node.messages_new.values())

            if sum(np.isclose(messages_v_to_c_old_list, messages_v_to_c_new_list)) < len(messages_v_to_c_old_list):
                return False

        for check_node in self.check_nodes:
                messages_c_to_v_old_list = sorted(check_node.messages_old.values())
                messages_c_to_v_new_list = sorted(check_node.messages_new.values())

                if sum(np.isclose(messages_c_to_v_old_list, messages_c_to_v_new_list)) < len(messages_c_to_v_old_list):
                    return False
        
        return True

class VariableNode:
    def __init__(self, index):
        self.index = index
        self.connected_check_nodes = []
        self.messages_old = {}
        self.messages_new = {}


    def connect_check_node(self, check_node):
        self.connected_check_nodes.append(check_node)
        self.messages_old[str(self.index) + ',' + str(check_node.index)] = 0

class CheckNode:
    def __init__(self, index):
        self.index = index
        self.connected_variable_nodes = []
        self.messages_old = {}
        self.messages_new = {}

    def connect_variable_node(self, variable_node):
        self.connected_variable_nodes.append(variable_node)
        self.messages_old[str(self.index) + ',' + str(variable_node.index)] = 0

# タナーグラフの構築
variable_nodes = []
for i in range(4 * N ** 2):
    variable_nodes.append(VariableNode(i))

check_nodes = []
for i in range(2 * N ** 2):
    check_nodes.append(CheckNode(i))

for i in range(2 * N ** 2):
    for j in range(4 * N ** 2):
        if H[i][j] == 1:
            variable_nodes[j].connect_check_node(check_nodes[i])
            check_nodes[i].connect_variable_node(variable_nodes[j])

log_error_rate = 3.74899243611

# タナーグラフ上でのBPアルゴリズムの実行と復号
tanner_graph = TannerGraph(variable_nodes, check_nodes)
tanner_graph.belief_propagation(syndrome)
decoded_message, error_rate = tanner_graph.decode()

# print("y_tensor:", y_tensor)
# print("y_true_tensor:", y_true_tensor)
print('Syndrome pattern:', np.array(syndrome))

print("Decoded message:", np.array(decoded_message))
print("Error rate:", np.array(error_rate))

# for i in range(len(check_nodes)):
#     print(check_nodes[i].message_old)
#     print(check_nodes[i].message_new)