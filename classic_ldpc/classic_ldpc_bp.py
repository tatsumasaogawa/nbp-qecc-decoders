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

# 使用例
n = 30
d_v = 2
d_c = 3
seed = np.random.RandomState(42)
##################################################################
# First we create an LDPC code i.e a pair of decoding and coding matrices
# H and G. H is a regular parity-check matrix with d_v ones per row
# and d_c ones per column

H, G = make_ldpc(n, d_v, d_c, seed=seed, systematic=True, sparse=True)
# print('H')
# print(H)
H_tensor = torch.from_numpy(H.astype(np.int32))
G_tensor = torch.from_numpy(G.astype(np.int32))

k, n = H.shape
print("Number of coded bits:", k)

v, y = encode_random_message(G, 6) # v: オリジナル, y: ノイズの乗った符号
y = np.where(y > 0, 0, 1)
v_tensor = torch.from_numpy(v.astype(np.int32))
y_tensor = torch.from_numpy(abs(y).astype(np.int32))
l_v = torch.tensor([0.023] * n)

# print(H_tensor)
# print(y_tensor)
s_c_tensor = torch.matmul(H_tensor, y_tensor)
s_c_tensor %= 2
print('s_c_tensor')
print(s_c_tensor)

y_true_tensor = abs(torch.matmul(G_tensor, v_tensor))
y_true_tensor %= 2
print('y')
print(y_tensor)
print('y_true')
print(y_true_tensor)
# exit()

e_v_tensor = abs(y_true_tensor - y_tensor).to(torch.float32)
print('e_v_tensor')
print(e_v_tensor)
# exit()

# タナーグラフの構築
variable_nodes = []
for i in range(n):
    variable_nodes.append(VariableNode(i))

check_nodes = []
for i in range(k):
    check_nodes.append(CheckNode(i))

for i in range(k):
    for j in range(n):
        if H[i, j] == 1:
            variable_nodes[j].connect_check_node(check_nodes[i])
            check_nodes[i].connect_variable_node(variable_nodes[j])

syndrome = s_c_tensor.tolist()
log_error_rate = 3.74899243611

# タナーグラフ上でのBPアルゴリズムの実行と復号
tanner_graph = TannerGraph(variable_nodes, check_nodes)
tanner_graph.belief_propagation(syndrome)
decoded_message, error_rate = tanner_graph.decode()

# print("y_tensor:", y_tensor)
# print("y_true_tensor:", y_true_tensor)
print('Syndrome pattern:', np.array(syndrome))
print("Error pattern:", e_v_tensor.to(torch.int32).numpy())

print("Decoded message:", np.array(decoded_message))
print("Error rate:", np.array(error_rate))

# for i in range(len(check_nodes)):
#     print(check_nodes[i].message_old)
#     print(check_nodes[i].message_new)