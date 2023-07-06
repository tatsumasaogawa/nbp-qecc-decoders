import numpy as np


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
variable_nodes = [VariableNode(0), VariableNode(1), VariableNode(2)]
check_nodes = [CheckNode(0), CheckNode(1)]

variable_nodes[0].connect_check_node(check_nodes[0])
variable_nodes[0].connect_check_node(check_nodes[1])
variable_nodes[1].connect_check_node(check_nodes[0])
variable_nodes[2].connect_check_node(check_nodes[1])

check_nodes[0].connect_variable_node(variable_nodes[0])
check_nodes[0].connect_variable_node(variable_nodes[1])
check_nodes[1].connect_variable_node(variable_nodes[0])
check_nodes[1].connect_variable_node(variable_nodes[2])

syndrome = [1, 1]
log_error_rate = 4.6 # log(0.99 / 0.01)

# タナーグラフ上でのBPアルゴリズムの実行と復号
tanner_graph = TannerGraph(variable_nodes, check_nodes)
tanner_graph.belief_propagation(syndrome)
decoded_message, error_rate = tanner_graph.decode()

print("Decoded message:", decoded_message)
print("Error rate:", error_rate)

# for i in range(len(check_nodes)):
#     print(check_nodes[i].message_old)
#     print(check_nodes[i].message_new)