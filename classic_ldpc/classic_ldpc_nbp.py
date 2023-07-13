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
                message *= F.tanh(mu_v_to_c[v_prime, c] / 2)
        
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
n = 30
d_v = 2
d_c = 3
seed = np.random.RandomState(0)
##################################################################
# First we create an LDPC code i.e a pair of decoding and coding matrices
# H and G. H is a regular parity-check matrix with d_v ones per row
# and d_c ones per column

H, G = make_ldpc(n, d_v, d_c, seed=seed, systematic=True, sparse=True)
H_tensor = torch.from_numpy(H.astype(np.int32))
G_tensor = torch.from_numpy(G.astype(np.int32))

k = G.shape[1]
v = np.random.randint(2, size=k)
print(v)

y = encode(G, v, 10 ** 100)
print(y)

y = np.where(y > 0, 0, 1)
print(y)


k, n = H.shape
print("Number of coded bits:", n)

v, y = encode_random_message(G, 6) # v: 送信語, y: 受信語 (注意: y の 1 は 0, -1 は 1 に対応!!)
print(y)
y = np.where(y > 0, 0, 1)

v_tensor = torch.from_numpy(v.astype(np.int32))
y_tensor = torch.from_numpy(y.astype(np.int32))
l_v = torch.tensor([3.74899243611] * n)
print('ここを見る')
print(torch.matmul(H_tensor, torch.matmul(G_tensor, v_tensor.unsqueeze(1))) % 2)

print(H_tensor.shape)
print(y_tensor.shape)
s_c_tensor = torch.matmul(H_tensor, y_tensor.unsqueeze(1))
s_c_tensor %= 2
print('s_c_tensor')
print(s_c_tensor)
print((-1) ** s_c_tensor[0])

y_true_tensor = torch.matmul(G_tensor, v_tensor)
y_true_tensor %= 2

print(y_true_tensor)
print(y_tensor)
e_v_tensor = abs(y_true_tensor - y_tensor).to(torch.float32)
print(e_v_tensor)

if sum(e_v_tensor) == 0: # y が符号語のときはつまらないので, exit()
    exit()

iterations = 5

model = BeliefPropagation(num_nodes=n, num_edges=k)
criterion = nn.BCEWithLogitsLoss()  # 二値クロスエントロピー損失関数
optimizer = optim.Adam(model.parameters(), lr=0.01)  # 最適化手法

# print('Error pattern:', e_v_tensor.to(torch.int32).numpy())
# sigma_mu_v = model(l_v=l_v, h=H_tensor, s_c=s_c_tensor, iterations=iterations)
# print('sigma_mu_v:')
# print(sigma_mu_v.numpy())
# exit()

for epoch in range(100):
    optimizer.zero_grad()
    # print(l_v[2])
    sigma_mu_v = model(l_v=l_v, h=H_tensor, s_c=s_c_tensor, iterations=iterations)

    loss = criterion(sigma_mu_v, e_v_tensor)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")


# 学習後のパラメータの値
# print("Learned parameters:")
# print("b_v:", model.b.data)
# print("w:", model.w.data)

print('Final loss:', loss.item())
print('Error pattern:', e_v_tensor.to(torch.int32).numpy())

decoded_message = np.where(sigma_mu_v < 0.5, 0, 1)
print("Decoded message:", decoded_message)