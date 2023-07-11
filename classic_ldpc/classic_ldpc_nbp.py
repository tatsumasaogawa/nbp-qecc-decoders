import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class BeliefPropagation(nn.Module):
    def __init__(self, num_nodes, num_edges):
        super(BeliefPropagation, self).__init__()
        
        # 学習可能なパラメータの定義
        self.b = nn.Parameter(torch.rand(num_nodes))
        self.w = nn.Parameter(torch.rand(num_nodes, num_edges))
    
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
        message = l_v[v] * b[v]

        for c_prime in h[:, v]:
            if c_prime != c:
                # print(c_prime, v)
                message += mu_c_to_v[c_prime, v] * w[v, c_prime]
        
        message = F.tanh(1 / 2 * message)

        return message

    def update_message_c_to_v(self, c, v, h, s, mu_v_to_c):
        message = 1

        # print(h[c, :])
        for v_prime in h[c, :].tolist():
            if v_prime != v:
                message *= F.tanh(mu_v_to_c[v_prime, c] / 2)
        
        message = (-1) ** s[c] * 2 * torch.atanh(message)

        return message
    
    def merginalization(self, v, l_v, h, mu_c_to_v, w, b):
        message = l_v[v] * b[v]

        for c in h[:, v]:
            message += mu_c_to_v[c, v] * w[v, c]
        
        message = F.tanh(1 / 2 * message)

        return message
    
    def hf_sigmoid(self, x):
        return 1 / (torch.exp(x) + 1)

# 使用例
num_nodes = 3
num_edges = 2
l_v = torch.tensor([4.6, 4.6, 4.6], dtype=torch.float32)
s_c = torch.tensor([1, 0], dtype=torch.float32)
iterations = 5
e_v = torch.tensor([1, 0, 1], dtype=torch.float32)  # ターゲットの値

h = torch.tensor([[1, 1, 0],  # パリティチェック行列
                  [1, 0, 1]], dtype=torch.int32)

model = BeliefPropagation(num_nodes, num_edges)
criterion = nn.BCEWithLogitsLoss()  # 二値クロスエントロピー損失関数
optimizer = optim.Adam(model.parameters(), lr=0.01)  # 最適化手法

for epoch in range(100):
    optimizer.zero_grad()
    # print(l_v[2])
    sigma_mu_v = model(l_v, h, s_c, iterations)
    # print(mu_v)
    loss = criterion(sigma_mu_v, e_v)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

# 学習後のパラメータの値
print("Learned parameters:")
print("b_v:", model.b.data)
print("w:", model.w.data)

print('final solution:')
print(sigma_mu_v.data)