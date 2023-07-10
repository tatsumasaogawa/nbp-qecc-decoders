# test

import torch
import torch.nn as nn
import torch.optim as optim

class BeliefPropagation(nn.Module):
    def __init__(self, num_nodes, num_edges):
        super(BeliefPropagation, self).__init__()
        
        # 学習可能なパラメータの定義
        self.b_v = nn.Parameter(torch.rand(num_nodes))
        self.w = nn.Parameter(torch.rand(num_edges, num_edges))
    
    # とりあえず行列のサイズを合わせた (7/11)
    def forward(self, l_v, s_c, iterations):
        num_nodes = l_v.size(0)
        num_edges = self.w.size(0)
        
        # メッセージ初期化
        mu_c_to_v = torch.zeros(num_nodes, num_edges, dtype=torch.float32)
        mu_v_to_c = torch.zeros(num_nodes, num_edges, dtype=torch.float32)
        
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
            mu_v_to_c = l_v.unsqueeze(1) * torch.cat([self.b_v.unsqueeze(1), self.b_v.unsqueeze(1)], dim=1) + torch.matmul(mu_c_to_v, self.w)
            
            # メッセージ伝播 c -> v
            # print(s_c.unsqueeze(1))
            # print(torch.ones(num_nodes, num_edges, dtype=torch.float32))
            a_mu_c_to_v = s_c.unsqueeze(0) * torch.ones(num_nodes, num_edges, dtype=torch.float32)
            mu_c_to_v = a_mu_c_to_v
        
        # 最終的なビリーフ計算
        # print(l_v)
        # print(self.b_v)
        mu_v = l_v * self.b_v + torch.sum(torch.matmul(mu_c_to_v, self.w), dim=1)
        
        return mu_v

# 使用例
num_nodes = 3
num_edges = 2
l_v = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32)
s_c = torch.tensor([0.7, 0.8], dtype=torch.float32)
iterations = 5
e_v = torch.tensor([1, 0, 1], dtype=torch.float32)  # ターゲットの値

model = BeliefPropagation(num_nodes, num_edges)
criterion = nn.BCEWithLogitsLoss()  # 二値クロスエントロピー損失関数
optimizer = optim.Adam(model.parameters(), lr=0.01)  # 最適化手法

for epoch in range(100):
    optimizer.zero_grad()
    mu_v = model(l_v, s_c, iterations)
    print(mu_v)
    loss = criterion(mu_v, e_v)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

# 学習後のパラメータの値
print("Learned parameters:")
print("b_v:", model.b_v.data)
print("w:", model.w.data)
