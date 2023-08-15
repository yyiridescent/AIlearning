import torch
import torch.nn as nn


def GRU(input,initial_states,w_ih,w_hh,b_ih,b_hh):
    pre_h=initial_states
    batch_size,T,i_size=input.shape
    h_size=w_ih.shape[0] // 3

    #权重扩维，复制成batch_size倍
    batch_w_ih=w_ih.unsqueeze(0).tile(batch_size,1,1)
    batch_w_hh = w_hh.unsqueeze(0).tile(batch_size, 1, 1)

    output=torch.zeros(batch_size,T,h_size)#GRU输出的状态序列

    for t in range(T):
        x=input[:,t,:]#t时刻的gru cell的输入特征向量 [batch_size,i_size]
        w_times_x=torch.bmm(batch_w_ih,x.unsqueeze(-1))  #[batch_size,3*h_size,1]
        w_times_x=w_times_x.squeeze(-1)

        w_times_h_pre = torch.bmm(batch_w_hh, pre_h.unsqueeze(-1))  # [batch_size,3*h_size,1]
        w_times_h_pre = w_times_h_pre.squeeze(-1)

        #重置门
        r_t=torch.sigmoid(w_times_x[:,:h_size]+w_times_h_pre[:,:h_size]+b_ih[:h_size]+b_hh[:h_size])
        #更新门
        z_t=torch.sigmoid(w_times_x[:,:h_size:2*h_size]+w_times_h_pre[:,:h_size:2*h_size]+b_ih[h_size:2*h_size]+b_hh[h_size:2*h_size])
        #候选状态
        n_t=torch.tanh(w_times_x[:,2*h_size:3*h_size]+b_ih[2*h_size:3*h_size]+r_t*(w_times_h_pre[:,2*h_size:3*h_size]+b_hh[2*h_size:3*h_size]))
        pre_h=(1-z_t)*n_t+z_t*pre_h
        output[:,t,:]=pre_h


    return output,pre_h

#test
batch_size,T,i_size,h_size=32,3,4,5
input=torch.randn(batch_size,T,i_size)
h_0=torch.randn(batch_size,h_size)

gru=nn.GRU(i_size,h_size,batch_first=True)
output,h_final=gru(input,h_0.unsqueeze(0))

print(output)
for k,v in gru.named_parameters():
    print(k,v.shape)

output_cus,h_final_cus=GRU(input,h_0,gru.weight_ih_l0,gru.weight_hh_l0,gru.bias_ih_l0,gru.bias_hh_l0)

print(torch.allclose(h_final,h_final_cus))