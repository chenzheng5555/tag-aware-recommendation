import torch as t
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

class GCN(MessagePassing):
    def __init__(self):
        super(GCN, self).__init__(aggr="add")

    def forward(self, x, edge_index):
        source, target = edge_index
        deg_s = degree(target, x.size(0), dtype=x.dtype)#出度，如果是无向图，则入度和出度相等
        deg_t = degree(target, x.size(0), dtype=x.dtype)#入度
        deg_sqrt_s = deg_s.pow(-0.5)
        deg_sqrt_t = deg_t.pow(-0.5)
        deg_sqrt_s[t.isinf(deg_sqrt_s)] = 1
        deg_sqrt_t[t.isinf(deg_sqrt_t)] = 1
        norm = deg_sqrt_s[source] * deg_sqrt_t[target]

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


if __name__ == "__main__":
    edge_index = t.tensor([[0,5],[0,3],[5,0],[3,0],[1,4],[4,1],[2,4],[4,2]], dtype=t.long)
    edge_index = edge_index.t().contiguous()
    x = t.tensor([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0],
                  [0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]], dtype=t.float)
    mode = GCN()
    x = mode.forward(x, edge_index)
    print(x)