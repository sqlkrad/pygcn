import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
    

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, sparse_input=False):
        if sparse_input:
            x[1] = F.dropout(x[1], self.dropout, training=self.training)
            x = torch.sparse.FloatTensor(*x).cuda()
        else:
            x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj, sparse_input))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
