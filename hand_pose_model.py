import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch_geometric

# num_classes = 10
class HandPoseGNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(HandPoseGNN, self).__init__()
        self.conv1 = GCNConv(2, 64)  # 2 features per node (x, y)
        self.conv2 = GCNConv(64, 128)
        self.fc = torch.nn.Linear(128, num_classes)  # num_classes: number of labels

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        # Global pooling (e.g., mean pooling)
        x = torch_geometric.nn.global_mean_pool(x, data.batch)

        return F.log_softmax(self.fc(x), dim=1)

# Instantiate the model
# model = HandPoseGNN(10)

