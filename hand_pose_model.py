import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, BatchNorm
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


class EnhancedHandPoseGNN(torch.nn.Module):
    def __init__(self, num_features=2, num_classes=31):
        super(EnhancedHandPoseGNN, self).__init__()

        # Define the architecture
        self.conv1 = GCNConv(num_features, 128)
        self.bn1 = BatchNorm(128)
        self.conv2 = GCNConv(128, 256)
        self.bn2 = BatchNorm(256)
        self.conv3 = GATConv(256, 512)  # Using GAT layer for additional expressive power
        self.bn3 = BatchNorm(512)

        self.fc1 = torch.nn.Linear(512, 256)  # Additional fully connected layers
        self.fc2 = torch.nn.Linear(256, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # First GCN layer with ReLU and BatchNorm
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Second GCN layer with ReLU and BatchNorm
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # GAT layer with BatchNorm
        x = F.elu(self.conv3(x, edge_index))  # ELU can be used for a change
        x = self.bn3(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Global mean pooling
        x = global_mean_pool(x, data.batch)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return F.log_softmax(x, dim=1)


