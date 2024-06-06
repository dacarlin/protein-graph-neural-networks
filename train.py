import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import GCNConv, GatedGraphConv, GATConv
from torch_geometric.nn import global_add_pool
import numpy as np
from sklearn.metrics import accuracy_score
from torch_geometric.nn import GATv2Conv
import json 

import torch 
import numpy as np 
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.transforms import PointPairFeatures, KNNGraph
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from ingraham.struct2seq.protein_features import ProteinFeatures
from ingraham.struct2seq.data import StructureDataset, StructureLoader
from ingraham.experiments.utils import featurize
from ingraham.struct2seq.struct2seq import Struct2Seq

from rich.progress import track 

#device = torch.device("mps")

# %%
neighbors = 16 
hidden = 128 

alphabet = "ACDEFGHIKLMNPQRSTVWY"
itos = {i: letter for i, letter in enumerate(alphabet)}
stoi = {v: k for k, v in itos.items()}    
vocab_size = len(set(alphabet))

features = ProteinFeatures(hidden, hidden, num_positional_embeddings=16, num_rbf=16, top_k=neighbors, features_type="full")

def transform_for_pyg(pkg):
    X, S, mask, lengths = featurize([pkg], torch.device("cpu"))
    # print(lengths)
    V, E, E_idx = features.forward(X, lengths, mask)
    tokens = torch.tensor(list(stoi[s] for s in pkg["seq"])).detach() 
    edges_1 = []
    edges_2 = [] 
    edge_index = 0 
    edge_attr = []
    for seq_pos in range(S.shape[1]):
        neighbors = E_idx[0, seq_pos]
        for nbr_idx, neighbor in enumerate(neighbors):
            # the edge is between seq_pos, and neighbor 
            edges_1.append(seq_pos)
            edges_2.append(neighbor)

            # get the data from `E`, recall E is [batch, seq, k, features] and you just want [features] for a particular one 
            # this one, in fact 
            my_edge_features = E[0, seq_pos, nbr_idx]
            edge_attr.append(my_edge_features)
            edge_index += 1

    edge_attr = torch.tensor(np.stack(edge_attr), dtype=torch.float)
    edges = torch.tensor((edges_1, edges_2)).detach()
    data = Data(x=V[0].detach(), edge_attr=edge_attr, edge_index=edges, y=tokens)
    return data 


max_samples = 200
data = []

count = 0 
with torch.no_grad():
    with open("data/cath/chain_set.jsonl") as fn:
        
        for line in track(fn.readlines()):
            try:
                pkg = json.loads(line)
                pkg = transform_for_pyg(pkg)
                data.append(pkg)
                count +=1 
            except:
                pass
            if count >= max_samples:
                break 
n1 = int(count * 0.8)
n2 = int(count * 0.9)

train_data = data[:n1]
val_data = data[n1:n2]
test_data = data[n2:]

print("splits sizes")
len(train_data), len(val_data), len(test_data)


class GraphTransformerLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, edge_features_dim, heads=4):
        super(GraphTransformerLayer, self).__init__()
        self.edge_embedding = Linear(edge_features_dim, in_channels)  # Embed edge features
        self.conv = GATv2Conv(in_channels, out_channels // heads, heads=heads, edge_dim=edge_features_dim, concat=True)
        self.lin = Linear(out_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        #x = self.node_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)
        x = self.conv(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = self.lin(x)
        return x

class GraphTransformerNetwork(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim, num_classes, num_layers=3):
        super(GraphTransformerNetwork, self).__init__()
        
        self.encoder_layers = torch.nn.ModuleList()
        self.decoder_layers = torch.nn.ModuleList()

        self.encoder_layers.append(GraphTransformerLayer(num_node_features, hidden_dim, num_edge_features))
        for _ in range(num_layers - 1):
            self.encoder_layers.append(GraphTransformerLayer(hidden_dim, hidden_dim, num_edge_features))

        self.decoder_layers.append(GraphTransformerLayer(hidden_dim, hidden_dim, num_edge_features))
        for _ in range(num_layers - 1):
            self.decoder_layers.append(GraphTransformerLayer(hidden_dim, hidden_dim, num_edge_features))

        self.linear_out = Linear(hidden_dim, num_classes)
    
    def forward(self, x, edge_index, edge_attr):
        for layer in self.encoder_layers:
            x = layer(x, edge_index, edge_attr)

        for layer in self.decoder_layers:
            x = layer(x, edge_index, edge_attr)

        return self.linear_out(x)


def train(model, data_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    perplexity_sum = 0
    
    for data in data_loader:
        data = data.to(torch.device("mps"))
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out, data.y)
        loss.backward()        
        optimizer.step()
        
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        total_correct += pred.eq(data.y).sum().item()
        total_samples += data.num_nodes
        
        prob_scores = F.softmax(out, dim=1)
        perplexity_sum += -torch.sum(torch.log(prob_scores[range(out.size(0)), data.y])).item()

        print(f"{loss=:.2f} total_nodes={total_samples} {perplexity_sum=:.2f}")
    
    average_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples * 100
    average_perplexity = np.exp(perplexity_sum / total_samples)
    
    return average_loss, accuracy, average_perplexity

def evaluate(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    perplexity_sum = 0
    
    with torch.no_grad():
        for data in data_loader:
            data = data.to(torch.device("mps"))
            out = model(data.x, data.edge_index, data.edge_attr)
            loss = criterion(out, data.y)
            
            total_loss += loss.item()
            pred = out.argmax(dim=1)
            total_correct += pred.eq(data.y).sum().item()
            total_samples += data.num_nodes
            
            prob_scores = F.softmax(out, dim=1)
            perplexity_sum += -torch.sum(torch.log(prob_scores[range(out.size(0)), data.y])).item()

    average_loss = total_loss / len(data_loader)
    accuracy = total_correct / total_samples * 100
    average_perplexity = np.exp(perplexity_sum / total_samples)
    
    return average_loss, accuracy, average_perplexity

# Hyperparameters
num_node_features = 128  # Flattened N, CA, C, O coordinates (if we keep all dimensions, adjust accordingly)
num_edge_features = 128
hidden_dim = 128
num_classes = 20
num_layers = 3
batch_size = 32
num_epochs = 100
learning_rate = 0.001

# Initialize model, optimizer, and loss function
model = GraphTransformerNetwork(num_node_features, num_edge_features, hidden_dim, num_classes, num_layers).to(torch.device("mps"))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
#model = model.to(device)

# Assume data_list is your list of graphs constructed using construct_graph function

data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# %%
for batch in data_loader:
    break  

batch 

# %%
# with torch.no_grad():
#     model(batch.x, batch.edge_index, batch.edge_attr)

# %%
# Main training loop
for epoch in range(num_epochs):
    train_loss, train_accuracy, train_perplexity = train(model, data_loader, optimizer, criterion)
    test_loss, test_accuracy, test_perplexity = evaluate(model, test_data_loader, criterion)
    
    print(f'Epoch {epoch+1:03d}, '
          f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Train Perplexity: {train_perplexity:.2f}, '
          f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%, Test Perplexity: {test_perplexity:.2f}')


