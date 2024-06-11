import torch
import torch.nn.functional as F
from torch.nn import Linear
import numpy as np
from torch_geometric.nn import GATv2Conv
import json 
import torch 
import numpy as np 
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from rich.progress import track 
import torch
from torch_geometric.data import DataLoader, Data
import numpy as np
import torch.nn as nn 


hidden = 128 
alphabet = "ACDEFGHIKLMNPQRSTVWY"
itos = {i: letter for i, letter in enumerate(alphabet)}
stoi = {v: k for k, v in itos.items()}    
vocab_size = len(set(alphabet))

def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features

class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, period_range=[2,1000]):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.period_range = period_range 

    def forward(self, E_idx):
        # i-j
        N_batch = E_idx.size(0)
        N_nodes = E_idx.size(1)
        N_neighbors = E_idx.size(2)
        ii = torch.arange(N_nodes, dtype=torch.float32).view((1, -1, 1))
        d = (E_idx.float() - ii).unsqueeze(-1)
        # Original Transformer frequencies
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, dtype=torch.float)
            * -(np.log(10000.0) / self.num_embeddings)
        )
        angles = d * frequency.view((1,1,1,-1))
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E


class MyProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=32, augment_eps=0., num_chain_embeddings=16):
        """ Extract protein features """
        super(MyProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        node_in, edge_in = 6, num_positional_embeddings + num_rbf*25
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=False)
        self.norm_edges = nn.LayerNorm(edge_features)

    def _dist(self, X, mask, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)
        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        sampled_top_k = self.top_k
        D_neighbors, E_idx = torch.topk(D_adjust, np.minimum(self.top_k, X.shape[1]), dim=-1, largest=False)
        return D_neighbors, E_idx

    def _rbf(self, D):
        device = D.device
        D_min, D_max, D_count = 2., 22., self.num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count, device=device)
        D_mu = D_mu.view([1,1,1,-1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
        return RBF

    def _get_rbf(self, A, B, E_idx):
        D_A_B = torch.sqrt(torch.sum((A[:,:,None,:] - B[:,None,:,:])**2,-1) + 1e-6) #[B, L, L]
        D_A_B_neighbors = gather_edges(D_A_B[:,:,:,None], E_idx)[:,:,:,0] #[B,L,K]
        RBF_A_B = self._rbf(D_A_B_neighbors)
        return RBF_A_B

    def forward(self, X, mask):
        b = X[:,:,1,:] - X[:,:,0,:]
        c = X[:,:,2,:] - X[:,:,1,:]
        a = torch.cross(b, c, dim=-1)
        Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + X[:,:,1,:]
        Ca = X[:,:,1,:]
        N = X[:,:,0,:]
        C = X[:,:,2,:]
        O = X[:,:,3,:]
 
        D_neighbors, E_idx = self._dist(Ca, mask)
        E_positional = self.embeddings(E_idx)

        # calculate pairwise distances and encode with RBF 
        RBF_all = []
        RBF_all.append(self._rbf(D_neighbors)) #Ca-Ca
        RBF_all.append(self._get_rbf(N, N, E_idx)) #N-N
        RBF_all.append(self._get_rbf(C, C, E_idx)) #C-C
        RBF_all.append(self._get_rbf(O, O, E_idx)) #O-O
        RBF_all.append(self._get_rbf(Cb, Cb, E_idx)) #Cb-Cb
        RBF_all.append(self._get_rbf(Ca, N, E_idx)) #Ca-N
        RBF_all.append(self._get_rbf(Ca, C, E_idx)) #Ca-C
        RBF_all.append(self._get_rbf(Ca, O, E_idx)) #Ca-O
        RBF_all.append(self._get_rbf(Ca, Cb, E_idx)) #Ca-Cb
        RBF_all.append(self._get_rbf(N, C, E_idx)) #N-C
        RBF_all.append(self._get_rbf(N, O, E_idx)) #N-O
        RBF_all.append(self._get_rbf(N, Cb, E_idx)) #N-Cb
        RBF_all.append(self._get_rbf(Cb, C, E_idx)) #Cb-C
        RBF_all.append(self._get_rbf(Cb, O, E_idx)) #Cb-O
        RBF_all.append(self._get_rbf(O, C, E_idx)) #O-C
        RBF_all.append(self._get_rbf(N, Ca, E_idx)) #N-Ca
        RBF_all.append(self._get_rbf(C, Ca, E_idx)) #C-Ca
        RBF_all.append(self._get_rbf(O, Ca, E_idx)) #O-Ca
        RBF_all.append(self._get_rbf(Cb, Ca, E_idx)) #Cb-Ca
        RBF_all.append(self._get_rbf(C, N, E_idx)) #C-N
        RBF_all.append(self._get_rbf(O, N, E_idx)) #O-N
        RBF_all.append(self._get_rbf(Cb, N, E_idx)) #Cb-N
        RBF_all.append(self._get_rbf(C, Cb, E_idx)) #C-Cb
        RBF_all.append(self._get_rbf(O, Cb, E_idx)) #O-Cb
        RBF_all.append(self._get_rbf(C, O, E_idx)) #C-O
        RBF_all = torch.cat(tuple(RBF_all), dim=-1)

        #offset = residue_idx[:,:,None]-residue_idx[:,None,:]
        #ffset = gather_edges(offset[:,:,:,None], E_idx)[:,:,:,0] #[B, L, K]
        #d_chains = ((chain_labels[:, :, None] - chain_labels[:,None,:])==0).long() #find self vs non-self interaction
        #E_chains = gather_edges(d_chains[:,:,:,None], E_idx)[:,:,:,0]
        #E_positional = self.embeddings(offset.long(), E_chains)
        E = torch.cat((E_positional, RBF_all), -1).to(torch.float)
        E = self.edge_embedding(E)
        E = self.norm_edges(E)
        return E, E_idx


features = MyProteinFeatures(hidden, hidden)
print("Number of trainable params in features", sum(p.numel() for p in features.parameters() if p.requires_grad))


def transform_for_pyg(pkg):
    with torch.no_grad():
        #X, S, mask, lengths = featurize([pkg], torch.device("cpu"))
        #X, S, mask, lengths, chain_M, residue_idx, mask_self, chain_encoding_all = featurize([pkg])
        xyz = pkg["coords"]
        #print(xyz)
        name = pkg["name"]
        letter = name[5]
        X = np.stack([xyz[c] for c in ["N", "CA", "C", "O"]], 1) #[chain_length,4,3]
        #print(X.shape)
        #print(f"{x.shape=}")
        #lengths = np.array([len(pkg['seq'])], dtype=np.int32) 
        mask = torch.tensor(np.isfinite(np.sum(X,(1, 2))), dtype=torch.float).unsqueeze(0)
        #print(mask.shape)
        # print(lengths)
        L = len(pkg["seq"])
        X = torch.tensor(X).unsqueeze(0)
        
        E, E_idx = features.forward(X, mask) 
        V = torch.zeros((1, L, 128))
        tokens = torch.tensor(list(stoi[s] for s in pkg["seq"])).detach() 
        edges_1 = []
        edges_2 = [] 
        edge_index = 0 
        edge_attr = []
    
        for seq_pos in range(E.shape[1]):
            neighbors = E_idx[0, seq_pos]
            for nbr_idx, neighbor in enumerate(neighbors):
                # the edge is between seq_pos, and neighbor 
                edges_1.append(seq_pos)
                edges_2.append(neighbor)

                # get the data from `E`, recall E is [batch, seq, k, features] and you just want [features] for a particular one 
                # this one, in fact 
                my_edge_features = E[0, seq_pos, nbr_idx].detach() 
                edge_attr.append(my_edge_features)
                edge_index += 1

        edge_attr = torch.tensor(np.stack(edge_attr), dtype=torch.float)
        edges = torch.tensor((edges_1, edges_2)).detach()
        data = Data(x=V[0].detach(), edge_attr=edge_attr, edge_index=edges, y=tokens)
        return data 


max_samples = 10
data = []

count = 0 
with torch.no_grad():
    with open("data/cath/chain_set.jsonl") as fn:
        for line in track(fn.readlines()):
            try:
                pkg = json.loads(line)
                pkg = transform_for_pyg(pkg)
                data.append(pkg.to(torch.device("mps")))
                count +=1 
            except Exception as e:
                if isinstance(e, KeyError):
                    pass
                else: 
                    raise Exception
            if count >= max_samples:
                break 
            
n1 = int(count * 0.8)
n2 = int(count * 0.9)

train_data = data[:n1]
val_data = data[n1:n2]
test_data = data[n2:]

print("len datasets", len(train_data), len(val_data), len(test_data))


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

        print(f"{loss=:.2f} tokens={total_samples}")
    
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
num_node_features = 128  
num_edge_features = 128
hidden_dim = 128
num_classes = 20
num_layers = 3
batch_size = 32
num_epochs = 100
learning_rate = 0.001

# Initialize model, optimizer, and loss function
model = GraphTransformerNetwork(num_node_features, num_edge_features, hidden_dim, num_classes, num_layers).to(torch.device("mps"))
n_params = sum(p.numel() for p in model.parameters())
print(f"Training a model with {n_params} parameters")
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Main training loop
for epoch in range(num_epochs):
    train_loss, train_accuracy, train_perplexity = train(model, train_data_loader, optimizer, criterion)
    val_loss, val_accuracy, val_perplexity = evaluate(model, val_data_loader, criterion)
    
    print("'---------------------------------")
    print(f'Epoch {epoch+1:03d}')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Train Perplexity: {train_perplexity:.2f}')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, Val Perplexity: {val_perplexity:.2f}')
    print("'---------------------------------")

