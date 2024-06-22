import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_add_pool
from torch_geometric.data import Data, DataLoader
from torch_geometric.transforms import RadiusGraph
from Bio.PDB import PDBParser
from Bio.Data import IUPACData
import numpy as np


# Constants
NUM_AMINO_ACIDS = 20
MAX_NUM_NEIGHBORS = 10
NUM_RBF = 16
MAX_DISTANCE = 20.0
NUM_DIHEDRAL_FEATURES = 4  # phi, psi, omega, and chi1
NUM_ATOM_FEATURES = 10  # Atom type, hybridization, aromaticity, etc.


# Global alphabet
aa_to_idx = {
    "ALA": 0,
    "ARG": 1,
    "ASN": 2,
    "ASP": 3,
    "CYS": 4,
    "GLN": 5,
    "GLU": 6,
    "GLY": 7,
    "HIS": 8,
    "ILE": 9,
    "LEU": 10,
    "LYS": 11,
    "MET": 12,
    "PHE": 13,
    "PRO": 14,
    "SER": 15,
    "THR": 16,
    "TRP": 17,
    "TYR": 18,
    "VAL": 19,
}
idx_to_aa = {v: k for k, v in aa_to_idx.items()}


class ProteinLigandGNN(nn.Module):
    """
    Graph Neural Network for protein sequence design, incorporating ligand information.

    This model processes both protein backbone features and ligand atom features
    to predict amino acid sequences that are compatible with the given structure
    and ligand binding.

    Args:
        protein_features (int): Number of input features for protein nodes.
        ligand_features (int): Number of input features for ligand nodes.
        edge_features (int): Number of input features for edges.
        hidden_dim (int): Dimension of hidden layers.
        num_layers (int): Number of GNN layers.
    """

    def __init__(
        self, protein_features, ligand_features, edge_features, hidden_dim, num_layers
    ):
        super(ProteinLigandGNN, self).__init__()
        self.protein_embedding = nn.Linear(protein_features, hidden_dim)
        self.ligand_embedding = nn.Linear(ligand_features, hidden_dim)
        self.edge_embedding = nn.Linear(edge_features, hidden_dim)

        # Multiple GAT layers for message passing
        self.gat_layers = nn.ModuleList(
            [
                GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
                for _ in range(num_layers)
            ]
        )

        # Layer normalization for stability
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        # Final output layer for amino acid prediction
        self.output_layer = nn.Linear(hidden_dim * 2, NUM_AMINO_ACIDS)

    def forward(self, x, edge_index, edge_attr, batch, ligand_x, ligand_batch):
        """
        Forward pass of the ProteinLigandGNN.

        Args:
            x (Tensor): Protein node features.
            edge_index (Tensor): Graph connectivity in COO format.
            edge_attr (Tensor): Edge features.
            batch (Tensor): Batch vector for protein nodes.
            ligand_x (Tensor): Ligand node features.
            ligand_batch (Tensor): Batch vector for ligand nodes.

        Returns:
            Tensor: Logits for amino acid prediction at each protein node.
        """
        protein_x = self.protein_embedding(x)
        edge_attr = self.edge_embedding(edge_attr)

        # Handle the case where we input only a single sequence, as in generating
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        if ligand_x.size(0) > 0:
            ligand_x = self.ligand_embedding(ligand_x)
            # Combine protein and ligand features
            combined_x = torch.cat([protein_x, ligand_x], dim=0)
            combined_batch = torch.cat([batch, ligand_batch], dim=0)
        else:
            combined_x = protein_x
            combined_batch = batch

        # Apply GAT layers with residual connections and layer normalization
        for gat, ln in zip(self.gat_layers, self.layer_norms):
            x_res = combined_x
            combined_x = gat(combined_x, edge_index, edge_attr)
            combined_x = ln(combined_x + x_res)
            combined_x = F.relu(combined_x)

        # Global pooling of ligand features
        if ligand_x.size(0) > 0:
            ligand_pooled = global_add_pool(ligand_x, ligand_batch)
        else:
            ligand_pooled = torch.zeros(
                (1, self.protein_embedding.out_features), device=protein_x.device
            )

        # Concatenate protein features with pooled ligand features
        protein_x = combined_x[: protein_x.size(0)]
        output = torch.cat([protein_x, ligand_pooled[batch]], dim=1)

        return self.output_layer(output)


# Featurization --------------


def rbf_encode(distances, num_rbf=NUM_RBF, max_distance=MAX_DISTANCE):
    """
    Encode distances using Radial Basis Functions (RBF).

    Args:
        distances (Tensor): Input distances to encode.
        num_rbf (int): Number of RBF kernels to use.
        max_distance (float): Maximum distance for RBF encoding.

    Returns:
        Tensor: RBF-encoded distances.
    """
    rbf_centers = torch.linspace(0, max_distance, num_rbf)
    rbf_widths = (max_distance / num_rbf) * torch.ones_like(rbf_centers)
    rbf_activation = torch.exp(
        -((distances.unsqueeze(-1) - rbf_centers) ** 2) / (2 * rbf_widths**2)
    )
    return rbf_activation


def phi_psi_atoms(residues):
    """
    Get the atoms necessary to compute phi and psi dihedral angles for the middle residue.

    Args:
        residues (list): A list of three consecutive Biopython residue objects.

    Returns:
        tuple: Two tuples containing atoms for phi and psi angle calculation.
               Returns (None, None) if the necessary atoms are not present.
    """
    if len(residues) != 3:
        return None, None

    phi_atoms = None
    psi_atoms = None

    if (
        "C" in residues[0]
        and "N" in residues[1]
        and "CA" in residues[1]
        and "C" in residues[1]
    ):
        phi_atoms = (
            residues[0]["C"],
            residues[1]["N"],
            residues[1]["CA"],
            residues[1]["C"],
        )

    if (
        "N" in residues[1]
        and "CA" in residues[1]
        and "C" in residues[1]
        and "N" in residues[2]
    ):
        psi_atoms = (
            residues[1]["N"],
            residues[1]["CA"],
            residues[1]["C"],
            residues[2]["N"],
        )

    return phi_atoms, psi_atoms


def calculate_dihedral(p1, p2, p3, p4):
    """
    Calculate dihedral angle between four points.

    Args:
        p1, p2, p3, p4 (Tensor): 3D coordinates of four points.

    Returns:
        float: Dihedral angle in radians.
    """
    b1 = p2 - p1
    b2 = p3 - p2
    b3 = p4 - p3
    n1 = torch.cross(b1, b2)
    n2 = torch.cross(b2, b3)
    m1 = torch.cross(n1, b2)
    x = torch.dot(n1, n2)
    y = torch.dot(m1, n2)
    return torch.atan2(y, x)


def get_backbone_dihedrals(residues):
    """
    Calculate backbone dihedral angles (phi, psi, omega, chi1) for a list of residues.

    Args:
        residues (list): List of Biopython residue objects.

    Returns:
        Tensor: Tensor of shape (num_residues, 4) containing dihedral angles.
    """
    dihedrals = []
    for i in range(len(residues)):
        phi, psi = phi_psi_atoms(residues[i - 1 : i + 2])

        # Calculate phi angle
        if phi is not None:
            phi_angle = calculate_dihedral(*[torch.tensor(atom.coord) for atom in phi])
        else:
            phi_angle = 0.0

        # Calculate psi angle
        if psi is not None:
            psi_angle = calculate_dihedral(*[torch.tensor(atom.coord) for atom in psi])
        else:
            psi_angle = 0.0

        # Calculate omega angle
        if i > 0:
            prev_c = residues[i - 1]["C"].coord
            curr_n = residues[i]["N"].coord
            curr_ca = residues[i]["CA"].coord
            curr_c = residues[i]["C"].coord
            omega_angle = calculate_dihedral(
                torch.tensor(prev_c),
                torch.tensor(curr_n),
                torch.tensor(curr_ca),
                torch.tensor(curr_c),
            )
        else:
            omega_angle = 0.0

        # Calculate chi1 angle (if possible)
        if "CB" in residues[i]:
            n = residues[i]["N"].coord
            ca = residues[i]["CA"].coord
            cb = residues[i]["CB"].coord
            cg = next(
                (
                    residues[i][atom].coord
                    for atom in ["CG", "SG", "OG", "OG1"]
                    if atom in residues[i]
                ),
                None,
            )
            if cg is not None:
                chi1_angle = calculate_dihedral(
                    torch.tensor(n),
                    torch.tensor(ca),
                    torch.tensor(cb),
                    torch.tensor(cg),
                )
            else:
                chi1_angle = 0.0
        else:
            chi1_angle = 0.0

        dihedrals.append([phi_angle, psi_angle, omega_angle, chi1_angle])

    return torch.tensor(dihedrals)


def encode_atom(atom):
    """
    Encode atom features.

    Args:
        atom (Bio.PDB.Atom): Biopython atom object.

    Returns:
        Tensor: Tensor of encoded atom features.
    """
    # This is a simplified version. You may want to expand this with more detailed atom features.
    atom_types = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "Other"]
    type_encoding = [1 if atom.element in atom_type else 0 for atom_type in atom_types]
    return torch.tensor(type_encoding, dtype=torch.float)


def load_protein_ligand_graph(pdb_file):
    """
    Load protein and ligand information from a PDB file and construct a graph.

    Args:
        pdb_file (str): Path to the PDB file.

    Returns:
        Data: A PyTorch Geometric Data object containing the graph representation.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    residues = list(structure.get_residues())
    ca_atoms = [residue["CA"] for residue in residues if "CA" in residue]
    coords = torch.tensor([atom.coord for atom in ca_atoms], dtype=torch.float)

    # Create k-nearest neighbors graph
    transform = RadiusGraph(r=MAX_DISTANCE, max_num_neighbors=MAX_NUM_NEIGHBORS)
    data = Data(pos=coords)
    data = transform(data)

    # Compute edge features (RBF-encoded distances)
    distances = torch.norm(
        coords[data.edge_index[0]] - coords[data.edge_index[1]], dim=1
    )
    edge_attr = rbf_encode(distances)

    # Node features (backbone dihedrals)
    x = get_backbone_dihedrals(residues)

    # Target (one-hot encoded amino acid types)
    amino_acids = [residue.resname for residue in residues if "CA" in residue]
    # aa_to_idx = {aa: idx for idx, aa in enumerate(sorted(set(amino_acids)))}
    print(aa_to_idx)
    y = F.one_hot(
        torch.tensor([aa_to_idx[aa] for aa in amino_acids]), num_classes=NUM_AMINO_ACIDS
    )

    # Extract ligand and small molecule information
    ligand_atoms = [atom for atom in structure.get_atoms() if atom.parent.id[0] != " "]

    if ligand_atoms:
        ligand_coords = torch.tensor(
            [atom.coord for atom in ligand_atoms], dtype=torch.float
        )
        ligand_features = torch.stack([encode_atom(atom) for atom in ligand_atoms])
    else:
        # Create dummy ligand data if no ligands are present
        ligand_coords = torch.zeros((1, 3), dtype=torch.float)
        ligand_features = torch.zeros((1, NUM_ATOM_FEATURES), dtype=torch.float)

    # Create a batch index for ligand atoms
    ligand_batch = torch.zeros(ligand_features.size(0), dtype=torch.long)

    return Data(
        x=x,
        edge_index=data.edge_index,
        edge_attr=edge_attr,
        y=y,
        ligand_x=ligand_features,
        ligand_pos=ligand_coords,
        ligand_batch=ligand_batch,
    )


def train(model, train_loader, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model (ProteinLigandGNN): The model to train.
        train_loader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to use for training.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    total_loss = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
            data.ligand_x,
            data.ligand_batch,
        )
        loss = F.cross_entropy(out, data.y.argmax(dim=1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


import torch
import torch.nn.functional as F


def generate_sequence(model, data, device, temperature=1.0, top_k=None):
    """
    Generate a protein sequence using the trained model with temperature-based sampling
    and optional top-k filtering.

    Args:
        model (ProteinLigandGNN): The trained model.
        data (Data): Input data containing protein and ligand information.
        device (torch.device): Device to use for inference.
        temperature (float): Temperature for softmax. Higher values increase randomness.
        top_k (int, optional): If set, only sample from the top k most probable tokens.

    Returns:
        Tensor: Generated sequence as indices of amino acids.
    """
    model.eval()
    data = data.to(device)
    with torch.no_grad():
        logits = model(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
            data.ligand_x,
            data.ligand_batch,
        )

        # Apply temperature
        logits = logits / temperature

        if top_k is not None:
            # Get the top k logits
            top_k_logits, top_k_indices = torch.topk(logits, k=top_k, dim=1)

            # Create a mask for the top k logits
            mask = torch.zeros_like(logits).scatter_(1, top_k_indices, 1.0)
            logits = torch.where(
                mask > 0, logits, torch.full_like(logits, float("-inf"))
            )

        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)

        # Sample from the probability distribution
        sequence = torch.multinomial(probs, num_samples=1).squeeze().tolist()
        sequence_letters = list(idx_to_aa[token] for token in sequence)
        sequence_letters = list(
            IUPACData.protein_letters_3to1[resname.capitalize()]
            for resname in sequence_letters
        )

    return "".join(sequence_letters)


# Example usage:
# generated_sequence = generate_sequence(model, test_data, device, temperature=0.8, top_k=5)

# Main training loop
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = ProteinLigandGNN(
    protein_features=NUM_DIHEDRAL_FEATURES,
    ligand_features=NUM_ATOM_FEATURES,
    edge_features=NUM_RBF,
    hidden_dim=64,
    num_layers=3,
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Assume we have a list of PDB files for training
pdb_files = ["data/dompdb/12asA00", "data/dompdb/132lA00", "data/dompdb/4a02A00"]
dataset = [load_protein_ligand_graph(pdb_file) for pdb_file in pdb_files]
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

num_epochs = 200
for epoch in range(num_epochs):
    loss = train(model, train_loader, optimizer, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.4f}")

# Generate sequence for a test protein
test_pdb = "data/dompdb/12asA00"
test_data = load_protein_ligand_graph(test_pdb).to(device)
generated_sequence = generate_sequence(
    model, test_data, device, temperature=1.0, top_k=None
)
print("Generated sequence:", generated_sequence)
