# Protein graph neural networks 

Exploring the use of graph neural networks for biomolecules including proteins, nucleic acids, metal ions, and small molecule ligands.

Here we compare some of the best methods for graph based protein design currently available and see if we can develop some excellent evals for assessing how and why the models perform differently, with an eye towards improving them. 

The main sources of inspiration are: 

- [Generative Models for Graph-Based Protein Design](https://papers.nips.cc/paper/9711-generative-models-for-graph-based-protein-design) by John Ingraham, Vikas Garg, Regina Barzilay and Tommi Jaakkola, NeurIPS 2019.
- [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) by Justas Dauparas, 2021 
- [ESM-IF](https://github.com/facebookresearch/esm) by Alex Rives and coworkers, 2021  

From the orginal implementation and idea by John Ingraham: our goal is to create a model that "'designs' protein sequences for target 3D structures via a graph-conditioned, autoregressive language model". 


## Goals for this repository 

- [ ] Present a simple and understandable implementation of state of the art algorithms for graph-based protein design
    - [ ] Create fast and flexible structure data loaders from PDB for PyTorch Geometric 
    - [x] Implement featurization from Ingraham
    - [x] Implement featurization scheme from ProteinMPNN
- [ ] Train baseline models 
    - [ ] Train Ingraham model with coarse feature set 
    - [ ] Train Ingraham model with full feature set 
    - [ ] Train ProteinMPNN model with settings from paper 
- [ ] Perform analysis of the model attention mechanism 
- [ ] Devise evals that probe the ability of models under different conditions 


## Code overview 

`prepare_cath_dataset.ipynb`. Create the CATH dataset from raw files. Creates the files `chains.jsonl` and `splits.jsonl`. 

`compare_features.ipynb`. Compare features from Ingraham and ProteinMPNN.  

`train.py`. Implements a GAT model and the training loop.  