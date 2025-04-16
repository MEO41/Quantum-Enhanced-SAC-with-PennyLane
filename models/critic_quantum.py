import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np
from pennylane import numpy as pnp

class QuantumCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256], 
                 n_qubits=4, n_layers=2, device='cpu'):
        """
        Quantum Critic using PennyLane QNN for Q-function approximation
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            hidden_dims: List of hidden layer dimensions for classical pre-processing
            n_qubits: Number of qubits in the quantum circuit
            n_layers: Number of variational layers
            device: Device to run the model on
        """
        super(QuantumCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.device = device
        
        # Classical preprocessing network for Q1 (encode high-dimensional inputs)
        self.q1_pre_layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], n_qubits)
        )
        
        # Classical preprocessing network for Q2 (double Q-learning)
        self.q2_pre_layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], n_qubits)
        )
        
        # Classical post-processing layer
        self.q1_post_layers = nn.Linear(1, 1)
        self.q2_post_layers = nn.Linear(1, 1)
        
        # Initialize quantum device and weights
        self.init_quantum_devices()
        
    def init_quantum_devices(self):
        """Initialize the quantum devices and trainable parameters"""
        # Define the quantum device
        dev1 = qml.device("default.qubit", wires=self.n_qubits)
        dev2 = qml.device("default.qubit", wires=self.n_qubits)
        
        # Define the number of parameters
        weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}
        
        # Create the quantum nodes
        self.q1_quantum = qml.QNode(self.circuit, dev1, interface="torch")
        self.q2_quantum = qml.QNode(self.circuit, dev2, interface="torch")
        
        # Initialize trainable parameters
        self.q1_weights = nn.Parameter(torch.Tensor(self.n_layers, self.n_qubits, 3).uniform_(-0.01, 0.01))
        self.q2_weights = nn.Parameter(torch.Tensor(self.n_layers, self.n_qubits, 3).uniform_(-0.01, 0.01))
    
    def circuit(self, inputs, weights):
        """
        Quantum circuit definition for the QNN
        
        Args:
            inputs: Data inputs to encode into the quantum state
            weights: Trainable weights for the variational circuit
        
        Returns:
            Expectation value of measurement
        """
        # Encode inputs into quantum state
        for i in range(self.n_qubits):
            qml.RY(inputs[i], wires=i)
        
        # Variational quantum circuit layers
        for l in range(self.n_layers):
            # Rotation gates
            for i in range(self.n_qubits):
                qml.RX(weights[l, i, 0], wires=i)
                qml.RY(weights[l, i, 1], wires=i)
                qml.RZ(weights[l, i, 2], wires=i)
            
            # Entangling gates (CNOT ladder)
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            
            # Connect last qubit to first for full entanglement
            if self.n_qubits > 2:
                qml.CNOT(wires=[self.n_qubits - 1, 0])
        
        # Return expectation value
        return qml.expval(qml.PauliZ(0))
    
    def forward(self, state, action):
        """Forward pass to get Q-values using quantum circuits"""
        x = torch.cat([state, action], dim=1)
        
        # Q1 forward pass
        q1_inputs = self.q1_pre_layers(x)
        q1_quantum_out = torch.zeros(q1_inputs.shape[0], 1, device=self.device)
        
        # Process each data point through quantum circuit (batching)
        for i in range(q1_inputs.shape[0]):
            q1_quantum_out[i] = self.q1_quantum(q1_inputs[i], self.q1_weights)
        q1 = self.q1_post_layers(q1_quantum_out)
        
        # Q2 forward pass
        q2_inputs = self.q2_pre_layers(x)
        q2_quantum_out = torch.zeros(q2_inputs.shape[0], 1, device=self.device)
        
        # Process each data point through quantum circuit (batching)
        for i in range(q2_inputs.shape[0]):
            q2_quantum_out[i] = self.q2_quantum(q2_inputs[i], self.q2_weights)
        q2 = self.q2_post_layers(q2_quantum_out)
        
        return q1, q2
    
    def q1_forward(self, state, action):
        """Forward pass for just Q1 network"""
        x = torch.cat([state, action], dim=1)
        
        # Q1 forward pass
        q1_inputs = self.q1_pre_layers(x)
        q1_quantum_out = torch.zeros(q1_inputs.shape[0], 1, device=self.device)
        
        # Process each data point through quantum circuit (batching)
        for i in range(q1_inputs.shape[0]):
            q1_quantum_out[i] = self.q1_quantum(q1_inputs[i], self.q1_weights)
        q1 = self.q1_post_layers(q1_quantum_out)
        
        return q1