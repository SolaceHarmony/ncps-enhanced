# Copyright (2017-2021)
# The Wormnet project
# Mathias Lechner (mlechner@ist.ac.at)
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from ncps.mlx import CfC, train_model
from ncps.mlx.wirings import FullyConnected, NCP, AutoNCP  # Import from correct location
from mlx.utils import tree_map, tree_flatten

# Generate sequence data
N = 1000  # sequence length
in_features = 2  # input dimension
out_features = 1  # output dimension
batch_size = 1  # using batch_size=1 for this example

# Generate input sequence: [batch_size, seq_length, features]
data_x = mx.stack([
    mx.sin(mx.linspace(0, 3 * mx.pi, N)), 
    mx.cos(mx.linspace(0, 3 * mx.pi, N))
], axis=1)  # Shape: [N, 2]
data_x = mx.expand_dims(data_x, axis=0)  # Shape: [1, N, 2]

# Generate target sequence: [batch_size, seq_length, output_dim]
data_y = mx.sin(mx.linspace(0, 6 * mx.pi, N))  # Shape: [N]
data_y = mx.reshape(data_y, (1, N, out_features))  # Shape: [1, N, 1]

# Generate time delta information: [batch_size, seq_length]
# Using variable time steps but stabilizing range
time_delta = mx.clip(mx.random.uniform(low=0.9, high=1.1, shape=(batch_size, N)), 0.9, 1.1)

# Initialize with explicit random seed for reproducibility
mx.random.seed(42)

# Create and build wirings first
auto_wiring = AutoNCP(units=16, output_size=out_features)
auto_wiring.build(in_features)

fc_wiring = FullyConnected(units=8, output_dim=out_features)
fc_wiring.build(in_features)

ncp_wiring = NCP(
    inter_neurons=6,  # Increased from 4
    command_neurons=3,  # Increased from 2
    motor_neurons=out_features,
    sensory_fanout=4,  # Increased from 3
    inter_fanout=2,  # Increased from 1
    recurrent_command_synapses=2,
    motor_fanin=3,  # Increased from 2
    seed=22222
)
ncp_wiring.build(in_features)

# Define loss function
def mse_loss(y_pred, y_true):
    return mx.mean((y_pred - y_true) ** 2)

# Gradient clipping function
def clip_grads(grads, max_norm, max_value):
    # First clip gradient values
    grads = tree_map(lambda g: mx.clip(g, -max_value, max_value), grads)
    
    # Then clip gradient norm
    grad_norm = mx.sqrt(sum(mx.sum(g * g) for _, g in tree_flatten(grads)))
    if grad_norm > max_norm:
        scale = max_norm / (grad_norm + 1e-6)
        grads = tree_map(lambda g: g * scale, grads)
    return grads

# Test different model configurations with names
model_configs = [
    (
        "AutoNCP-16", 
        CfC(
            wiring=auto_wiring,
            mode="default",
            return_sequences=True,
            activation="tanh"
        ),
        0.0001
    ),
    (
        "FullyConnected-8",
        CfC(
            wiring=fc_wiring,
            mode="pure",
            return_sequences=True,
            activation="tanh"
        ),
        0.0005
    ),
    (
        "NCP-8",
        CfC(
            wiring=ncp_wiring,
            mode="pure",
            return_sequences=True,
            activation="tanh"
        ),
        0.0001  # Increased from 0.00005 since training is stable
    ),
]

# Training configuration
training_config = {
    'num_epochs': 150,  # Increased epochs
    'max_grad_norm': 0.1,
    'max_grad_value': 1.0,
}

# Train each model configuration
for name, model, lr in model_configs:
    print(f"\nTraining {name}")
    
    # Initialize optimizer with beta parameters for more stability
    optimizer = optim.Adam(
        learning_rate=lr,
        betas=[0.9, 0.999],
        eps=1e-8
    )
    
    # Define loss function that will be used for gradient computation
    def loss_fn(params):
        model.update(params)
        pred = model(data_x, time_delta=time_delta)
        return mse_loss(pred, data_y)
    
    # Training loop
    best_loss = float('inf')
    patience = 15  # Increased patience
    patience_counter = 0
    
    for epoch in range(training_config['num_epochs']):
        # Compute loss and gradients
        loss, grads = mx.value_and_grad(loss_fn)(model.trainable_parameters())
        
        # Clip gradients
        grads = clip_grads(grads, 
                          training_config['max_grad_norm'],
                          training_config['max_grad_value'])
        
        # Update model parameters
        optimizer.update(model, grads)
        
        if mx.isnan(loss):
            print(f"Training {name} failed at epoch {epoch} with NaN loss.")
            break
        
        # Early stopping check
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.6f}")
            
        # Evaluate predictions periodically
        if (epoch + 1) % 50 == 0:
            pred = model(data_x, time_delta=time_delta)
            eval_loss = mse_loss(pred, data_y)
            print(f"Evaluation Loss: {eval_loss.item():.6f}")

print("Training complete.")