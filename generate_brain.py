import numpy as np
import scipy.sparse as sp

print("Generating 10,500-neuron sparse brain...")
num_neurons = 10500
density = 0.01
total_connections = int(num_neurons * num_neurons * density)

# Generate random connections (row, col, data)
rows = np.random.randint(0, num_neurons, total_connections)
cols = np.random.randint(0, num_neurons, total_connections)
data = np.random.randn(total_connections) * 0.1

# Initial state
V = np.zeros(num_neurons)

# Save to compressed .npz
np.savez_compressed("vibrane_brain.npz", 
                    data=data, 
                    rows=rows, 
                    cols=cols, 
                    V=V, 
                    shape=(num_neurons, num_neurons))

print("✅ Brain saved to vibrane_brain.npz")
