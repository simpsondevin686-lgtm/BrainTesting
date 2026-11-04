import http.server
import socketserver
import json
import numpy as np
import scipy.sparse as sp
import os

class MegaBrain:
    def __init__(self, filename="vibrane_brain.npz"):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Brain file {filename} not found. Run generate_brain.py first!")
        
        print("Loading brain from .npz vial...")
        brain_data = np.load(filename)
        
        self.shape = tuple(brain_data['shape'])
        self.V = brain_data['V']
        self.W = sp.csr_matrix((brain_data['data'], (brain_data['rows'], brain_data['cols'])), shape=self.shape)
        
        self.num_neurons = self.shape[0]
        self.num_inputs = 130
        self.num_outputs = 10
        self.idx_output = np.arange(self.num_neurons - self.num_outputs, self.num_neurons)
        self.learning_rate = 0.01
        print(f"✅ Brain loaded: {self.num_neurons} neurons, {len(brain_data['data'])} synapses.")

    def step(self, inputs, dopamine):
        self.V[:self.num_inputs] = inputs
        self.V = self.V * 0.95 + self.W.dot(self.V)
        self.V = np.maximum(0, self.V)
        
        # Dopamine-gated Hebbian learning
        if dopamine > 0.01:
            active = np.where(self.V > 1.0)[0]
            if len(active) > 1:
                for i in range(min(len(active), 50)):
                    for j in range(min(len(active), 50)):
                        self.W[active[i], active[j]] += self.learning_rate * dopamine
                        
        self.W.data *= 0.9999
        return self.V[self.idx_output]
    
    def save(self, filename="vibrane_brain.npz"):
        np.savez_compressed(filename, 
                            data=self.W.data, 
                            rows=self.W.tocoo().row, 
                            cols=self.W.tocoo().col, 
                            V=self.V, 
                            shape=self.shape)

brain = MegaBrain()
world = {
    "creature": {"x": 0.0, "y": 0.0, "z": 0.0, "angle": 0.0, "climbing": False},
    "trees": [],
    "drive": 1.0
}

# Generate Trees
for _ in range(30):
    world["trees"].append({
        "x": np.random.uniform(-75, 75),
        "z": np.random.uniform(-75, 75),
        "height": np.random.uniform(5, 15),
        "radius": 1.0
    })

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('index.html', 'rb') as f:
                self.wfile.write(f.read())
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/api/step':
            c = world["creature"]
            nearest_tree = None
            min_dist = 999
            
            for t in world["trees"]:
                dist = np.sqrt((t["x"] - c["x"])**2 + (t["z"] - c["z"])**2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_tree = t
            
            # Sensory Inputs
            vision_inputs = np.zeros(100)
            if nearest_tree:
                dx = nearest_tree["x"] - c["x"]
                dz = nearest_tree["z"] - c["z"]
                angle = np.arctan2(dz, dx)
                vision_inputs[0] = 1.0 / (min_dist + 1)
                vision_inputs[1] = np.sin(angle)
                vision_inputs[2] = np.cos(angle)
            
            pain_inputs = np.zeros(10)
            if c["y"] < 0: pain_inputs[0] = 1.0
            
            # The "Urge" (Drive to climb)
            drive = max(0, 10 - c["y"]) 
            world["drive"] = drive
            drive_inputs = np.zeros(10)
            drive_inputs[0] = drive
            
            prop_inputs = np.zeros(10)
            prop_inputs[0] = c["y"]
            
            total_inputs = np.concatenate([vision_inputs, pain_inputs, drive_inputs, prop_inputs])
            
            # Dopamine Reward (High up = good)
            dopamine = c["y"] * 0.1 if c["y"] > 2.0 else 0.0
            
            motor_out = brain.step(total_inputs, dopamine)
            
            move_x = np.clip(motor_out[0] * 0.5, -1.0, 1.0)
            move_z = np.clip(motor_out[1] * 0.5, -1.0, 1.0)
            climb_cmd = np.clip(motor_out[2], 0, 1)
            look_cmd = np.clip(motor_out[3] * 0.05, -0.1, 0.1)
            
            c["angle"] += look_cmd
            c["x"] += move_x * np.cos(c["angle"]) - move_z * np.sin(c["angle"])
            c["z"] += move_x * np.sin(c["angle"]) + move_z * np.cos(c["angle"])
            
            if nearest_tree and min_dist < 2.5:
                c["climbing"] = True
                c["y"] += climb_cmd * 0.5
                if c["y"] > nearest_tree["height"]:
                    c["y"] = nearest_tree["height"]
            else:
                c["climbing"] = False
                
            if not c["climbing"]:
                c["y"] -= 0.2
                if c["y"] < 0: c["y"] = 0
            
            # Save brain state periodically
            if np.random.rand() < 0.01:
                brain.save()
            
            res = json.dumps({"creature": c, "brain_drive": world["drive"]})
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(res.encode())
        else:
            self.send_error(404)

PORT = 8080
print(f"Server running at http://localhost:{PORT}")
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()
