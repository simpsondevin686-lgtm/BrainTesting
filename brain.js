// 30,000 Neuron Advanced Parasitic Brain
class ParasiteBrain {
    constructor() {
        this.numNeurons = 30000;
        this.numInputs = 10;
        this.numOutputs = 6; // PitchUp, PitchDown, YawL, YawR, Fwd, Stop
        
        // Regions
        this.idx_S = 0;           // 0-499 Sensory
        this.idx_M = 500;         // 500-999 Motor
        this.idx_D = 1000;        // 1000-1499 Drive (Internal state)
        this.idx_E = 1500;        // 1500-22499 Excitatory Cortex
        this.idx_I = 22500;       // 22500-29999 Inhibitory Cortex

        this.V = new Float32Array(this.numNeurons);
        this.newV = new Float32Array(this.numNeurons);
        this.in = new Float32Array(this.numInputs);
        
        this.lr = 0.005;
        this.decayRate = 0.99;
        this.decayTimer = 0;
        
        this.buildMatrix();
    }

    buildMatrix() {
        const totalSynapses = 3000000; // 3 million
        this.wd = new Float32Array(totalSynapses);
        this.wi = new Int32Array(totalSynapses);
        this.wp = new Int32Array(this.numNeurons + 1);

        let edge = 0;
        for (let i = 0; i < this.numNeurons; i++) {
            this.wp[i] = edge;
            let isInhibitory = (i >= this.idx_I);
            let baseWeight = Math.random() * 0.5;
            let finalWeight = isInhibitory ? -baseWeight : baseWeight;

            // 100 outgoing connections per neuron
            for (let k = 0; k < 100; k++) {
                let target;
                // Structured Modular Routing
                if (i < this.idx_M) {
                    target = this.idx_E + Math.floor(Math.random() * (this.idx_I - this.idx_E));
                } else if (i < this.idx_E) {
                    target = this.idx_M + Math.floor(Math.random() * 500);
                } else {
                    // Cortex routes to Cortex and Motor
                    if (Math.random() < 0.9) target = this.idx_E + Math.floor(Math.random() * (this.idx_I - this.idx_E));
                    else target = this.idx_M + Math.floor(Math.random() * 500);
                }
                this.wd[edge] = finalWeight;
                this.wi[edge] = target;
                edge++;
            }
        }
        this.wp[this.numNeurons] = edge;
    }

    step(reward) {
        // Drive neurons: constant baseline of exploration
        this.V[this.idx_D] = 1.0;
        this.V[this.idx_D + 1] = 1.0;

        // Leak previous state
        for (let i = 0; i < this.numNeurons; i++) this.newV[i] = this.V[i] * 0.9;

        // Matrix multiply
        for (let i = 0; i < this.numNeurons; i++) {
            let s = this.wp[i], e = this.wp[i+1], sv = this.V[i];
            if (sv > 0.1) {
                for (let j = s; j < e; j++) {
                    this.newV[this.wi[j]] += sv * this.wd[j];
                }
            }
        }

        // Activation
        for (let i = 0; i < this.numNeurons; i++) {
            this.V[i] = Math.max(0, Math.min(10.0, this.newV[i]));
        }

        // Homeostasis
        this.decayTimer++;
        if (this.decayTimer >= 10) {
            for (let j = 0; j < this.wd.length; j++) this.wd[j] *= 0.9;
            this.decayTimer = 0;
        }

        // Hebbian Plasticity (Learning)
        if (reward > 0.01) {
            for (let i = 0; i < this.numNeurons; i++) {
                let sv = this.V[i];
                if (sv > 1.0) {
                    let isI = (i >= this.idx_I);
                    let s = this.wp[i], e = this.wp[i+1];
                    for (let j = s; j < e; j++) {
                        if (this.V[this.wi[j]] > 1.0) {
                            let dw = this.lr * reward;
                            dw = isI ? -Math.abs(dw) : Math.abs(dw);
                            this.wd[j] = Math.max(-5.0, Math.min(5.0, this.wd[j] + dw));
                        }
                    }
                }
            }
        }

        // Read Motor outputs
        let out = { pitch: 0, yaw: 0, thrust: 0 };
        for (let i = 0; i < 2; i++) out.pitch += this.V[this.idx_M + i] - this.V[this.idx_M + 2 + i];
        for (let i = 0; i < 2; i++) out.yaw += this.V[this.idx_M + 4 + i] - this.V[this.idx_M + 6 + i];
        out.thrust = this.V[this.idx_M + 8];

        return out;
    }
}
