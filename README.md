# 🚀 Multi-Source Distillation for Ultra-Lightweight CNNs  
### Structure–Distillation Co-Optimization Framework on VanillaNet-6

> Achieving **near-ResNet-18 performance** while dramatically reducing computational cost  
> → Parameters ↓ 85.3% → FLOPs ↓ 51.4% → Latency ↓ 30.1%

---

## 🔥 Key Results 

Extensive experiments on CIFAR benchmarks demonstrate that the proposed framework consistently improves lightweight CNN performance under strict efficiency constraints:






### 📊 CIFAR-100 Performance

We evaluate two variants of VanillaNet-6 under the proposed framework:


<p align="center">
  <img src="assets/Proposed Model Performance on CIFAR-100.png" width="60%">
</p>


#### 🔹 VanillaNet-6a (High-Performance Setting)

- Achieves **80.48% Top-1 accuracy**     → approaching **ResNet-18 (81.48%)**

- While significantly reducing computational cost:
  - Parameters ↓ **85.3%**             - FLOPs ↓ **51.4%**              - Latency ↓ **30.1%**

✔ Demonstrates strong efficiency–accuracy trade-off under practical deployment constraints


#### 🔹 VanillaNet-6b (Extreme Lightweight Setting)

- Improves Top-1 accuracy:              → **67.65% → 78.73% (↑ +11.08%)**

- Under nearly identical computational budget:
  - Parameters → unchanged               - Latency ↓ **3.0%**

✔ Demonstrates strong robustness in ultra-low resource regimes




---
## 📎 Other Results

Please see additional experimental results below for more detailed analysis:

- ResNet-18 Benchmark (Teacher Model)
- CIFAR-10 Performance
- Grad-CAM++ Visualization
- Training Dynamics Curves





---

## 🧪 Ablation Study

<p align="center">
  <img src="assets/Ablation Study.png" width="60%">
</p>


✔ Each component provides:     → consistent improvement       → strong compatibility  












---

## ⚡ Efficiency Study

### 🔹 Depthwise Separable Convolution (DWConv)

<p align="center">
  <img src="assets/DW_Block study.png" width="60%">
</p>

- FLOPs:
  → **0.527G → 0.218G (↓ 58.52%)**

- Accuracy:
  → **75.78% → 77.98% (↑)**

✔ Most effective in:
→ shallow and intermediate layers  

✔ Key observation:
→ early-stage feature maps dominate computational cost  

---

### 🔹 HSM-SSD Module

<p align="center">
  <img src="assets/HSM-SSD Module Overhead.png" width="60%">
</p>

- Accuracy gain:
  → **↑ +1.27% ~ +1.43%**

- FLOPs increase:
  → **< 7%**

✔ Controlled by:
→ state_dim ratio (ρ = state_dim / C_out)

✔ Observation:
→ performance saturates beyond ρ = 1/16      → cost grows linearly  







---

## 💡 Core Contributions 

This work proposes a **unified structure–distillation co-optimization framework** for ultra-lightweight CNNs, with the following key contributions:



### 1️⃣ Structure–Distillation Co-Design Paradigm

We introduce a novel perspective that jointly optimizes:

- **Architectural representation capacity**
- **Training-time supervision signals**

→ Instead of treating network design and knowledge distillation independently,  
we show that their **co-design leads to significantly improved performance** in shallow networks



### 2️⃣ Deployment-Friendly Structural Enhancement

We systematically enhance VanillaNet using lightweight yet effective modules:

- **Depthwise Separable Convolution (DWConv)**  
  → improves spatial modeling  
  → reduces redundant computation in high-resolution stages  

- **Gated Activation Mechanism**  
  → replaces linear/non-smooth activations  
  → enhances nonlinear representation capability  

- **HSM-SSD (Hidden-State Mixer-based State Space Duality)**  
  → introduces efficient global context modeling  
  → captures long-range dependencies with controllable overhead  

✔ All modules are carefully designed to maintain **hardware-friendly structure**



### 3️⃣ Multi-Source Knowledge Distillation Framework

We propose a unified distillation scheme that integrates complementary supervision signals:

- **Soft Label Distillation (KD)**  
  → transfers semantic knowledge  

- **Attention Transfer (AT)**  
  → aligns spatial attention distributions  

- **Activation Alignment**  
  → enforces consistency in intermediate representations  

✔ Enables knowledge transfer at:
→ semantic level  
→ spatial level  
→ feature representation level  



### 4️⃣ Train–Inference Decoupling Mechanism

A key practical contribution is the design of a **deployment-oriented training strategy**:

- All auxiliary modules are **introduced only during training**
- During inference:
  → removed  
  → fused  
  → or re-parameterized  

✔ Ensures:
→ **zero additional inference cost**  
→ full compatibility with efficient deployment pipelines  



### 5️⃣ Superior Efficiency–Accuracy Trade-off

The proposed framework achieves:

- Accuracy ↑ significantly  
- Parameters ↓ drastically  
- FLOPs ↓ substantially  
- Latency ↓ consistently  

→ Demonstrating a **practical and scalable paradigm** for real-world lightweight vision systems







---

## 🧱 Framework Overview

We build a **training-time optimization framework** upon VanillaNet-6, targeting:

> ⚠️ Enhancing representation capability **without increasing inference complexity**

The framework consists of two tightly coupled components:



### 1️⃣  Structural Enhancement

Improves intrinsic modeling capability:

- DWConv → efficient spatial feature extraction  
- Gated Activation → enhanced nonlinear transformation  
- HSM-SSD → global context aggregation  



### 2️⃣ Multi-Level Supervision

Improves training dynamics and knowledge transfer:

- KD → semantic guidance  
- AT → spatial guidance  
- Activation Alignment → representation guidance  



### 3️⃣ Train–Inference Decoupling

- Training: full model with all modules  
- Inference: simplified VanillaNet  

✔ Achieves:
→ high training capacity  
→ minimal deployment cost  






---

## ⚙️ Method Overview (Motivation & Design)

Minimalist CNNs (e.g., VanillaNet) exhibit strong deployment efficiency due to:

- Simple sequential architecture  
- Hardware-friendly operators  
- Stable re-parameterization  

However, this simplicity introduces critical limitations:

- Limited receptive field  
- Weak global context modeling  
- Insufficient nonlinear expressiveness  
- Ineffective knowledge distillation in shallow structures  

---

### 🔍 Key Insight

> Lightweight CNN performance is not fundamentally limited by architecture simplicity,  
but by the **lack of appropriate structural priors and supervision signals**

### ✔ Our Solution

We address these issues through:

- Structural enhancement → improving feature extraction capacity  
- Distillation guidance → improving learning efficiency  
- Co-design strategy → aligning both aspects  







---

## 🏆 Other Results

### 📊 New ResNet-18 Benchmark (Teacher)

<p align="center">
  <img src="assets/New_Benchmarks for_ResNet-18.png" width="60%">
</p>

- CIFAR-100:
  → **81.48% Top-1 / 95.45% Top-5**

- CIFAR-10:
  → **95.94% Top-1 / 99.76% Top-5**

✔ Serves as **teacher model** and strong CNN baseline



---

### 📊 CIFAR-10 Performance


<p align="center">
  <img src="assets/Proposed Model Performance on CIFAR-10.png" width="50%">
</p>

✔ Significant gain under low-complexity setting
- Baseline VanillaNet-6:                    → **90.66% Top-1**
- With proposed framework:                  → **95.15% Top-1 (↑ +4.49%)**
- Approaching teacher:           → **ResNet-18: 95.94%**




---

## 🔍 Visualization

### 🔹 Grad-CAM++

<p align="center">
  <img src="assets/Grad-CAM++ visualization.png" width="95%">
</p>

- ResNet-18:         → highly localized attention  
- VanillaNet:        → diffused activation  

✔ Ours improves:          → spatial focus          → discriminative regions  



---

### 🔹 Training Dynamics

<p align="center">
  <img src="assets/training dynamics.png" width="70%">
</p>

- Top-1 accuracy:
  → **67.65% → 80.33% (↑ +12.68%)**

- Loss:
  → **1.544 → 0.851 (↓)**

✔ Observations:

- Faster convergence  
- More stable optimization  
- Lower final loss  



## ✅ Conclusion

Lightweight CNNs are widely used in resource-constrained scenarios due to their efficiency, but their performance is limited by restricted receptive fields, weak global context modeling, limited nonlinear representation, and suboptimal knowledge transfer in shallow architectures.

To address these limitations, this work proposes a structure–distillation co-optimization framework based on VanillaNet-6, enhancing model capacity from both architectural and training perspectives while maintaining strict efficiency constraints. Specifically, we introduce DWConv, Gated Activation, and HSM-SSD for efficient structural modeling, together with multi-source knowledge distillation (KD, AT, and Activation Alignment) for improved supervision.

A key result of this work is that lightweight model performance is not only determined by architectural efficiency, but also by the quality of training supervision and structural inductive bias. By jointly optimizing both aspects, the proposed framework significantly narrows the performance gap between lightweight models and larger networks while preserving zero additional inference overhead.
