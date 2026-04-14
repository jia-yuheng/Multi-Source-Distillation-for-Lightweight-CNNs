# Multi-Source-Distillation-for-Lightweight-CNNs
Training-time optimization framework for VanillaNet-6 improving representation and knowledge transfer via structural augmentation and multi-source distillation. Experiments show consistent accuracy gains while maintaining low computational cost, enabling efficient ultra-lightweight CNN optimization.


# AMINet-VI-ReID-CV-ML

##  Overview

Minimalist convolutional neural networks (CNNs), such as VanillaNet, offer exceptional deployment efficiency due to their simple and hardware-friendly design. However, their limited depth and lack of structural diversity severely restrict spatial modeling, semantic representation, and cross-layer information propagation, leading to suboptimal performance in complex visual tasks.

To address this fundamental limitation, we propose a **structure–distillation co-optimization framework** built upon VanillaNet-6, aiming to enhance representation capacity **without introducing any additional inference overhead**.

Our approach systematically improves lightweight networks from both **architectural design** and **training supervision** perspectives:

- **Structural Enhancement**:  
  We integrate **Depthwise Separable Convolution (DWConv)**, **Gated Activation**, and a lightweight **Hidden-State Mixer-based State Space Duality (HSM-SSD)** module to strengthen spatial modeling, nonlinearity, and global context representation.

- **Multi-Source Knowledge Distillation**:  
  A unified distillation framework combines **soft-label guidance**, **attention transfer**, and **activation alignment**, enabling effective knowledge transfer at semantic, spatial, and representational levels.

- **Training–Inference Decoupling**:  
  All auxiliary modules are **training-time only** and are removed or reparameterized during inference via structured fusion and pruning, ensuring **zero additional deployment cost**.



##  Key Results

Extensive experiments on CIFAR benchmarks demonstrate that our framework significantly improves performance under strict resource constraints:

- On **CIFAR-100**:
  - VanillaNet-6a achieves **80.48% Top-1 accuracy**, comparable to **ResNet-18 (81.48%)**
  - While reducing:
    - **Parameters by 85.3%**
    - **FLOPs by 51.4%**
    - **Latency by 30.1%**

- On **extreme lightweight setting (VanillaNet-6b)**:
  - Improves Top-1 accuracy by **+11.08%** (78.73% vs. 67.65%)
  - With **nearly identical parameter cost** and **−3.0% latency**

- On **CIFAR-10**:
  - Accuracy improves from **90.66% → 95.15%**, approaching teacher performance (**95.94%**)



## 💡 Core Contributions

- **Structure–Distillation Co-Optimization Framework**  
  A unified paradigm that jointly enhances representation capacity and knowledge transfer for ultra-lightweight CNNs.

- **Deployment-Friendly Structural Design**  
  Incorporation of **DWConv**, **Gated Activation**, and **HSM-SSD** to improve spatial modeling and global context understanding while maintaining minimal complexity.

- **Multi-Level Distillation Strategy**  
  Integration of **soft labels, attention transfer, and activation alignment** to provide complementary supervision signals.

- **Zero-Overhead Deployment Mechanism**  
  A **train–inference decoupling strategy** that ensures all enhancements are removed or fused during inference.

- **Strong Efficiency–Accuracy Trade-off**  
  Achieves near-ResNet performance with **significantly reduced parameters, FLOPs, and latency**, demonstrating practical value for real-world deployment.



## 🔍 Motivation

While existing lightweight models (e.g., MobileNet, ShuffleNet) rely on complex components such as inverted residuals and attention mechanisms, they often introduce hardware inefficiencies and deployment constraints.

In contrast, **VanillaNet** adopts a strictly sequential and homogeneous design, enabling efficient operator fusion and stable deployment. However, this simplicity comes at the cost of:

- Limited receptive field and weak spatial modeling  
- Lack of global context awareness  
- Inefficient knowledge distillation in shallow networks  

This work demonstrates that:

> **Even extremely minimalist architectures can achieve strong performance through principled co-design of structure and supervision.**










## New Benchmarks for ResNet-18

<div style="width: 90%; margin: 0 auto; text-align: justify;">

<p align="center">
  <img src="assets/New_Benchmarks for_ResNet-18.png" style="width:60%;">
</p>

<b>Figure.</b> New benchmarks for ResNet-18 trained from scratch on CIFAR-10 and CIFAR-100 datasets.

We train a standard ResNet-18 from scratch on CIFAR-10 and CIFAR-100 as a baseline and teacher model for subsequent distillation experiments.

ResNet-18 achieves **81.48% Top-1 / 95.45% Top-5** on CIFAR-100 and **95.94% Top-1 / 99.76% Top-5** on CIFAR-10.

These results demonstrate that a pure CNN architecture can still provide strong classification performance and serve as a reliable and stable teacher for knowledge distillation.

</div>





## Proposed Model Performance on CIFAR-100

<div style="width: 90%; margin: 0 auto; text-align: justify;">

<p align="center">
  <img src="assets/Proposed Model Performance on CIFAR-100.png" style="width:65%;">
</p>

<b>Figure.</b> Performance comparison of the proposed VanillaNet-6 variants on CIFAR-100.

We evaluate two variants of VanillaNet-6 under a structure–distillation co-optimization framework.

VanillaNet-6a achieves **80.48% Top-1 accuracy**, close to ResNet-18 (**81.48%**), while reducing **85.3% parameters**, **51.4% FLOPs**, and **30.1% inference latency**, demonstrating strong efficiency under deployment constraints.

VanillaNet-6b maintains similar computational cost to the baseline but improves Top-1 accuracy by **+11.08% (78.73% → 67.65%)**, with further latency reduction, showing strong robustness in extremely resource-limited settings.

These results validate the effectiveness of the proposed co-optimization framework for efficient and deployable visual recognition.

</div>







## Proposed Model Performance on CIFAR-10

<div style="width: 90%; margin: 0 auto; text-align: justify;">

<p align="center">
  <img src="assets/Proposed Model Performance on CIFAR-10.png" style="width:50%;">
</p>

<b>Figure.</b> Performance comparison of VanillaNet-6 on CIFAR-10.

The baseline VanillaNet-6 achieves **90.66% Top-1 accuracy** without distillation.

With the proposed distillation strategy, performance is significantly improved to **95.15% Top-1** and **99.71% Top-5**, approaching the teacher model (**95.94% Top-1**).

These results demonstrate that the proposed distillation framework effectively improves both learning capacity and generalization, even under low-complexity classification settings.

</div>







## Ablation Study on VanillaNet-6

<div style="width: 90%; margin: 0 auto; text-align: justify;">

<p align="center">
  <img src="assets/Ablation Study.png" style="width:60%;">
</p>

<b>Figure.</b> Ablation study of VanillaNet-6 on CIFAR-100 evaluating progressive integration of KD, AT, DWConv, Gated Activation, and HSM-SSD modules.

We conduct a stepwise ablation study to analyze the contribution of each component in VanillaNet-6 under a structure–distillation co-optimization framework.

Starting from the baseline (**67.65% Top-1**), Knowledge Distillation (KD) improves performance to **74.60%**, followed by Attention Transfer (AT) increasing it to **75.78%**, demonstrating the benefit of semantic and spatial supervision.

Replacing standard convolutions with Depthwise Separable Convolution (DWConv) further improves accuracy to **77.98%**, while Gated Activation enhances nonlinear representation to **78.85%**.

Finally, integrating the HSM-SSD module yields the best performance (**80.28% Top-1 / 95.22% Top-5**), confirming its effectiveness in capturing long-range dependencies with minimal computational overhead.

Overall, each component provides consistent gains, and their combination demonstrates strong additive effects and structural compatibility for lightweight model optimization.

</div>






<div style="width: 90%; margin: 0 auto; text-align: justify;">

<p align="center">
  <img src="assets/DW_Block study.png" style="width:60%;">
</p>

<b>Figure.</b> System-level optimization of VanillaNet-6 with Depthwise Separable Convolution (DWConv). The study evaluates progressive replacement across backbone blocks and reports changes in parameters, FLOPs, and Top-1 accuracy.

<b>Conclusion.</b> The results show that introducing DWConv significantly improves the efficiency–accuracy trade-off in VanillaNet-6. As replacement depth increases, FLOPs are reduced from 0.5272G to 0.2187G (↓58.52%), while Top-1 accuracy increases from 75.78% to 77.98%. The best performance is achieved when mid-level blocks are replaced, indicating that shallow and intermediate layers contain higher redundancy.

Importantly, DWConv reduces computational cost more effectively than parameter count due to high-resolution feature maps in early stages. The results confirm that structured replacement in shallow CNN stages is the most effective strategy for lightweight model optimization.

</div>





<div style="width: 90%; margin: 0 auto; text-align: justify;">

<p align="center">
  <img src="assets/HSM-SSD Module Overhead.png" style="width:60%;">
</p>

<b>Figure.</b> Efficiency–accuracy trade-off of the Hidden-State Mixer-based State Space Duality (HSM-SSD) module under different hidden state dimensions (state_dim).

<b>Conclusion.</b> HSM-SSD introduces a lightweight context modeling mechanism based on learnable hidden-state interaction, enabling efficient global dependency modeling with controlled computational overhead. The module provides a configurable trade-off via the hidden state dimension (<i>state_dim</i>), where the ratio ρ = state_dim / C_out governs the relative modeling capacity.

Experimental results show that HSM-SSD consistently improves performance with minimal overhead. Even under low configuration settings (ρ = 1/64–1/32), it achieves +1.27% to +1.43% Top-1 accuracy gain while keeping FLOPs increase below 7%. As ρ increases, performance improves further but saturates beyond ρ = 1/16 on CIFAR-100, while computational cost grows linearly, indicating diminishing returns.

These results demonstrate that HSM-SSD is a plug-and-play, task-adaptive context modeling module that enables flexible balancing between accuracy and efficiency in lightweight neural networks.

</div>







<div style="width: 90%; margin: 0 auto; text-align: justify;">

<p align="center">
  <img src="assets/Grad-CAM++ visualization.png" style="width:95%;">
</p>

<b>Figure.</b> Grad-CAM++ visualization of VanillaNet-6 and ResNet-18 on CIFAR-100, showing model inputs (with GT and Pred) and corresponding activation heatmaps. Red regions indicate the most discriminative areas contributing to predictions.

<b>Conclusion.</b> The Grad-CAM++ results reveal clear differences in spatial attention behavior between models. ResNet-18 produces highly localized activations concentrated on object cores, demonstrating strong spatial selectivity and precise semantic focus. In contrast, VanillaNet-6 exhibits broader and more diffused activation patterns, often extending beyond target regions into background areas.

This behavior reflects the inherent limitations of shallow architectures, where limited depth and weaker nonlinearity constrain the formation of compact and discriminative features. While such distributed attention improves robustness under strict parameter constraints, it reduces semantic focus and increases susceptibility to background interference, particularly in fine-grained or visually similar categories.

</div>





<div style="width: 90%; margin: 0 auto; text-align: justify;">

<p align="center">
  <img src="assets/training dynamics.png" style="width:70%;">
</p>

<b>Figure.</b> Training dynamics comparison between the proposed model and the baseline on CIFAR-100, including Top-1 / Top-5 accuracy and training loss curves.

<b>Conclusion.</b> The proposed model demonstrates significantly improved training efficiency and performance. It achieves faster convergence and consistently higher accuracy, reaching <b>80.33%</b> Top-1 and <b>95.49%</b> Top-5 accuracy at convergence, outperforming the baseline (<b>67.65%</b> / <b>88.62%</b>) by <b>+12.68%</b> and <b>+6.87%</b>, respectively.

In addition, the model maintains a lower and more stable training loss, converging to <b>0.851</b> compared to <b>1.544</b> for the baseline (↓<b>0.693</b>). The smoother optimization trajectory indicates improved numerical stability and optimization dynamics.

These results confirm that the integration of multi-source distillation, nonlinear activation, and efficient context modeling jointly enhances convergence speed, representation capacity, and overall generalization performance.

</div>








<div style="width: 90%; margin: 0 auto; text-align: justify;">

<b>Conclusion.</b> This work presents a unified structure–distillation co-optimization framework built upon the VanillaNet-6 baseline, targeting performance enhancement under extremely lightweight constraints. The proposed method improves representation capacity from three complementary aspects: structural compression (Depthwise Separable Convolution), nonlinear modeling (Gated Activation), and global context modeling (HSM-SSD).

In parallel, a multi-source distillation scheme is introduced, integrating soft label distillation, attention transfer, and activation alignment to provide complementary supervision at semantic, spatial, and representational levels. All modules follow a train–inference decoupling principle, ensuring zero additional cost during deployment through re-parameterization and path pruning.

Extensive experiments demonstrate that the proposed framework consistently improves spatial modeling, feature expressiveness, and optimization dynamics, achieving significant accuracy gains over the VanillaNet-6 baseline without increasing inference overhead. These results validate that effective co-design of architecture and distillation is a practical and scalable paradigm for high-performance lightweight CNNs.

</div>







