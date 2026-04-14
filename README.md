# Multi-Source-Distillation-for-Lightweight-CNNs
Training-time optimization framework for VanillaNet-6 improving representation and knowledge transfer via structural augmentation and multi-source distillation. Experiments show consistent accuracy gains while maintaining low computational cost, enabling efficient ultra-lightweight CNN optimization.


## New Benchmarks for ResNet-18

<div style="width: 90%; margin: 0 auto; text-align: justify;">

<p align="center">
  <img src="assets/New_Benchmarks for_ResNet-18.png" style="width:100%;">
</p>

<b>Figure.</b> New benchmarks for ResNet-18 trained from scratch on CIFAR-10 and CIFAR-100 datasets.

We train a standard ResNet-18 from scratch on CIFAR-10 and CIFAR-100 as a baseline and teacher model for subsequent distillation experiments.

ResNet-18 achieves **81.48% Top-1 / 95.45% Top-5** on CIFAR-100 and **95.94% Top-1 / 99.76% Top-5** on CIFAR-10.

These results demonstrate that a pure CNN architecture can still provide strong classification performance and serve as a reliable and stable teacher for knowledge distillation.

</div>





## Proposed Model Performance on CIFAR-100

<div style="width: 90%; margin: 0 auto; text-align: justify;">

<p align="center">
  <img src="assets/Proposed_Model_Performance_on_CIFAR-100.png" style="width:100%;">
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
  <img src="assets/Proposed_Model_Performance_on_CIFAR-10.png" style="width:100%;">
</p>

<b>Figure.</b> Performance comparison of VanillaNet-6 on CIFAR-10.

The baseline VanillaNet-6 achieves **90.66% Top-1 accuracy** without distillation.

With the proposed distillation strategy, performance is significantly improved to **95.15% Top-1** and **99.71% Top-5**, approaching the teacher model (**95.94% Top-1**).

These results demonstrate that the proposed distillation framework effectively improves both learning capacity and generalization, even under low-complexity classification settings.

</div>












