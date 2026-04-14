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





