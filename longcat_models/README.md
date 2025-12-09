---
license: apache-2.0
language:
- en
- zh
pipeline_tag: text-to-image
library_name: transformers
---

<div align="center">
  <img src="assets/longcat-image_logo.svg" width="45%" alt="LongCat-Image" />
</div>
<hr>

<div align="center" style="line-height: 1;">
    <a href='https://github.com/meituan-longcat/LongCat-Image/blob/main/assets/LongCat_Image_Technical_Report.pdf'><img src='https://img.shields.io/badge/Technical-Report-red'></a>
    <a href='https://github.com/meituan-longcat/LongCat-Image'><img src='https://img.shields.io/badge/GitHub-Code-black'></a>
    <a href='https://github.com/meituan-longcat/LongCat-Flash-Chat/blob/main/figures/wechat_official_accounts.png'><img src='https://img.shields.io/badge/WeChat-LongCat-brightgreen?logo=wechat&logoColor=white'></a>
    <a href='https://x.com/Meituan_LongCat'><img src='https://img.shields.io/badge/Twitter-LongCat-white?logo=x&logoColor=white'></a>
</div>

<div align="center" style="line-height: 1;">

[//]: # (  <a href='https://meituan-longcat.github.io/LongCat-Image/'><img src='https://img.shields.io/badge/Project-Page-green'></a>)
  <a href='https://huggingface.co/meituan-longcat/LongCat-Image'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LongCat--Image-blue'></a>
  <a href='https://huggingface.co/meituan-longcat/LongCat-Image-Dev'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LongCat--Image--Dev-blue'></a>
  <a href='https://huggingface.co/meituan-longcat/LongCat-Image-Edit'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-LongCat--Image--Edit-blue'></a>
</div>

## Introduction
**LongCat-Image-Dev** is a development variant of LongCat-Image, representing a mid-training checkpoint that is released to facilitate downstream development by the community, such as secondary fine-tuning via SFT, LoRA, and other customization methods. 
<div align="center">
  <img src="assets/model_struct.jpg" width="90%" alt="LongCat-Image Model Architecture" />
</div>

### Key Features

- 🔧 **True Developer-Ready Foundation**: Unlike typical release-the-final-model-only approaches, we provide the Dev—a high-plasticity, unconstrained state that avoids RL-induced rigidity. This enables seamless fine-tuning without fighting against over-aligned parameter spaces.

- 🛠️ **Full-Stack Training Framework**: We ship production-ready code for **SFT**, **LoRA fine-tuning**, **DPO/GRPO/MPO alignment**, and **specialized Edit training**. Every stage from pre-training data curation to reward model integration is reproducible, empowering researchers to build on our exact pipeline rather than reverse-engineering it.