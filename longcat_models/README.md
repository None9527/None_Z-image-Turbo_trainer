---
license: apache-2.0
language:
- en
- zh
pipeline_tag: text-to-image
library_name: transformers
---

<div align="center">
  <img src="https://github.com/meituan-longcat/LongCat-Flash-Chat/blob/main/figures/longcat_logo.svg" width="45%" alt="LongCat-Image" />
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
We introduce **LongCat-Image**, a pioneering open-source and bilingual (Chinese-English) foundation model for image generation, designed to address core challenges in multilingual text rendering, photorealism, deployment efficiency, and developer accessibility prevalent in current leading models.
<div align="center">
  <img src="assets/model_struct.jpg" width="90%" alt="LongCat-Image Generation Examples" />
</div>


### Key Features
- 🌟 **Exceptional Efficiency and Performance**: With only **6B parameters**, LongCat-Image surpasses numerous open-source models that are several times larger across multiple benchmarks, demonstrating the immense potential of efficient model design.
- 🌟 **Powerful Chinese Text Rendering**: LongCat-Image demonstrates superior accuracy and stability in rendering common Chinese characters compared to existing SOTA open-source models and achieves industry-leading coverage of the Chinese dictionary.
- 🌟 **Remarkable Photorealism**: Through an innovative data strategy and training framework, LongCat-Image achieves remarkable photorealism in generated images.

[//]: # (For more details, please refer to the comprehensive [***LongCat-Image Technical Report***]&#40;https://arxiv.org/abs/2412.11963&#41;.)

## 🎨 Showcase

<div align="center">
  <img src="assets/gallery.jpeg" width="90%" alt="LongCat-Image Generation Examples" />
</div>

## Quick Start

### Installation

Clone the repo:

```shell
git clone --single-branch --branch main https://github.com/meituan-longcat/LongCat-Image
cd LongCat-Image
```

Install dependencies:

```shell
# create conda environment
conda create -n longcat-image python=3.10
conda activate longcat-image

# install other requirements
pip install -r requirements.txt
python setup.py develop
```

### Run Text-to-Image Generation
**💡 Tip**: Using a stronger LLM model for prompt engineering can further improve image generation quality. Please refer to [inference_t2i.py](https://github.com/meituan-longcat/LongCat-Image/blob/main/scripts/inference_t2i.py#L28) for detailed usage.
```shell
import torch
from transformers import AutoProcessor
from longcat_image.models import LongCatImageTransformer2DModel
from longcat_image.pipelines import LongCatImagePipeline

device = torch.device('cuda')
checkpoint_dir = './weights/LongCat-Image'

text_processor = AutoProcessor.from_pretrained( checkpoint_dir, subfolder = 'tokenizer'  )
transformer = LongCatImageTransformer2DModel.from_pretrained( checkpoint_dir , subfolder = 'transformer', 
    torch_dtype=torch.bfloat16, use_safetensors=True).to(device)

pipe = LongCatImagePipeline.from_pretrained(
    checkpoint_dir,
    transformer=transformer,
    text_processor=text_processor
)
pipe.to(device, torch.bfloat16)

prompt = '一个年轻的亚裔女性，身穿黄色针织衫，搭配白色项链。她的双手放在膝盖上，表情恬静。背景是一堵粗糙的砖墙，午后的阳光温暖地洒在她身上，营造出一种宁静而温馨的氛围。镜头采用中距离视角，突出她的神态和服饰的细节。光线柔和地打在她的脸上，强调她的五官和饰品的质感，增加画面的层次感与亲和力。整个画面构图简洁，砖墙的纹理与阳光的光影效果相得益彰，突显出人物的优雅与从容。'

image = pipe(
    prompt,
    height=768,
    width=1344,
    guidance_scale=4.5,
    num_inference_steps=50,
    num_images_per_prompt=1,
    generator=torch.Generator("cpu").manual_seed(43),
    enable_cfg_renorm=True,
    enable_prompt_rewrite=True # Reusing the text encoder as a built-in prompt rewriter
).images[0]
image.save('./t2i_example.png')
```