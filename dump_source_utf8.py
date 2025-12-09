
import inspect
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
import sys

try:
    source = inspect.getsource(ZImageTransformer2DModel.forward)
    with open('d:/AI/None_Z-image-Turbo_trainer/zimage_forward_source.py', 'w', encoding='utf-8') as f:
        f.write(source)
    print("Source dumped successfully.")
except Exception as e:
    print(f"Error: {e}")
