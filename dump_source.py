
import inspect
from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
import sys

try:
    source = inspect.getsource(ZImageTransformer2DModel.forward)
    # Output manually to avoid buffer issues? No, standard print is fine if not interleaved.
    print(source)
except Exception as e:
    print(e)
