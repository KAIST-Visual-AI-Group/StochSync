import torch
from diffusers.utils import load_image

from diffusers import FluxControlNetPipeline, FluxControlNetModel
from diffusers.models import FluxMultiControlNetModel

base_model = 'black-forest-labs/FLUX.1-dev'
controlnet_model_union = 'Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro'

controlnet_union = FluxControlNetModel.from_pretrained(controlnet_model_union, torch_dtype=torch.bfloat16)
controlnet = FluxMultiControlNetModel([controlnet_union]) # we always recommend loading via FluxMultiControlNetModel

pipe = FluxControlNetPipeline.from_pretrained(base_model, controlnet=controlnet, torch_dtype=torch.bfloat16)
pipe.to("cuda")

prompt = 'A bohemian-style female travel blogger with sun-kissed skin and messy beach waves.'
control_image_depth = load_image("https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro/resolve/main/assets/depth.jpg")
control_mode_depth = 2
# crop the image so that the width and the height is divisible by 8
shape = control_image_depth.size
new_shape = (shape[0] - shape[0] % 8, shape[1] - shape[1] % 8)
control_image_depth = control_image_depth.resize(new_shape)

control_image_canny = load_image("https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro/resolve/main/assets/canny.jpg")
control_mode_canny = 0
# crop the image so that the width and the height is divisible by 8
shape = control_image_canny.size
new_shape = (shape[0] - shape[0] % 8, shape[1] - shape[1] % 8)
control_image_canny = control_image_canny.resize(new_shape)

width, height = control_image_depth.size

image = pipe(
    prompt, 
    control_image=[control_image_depth, control_image_canny],
    control_mode=[control_mode_depth, control_mode_canny],
    width=width,
    height=height,
    controlnet_conditioning_scale=[0.2, 0.4],
    num_inference_steps=24, 
    guidance_scale=3.5,
    generator=torch.manual_seed(42),
).images[0]

# save
image.save("output.jpg")