from dataclasses import dataclass
import torch
from nvdiffrast.torch import rasterize, interpolate
from k_utils.image_utils import torch_to_pil, concat_images
import shared_modules as sm
from data import DATASETs
from model import MODELs
from prior import PRIORs
import os
from PIL import Image

@dataclass
class Config:
    root_dir: str = "./results/test"
    text_prompt: str = "a zoomed out DSLR photo of a baby dragon"
    max_steps: int = 2000
    batch_size: int = 1
    dataset: str = "random"
    convention: str = "OpenGL"
    up_vec: str = "y"
    fov: float = 60
    background: str = "solid"
    model: str = "mesh"
    prior: str = "controlnet"
    height: int = 768
    width: int = 768
    guidance_scale: float = 7.5
    mesh_path: str = "face.obj"
cfg = Config()
cfg_dict = cfg.__dict__

sm.dataset = DATASETs[cfg.dataset](cfg_dict)
sm.prior = PRIORs[cfg.prior](cfg_dict)
sm.model = MODELs[cfg.model](cfg_dict)
sm.model.prepare_optimization()

scheduler = sm.prior.scheduler
scheduler.set_timesteps(50)
pipeline = sm.prior.pipeline
predict = sm.prior.predict
get_tweedie = sm.prior.get_tweedie
get_eps = sm.prior.get_eps
get_noisy_sample = sm.prior.get_noisy_sample
decode_latent = sm.prior.decode_latent


dist = "/home/aaaaa/data/texgen/gt"
os.makedirs(dist, exist_ok=True)

models = [  
    "eb219212147f4d84b88f8e103af8ea10",
    "eb219212147f4d84b88f8e103af8ea10",
    "a8813ea1e0ce47ab97a416637a7520d7",
    "a8813ea1e0ce47ab97a416637a7520d7",
    "e0417d1e05984727a50f9ab1451d162d",
    "e0417d1e05984727a50f9ab1451d162d",
    "9fa2da2c42234b58896e8d23393cac24",
    "9fa2da2c42234b58896e8d23393cac24",
    "9fa2da2c42234b58896e8d23393cac24",
    "a51751c9989940e592eb61be41ee35cc",
    "a51751c9989940e592eb61be41ee35cc",
    "f73e2e1c8ad241ff859aca7e032ec262",
    "f73e2e1c8ad241ff859aca7e032ec262",
    "91c5283b27c74583900d5e26e2fcd086",
    "91c5283b27c74583900d5e26e2fcd086",
    "b6db59bd7f10424eae54c71d19663a65",
    "b6db59bd7f10424eae54c71d19663a65",
    "a2832b845e4e4edd9d439342cf4fd590",
    "a2832b845e4e4edd9d439342cf4fd590",
    "b19ef2650b4347348710eb6364ca90bd",
    "b19ef2650b4347348710eb6364ca90bd",
    "bd384d46514548cf8c4202f1ae6ea551",
    "bd384d46514548cf8c4202f1ae6ea551",
    "f1aa479977a74a608d362679ed5ca721",
    "f1aa479977a74a608d362679ed5ca721",
    "4c4690ba918f477b829990dd2e960c21",
    "4c4690ba918f477b829990dd2e960c21",
    "f87caf6ac5a445ccad1a97653688e16e",
    "f87caf6ac5a445ccad1a97653688e16e",
    "f15298421b3d4e0fab4c43863a7e72fd",
    "f15298421b3d4e0fab4c43863a7e72fd",
    "d4c560493a0846c5943f3aeea58acb72",
    "d4c560493a0846c5943f3aeea58acb72",
    "c6509a8fe1f44a5eac8aebe12be2699e",
    "c6509a8fe1f44a5eac8aebe12be2699e",
    "fa2c41a7a6c84fcb871a24016fa9a932",
    "fa2c41a7a6c84fcb871a24016fa9a932",
    "f05b0c2f9bcf41cea188a4b4c848068a",
    "f05b0c2f9bcf41cea188a4b4c848068a",
    "bff537fb09b641c59b2ad123da0ca3dc",
    "bff537fb09b641c59b2ad123da0ca3dc",
    "d726514a97f74f168b104fd6ba538331",
    "d726514a97f74f168b104fd6ba538331",
    "01ab0842feb1448bb18e8c7b85326d11",
    "01ab0842feb1448bb18e8c7b85326d11",
    "f2d31eb0ddac4d21944df7dcc4af6d28",
    "f2d31eb0ddac4d21944df7dcc4af6d28",
    "fc9cc06615084298b4c0c0a02244f356",
    "fc9cc06615084298b4c0c0a02244f356",
    "7adc9c74b75e4860b0a51c850bde9957",
    "7adc9c74b75e4860b0a51c850bde9957",
    "2fc0fc6ebe564a249c4617e6b3e6da93",
    "2fc0fc6ebe564a249c4617e6b3e6da93",
    "14b8ae60eae240ff8bf1abdf9af5e49c",
    "14b8ae60eae240ff8bf1abdf9af5e49c",
    "62897c52e967469c85df9c6abdd09d16",
    "62897c52e967469c85df9c6abdd09d16",
    "6f5480698a7a43c7a8c0a8b1e295e4a0",
    "6f5480698a7a43c7a8c0a8b1e295e4a0",
    "e1f96691aaf648b885d927f5c3f5be61",
    "e1f96691aaf648b885d927f5c3f5be61",
    "8a60954eccad433e987bbcafc7657140",
    "8a60954eccad433e987bbcafc7657140",
    "f98c5ee54c4a48f8b5eafd35a81dde4d",
    "f98c5ee54c4a48f8b5eafd35a81dde4d",
    "fadefc1eee3246a189f6b79c7c671343",
    "fadefc1eee3246a189f6b79c7c671343",
    "9a0c52d350634e419aaf0eea1e67d9da",
    "9a0c52d350634e419aaf0eea1e67d9da",
    "0db114d7753344d6825aa4f21ec56db9",
    "0db114d7753344d6825aa4f21ec56db9",
    "72826cd5c17a42798a8e8e36c05c5035",
    "72826cd5c17a42798a8e8e36c05c5035",
    "ac5df73de2c54239833643423a152592",
    "ac5df73de2c54239833643423a152592",
    "90009fa6fa0b4d4bb1a1203431954097",
    "90009fa6fa0b4d4bb1a1203431954097",
    "b26a53419075442ca284cdf1d5541765",
    "b26a53419075442ca284cdf1d5541765",
    "f75caead1dc1474195eb32a7f4c71117",
    "f75caead1dc1474195eb32a7f4c71117",
    "edbeb81ef32645cea8bef89338f7e213",
    "edbeb81ef32645cea8bef89338f7e213",
    "napoleon",
    "napoleon",
    "napoleon",
    "dragon",
    "dragon",
    "francois",
    "francois",
    "provost",
    "provost",
]
prompts = [
    "A robotic frog",
    "A green frog",
    "A Mandalorian helmet in silver",
    "A black helmet",
    "A stone lantern",
    "A medieval lantern",
    "A backpack in ironman style",
    "A backpack in spiderman style",
    "A 3D backpack",
    "A baby owl with fluffy wings",
    "A toy owl",
    "A cute 3D cartoon lion with brown hair",
    "A marble lion",
    "A wooden mug surrounded by silver rings",
    "A mug with cloud",
    "A next gen nascar in red",
    "A next gen nascar",
    "Statue of a wolf",
    "A white wolf",
    "A black penguin",
    "A penguin covered by a blue sweater",
    "A wooden refrigerator",
    "A high tech refrigerator",
    "A medieval piano",
    "A piano with flowers",
    "A golden lion",
    "A cyber punk lion",
    "A wooden dresser",
    "A marble dresser",
    "A deep ocean shark",
    "A dark blue shark",
    "A soccer ball in black and white",
    "A stone soccer ball",
    "A tiger walking on the grass",
    "A plastic toy tiger",
    "A chocolate doughnut",
    "An icecream doughnut",
    "A fireplug, red and yellow",
    "A fireplug with yellow top",
    "A metal turtle with red eyes",
    "A sea turtle",
    "An ancient vase",
    "A painted vase",
    "An antique pottery",
    "A pottery with flowers",
    "A coca cola vending machine",
    "A silver vending machine",
    "A medieval piano",
    "A piano with flowers",
    "A princess dress",
    "A dress with spider patterns",
    "A brick fireplace",
    "A stone fireplace",
    "A wooden refrigerator",
    "A high tech refrigerator",
    "A doll with yellow hairs",
    "A spiderman doll",
    "A pumpkin with red eyes",
    "A Halloween pumpkin",
    "A red apple",
    "An oil painted apple",
    "A medieval armor",
    "A Japanese armor",
    "A metal owl with glowing eyes",
    "A wooden owl",
    "A lion looking forward",
    "Statue of a lion",
    "A golden knight",
    "A silver knight",
    "A wooden crate",
    "A bronze crate",
    "A medieval clock",
    "A electric clock",
    "A wooden dresser",
    "A marble dresser",
    "A metal keg in silver",
    "A wooden keg",
    "A mac monitor",
    "An ironman monitor",
    "A game controller with black buttons on the top",
    "A PS5 controller",
    "A telephone with golden dials",
    "A classic telephone",
    "A high quality color photo of Tom Cruise",
    "A high quality color photo of Benedict Cumberbatch",
    "A high quality color photo of Robert Downey Jr.",
    "Cartoon dragon, red and green",
    "A 3D dragon",
    "Spiderman with white hairs",
    "A boy in suits",
    "Portrait of Provost, oil paint",
    "A statue of Provost",
]

def attach(*args):
    return torch.cat(args, dim=-1)

def take(x, a):
    B, C, H, W = x.shape
    b = int(H * a)
    return x[..., b:b+H]

def insert(x, y, a):
    B, C, H, W = x.shape
    b = int(W * a)
    # if x width not enough, pad it
    x = x.clone()
    if b + H > W:
        x = F.pad(x, (0, b + H - W))
    x[..., b:b+H] = y
    return x

def visualize(x):
    if isinstance(x, list):
        if isinstance(x[0], Image.Image):
            return concat_images(x, 1, len(x))
        if x[0].shape[1] == 3:
            x = torch.cat(x, dim=3)
        else:
            x = [sm.prior.decode_latent(x_) for x_ in x]
            x = torch.cat(x, dim=3)
    else:
        x = sm.prior.decode_latent_if_needed(x)
        x = torch.cat([x_.unsqueeze(0) for x_ in x], dim=3)
    return torch_to_pil(x)

@torch.no_grad()
def generate_image(model, prompt):
    cameras = sm.dataset.params_to_cameras(
        [2.5] * 10,
        [11, 7, 7, 9, 6, 24, 25, 5, 14, 26],
        [25, 32, 84, 132, 138, 144, 147, 232, 295, 355],
    )
    cfg_dict['mesh_path'] = f"/home/aaaaa/data/texgen/processed/{model}.obj"
    sm.model = MODELs[cfg.model](cfg_dict)
    sm.model.prepare_optimization()
    sm.prior.prev_camera_hash = None
    sm.prior.cfg.text_prompt = prompt
    latent = sm.prior.ddim_loop(cameras, torch.randn(10, 4, 96, 96, device='cuda'), 999, 0, num_steps=50)
    images = sm.prior.decode_latent(latent)
    alphas = sm.model.render(cameras)['alpha']
    images = images * alphas + (1 - alphas)
    images = [torch_to_pil(image) for image in images]
    return images

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5)
    parser.add_argument("--i", type=int, default=2)
    args = parser.parse_args()
    n = args.n
    i = args.i
    print(f"n = {n}")
    print(f"i = {i}")

    # n = 5
    # i = 2
    #n, i
    indices = list(range(len(models)))
    indices = indices[i::n]
    print(f"Indices: {indices}")
    for j in indices:
        model = models[j]
        prompt = prompts[j]
        print(f"Processing {model} {prompt}")
        images = generate_image(model, prompt)
        for j, image in enumerate(images):
            image.save(f"{dist}/{model}_{prompt.replace(' ', '_')}_{j}.png")