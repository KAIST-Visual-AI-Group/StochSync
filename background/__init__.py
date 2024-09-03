from .background import SolidBackground, LatentSolidBackground, RandomSolidBackground, BlackWhiteBackground, NeRFBackground

BACKGROUNDs = {
    "solid": SolidBackground,
    "latent_solid": LatentSolidBackground,
    "random_solid": RandomSolidBackground,
    "black_white": BlackWhiteBackground,
    "nerf": NeRFBackground,
}