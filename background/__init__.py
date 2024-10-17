from .background import SolidBackground, LatentSolidBackground, RandomSolidBackground, BlackWhiteBackground, NeRFBackground, CacheBackground

BACKGROUNDs = {
    "solid": SolidBackground,
    "cache": CacheBackground,
    "latent_solid": LatentSolidBackground,
    "random_solid": RandomSolidBackground,
    "black_white": BlackWhiteBackground,
    "nerf": NeRFBackground,
}