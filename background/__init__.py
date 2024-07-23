from .background import SolidBackground, LatentSolidBackground, RandomSolidBackground, BlackWhiteBackground

BACKGROUNDs = {
    "solid": SolidBackground,
    "latent_solid": LatentSolidBackground,
    "random_solid": RandomSolidBackground,
    "black_white": BlackWhiteBackground,
}