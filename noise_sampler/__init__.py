from .base import SDSSampler, SDISampler, DDIMSampler, RandomizedDDIMSampler, GeneralizedDDIMSampler, RandomizedSDISampler

NOISE_SAMPLERs = {
    "sds": SDSSampler,
    "sdi": SDISampler,
    "ddim": DDIMSampler,
    "randomized_ddim": RandomizedDDIMSampler,
    "generalized_ddim": GeneralizedDDIMSampler,
    "randomized_sdi": RandomizedSDISampler,
}