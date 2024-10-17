from .base import SDSSampler, SDISampler, DDIMSampler, RandomizedDDIMSampler, GeneralizedDDIMSampler, RandomizedSDISampler, ISMSampler

NOISE_SAMPLERs = {
    "sds": SDSSampler,
    "sdi": SDISampler,
    "ism": ISMSampler,
    "ddim": DDIMSampler,
    "randomized_ddim": RandomizedDDIMSampler,
    "generalized_ddim": GeneralizedDDIMSampler,
    "randomized_sdi": RandomizedSDISampler,
}