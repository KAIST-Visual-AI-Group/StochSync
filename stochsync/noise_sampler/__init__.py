from .base import SDSSampler, SDISampler, DDIMSampler, GeneralizedDDIMSampler

NOISE_SAMPLERs = {
    "sds": SDSSampler,
    "sdi": SDISampler,
    "ddim": DDIMSampler,
    "generalized_ddim": GeneralizedDDIMSampler,
}

SAMPLERs_REQUIRING_PREV_EPS = [
    DDIMSampler,
    GeneralizedDDIMSampler,
]