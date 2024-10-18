from .base import SDSSampler, SDISampler, DDIMSampler, GeneralizedDDIMSampler, ISMSampler

NOISE_SAMPLERs = {
    "sds": SDSSampler,
    "sdi": SDISampler,
    "ism": ISMSampler,
    "ddim": DDIMSampler,
    "generalized_ddim": GeneralizedDDIMSampler,
}

SAMPLERs_REQUIRING_PREV_EPS = [
    DDIMSampler,
    GeneralizedDDIMSampler,
]