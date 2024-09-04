from .sds import SDSSampler, DreamTimeSampler, SDISampler, SDIppSampler, HifaSampler
from .csd import CSDSampler, DreamTimeCSDSampler, BSDSampler, BSDISampler
SAMPLERs = {
    "sds": SDSSampler,
    #"dreamtime": DreamTimeSampler,
    "hifa": HifaSampler,
    "sdi": SDISampler,
    "sdipp": SDIppSampler,
    "csd": CSDSampler,
    "dreamtime_csd": DreamTimeCSDSampler,
    "bsd": BSDSampler,
    "bsdi": BSDISampler
}