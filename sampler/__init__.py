from .sds import SDSSampler, DreamTimeSampler, SDISampler, SDIppSampler

SAMPLERs = {
    "sds": SDSSampler,
    "dreamtime": DreamTimeSampler,
    "sdi": SDISampler,
    "sdipp": SDIppSampler,
}