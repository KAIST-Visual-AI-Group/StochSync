from .sds import SDSSampler, DreamTimeSampler, SDISampler

SAMPLERs = {
    "sds": SDSSampler,
    "dreamtime": DreamTimeSampler,
    "sdi": SDISampler,
}