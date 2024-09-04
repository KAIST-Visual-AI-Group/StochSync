from .linear_annealing import LinearAnnealingTimeSampler
from .sds import SDSTimeSampler

TIME_SAMPLERs = {
    "sds": SDSTimeSampler,
    "linear_annealing": LinearAnnealingTimeSampler,
}