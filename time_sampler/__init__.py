from .base import SDSTimeSampler, LinearAnnealingTimeSampler, RepeatingTimeSampler

TIME_SAMPLERs = {
    "sds": SDSTimeSampler,
    "linear_annealing": LinearAnnealingTimeSampler,
    "repeating": RepeatingTimeSampler,
}