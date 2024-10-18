from .base import SDSTimeSampler, LinearAnnealingTimeSampler, RepeatingTimeSampler, GoodTimeSampler

TIME_SAMPLERs = {
    "sds": SDSTimeSampler,
    "linear_annealing": LinearAnnealingTimeSampler,
    "repeating": RepeatingTimeSampler,
    "good": GoodTimeSampler,
}