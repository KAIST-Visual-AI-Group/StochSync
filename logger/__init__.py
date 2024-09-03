from .logger import NullLogger, SimpleLogger, SimpleLatentRawLogger, SimpleLatentLogger, ProcedureLogger, RendererLogger, LatentRendererLogger, SelfLogger

LOGGERs = {
    "null": NullLogger,
    "simple": SimpleLogger,
    "simple_latent_raw": SimpleLatentRawLogger,
    "simple_latent": SimpleLatentLogger,
    "procedure": ProcedureLogger,
    "renderer": RendererLogger,
    "renderer_latent": LatentRendererLogger,
    "self": SelfLogger,
}
