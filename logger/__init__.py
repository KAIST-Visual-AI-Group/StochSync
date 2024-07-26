from .logger import SimpleLogger, SimpleLatentRawLogger, SimpleLatentLogger, ProcedureLogger, RendererLogger, LatentRendererLogger

LOGGERs = {
    "simple": SimpleLogger,
    "simple_latent_raw": SimpleLatentRawLogger,
    "simple_latent": SimpleLatentLogger,
    "procedure": ProcedureLogger,
    "renderer": RendererLogger,
    "renderer_latent": LatentRendererLogger,
}
