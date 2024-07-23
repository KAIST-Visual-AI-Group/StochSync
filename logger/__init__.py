from .logger import SimpleLogger, SimpleLatentLogger, ProcedureLogger, RendererLogger, LatentRendererLogger

LOGGERs = {
    "simple": SimpleLogger,
    "simple_latent": SimpleLatentLogger,
    "procedure": ProcedureLogger,
    "renderer": RendererLogger,
    "renderer_latent": LatentRendererLogger,
}
