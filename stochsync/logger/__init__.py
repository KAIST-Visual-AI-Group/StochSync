from .logger import NullLogger, SimpleLogger, SimpleLatentPreviewLogger, SimpleLatentLogger, RendererLogger, LatentRendererLogger, SelfLogger

LOGGERs = {
    "null": NullLogger,
    "simple": SimpleLogger,
    "simple_latent_preview": SimpleLatentPreviewLogger,
    "simple_latent": SimpleLatentLogger,
    # "procedure": ProcedureLogger,
    "renderer": RendererLogger,
    "renderer_latent": LatentRendererLogger,
    "self": SelfLogger,
}
