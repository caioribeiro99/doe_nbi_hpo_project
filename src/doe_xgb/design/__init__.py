"""DesignProvider abstraction.

Loads or generates experimental designs and returns a :class:`DesignArtifact`
with both coded and uncoded matrices, full metadata, and validation
diagnostics.

The article-track methodology is anchored in CCD/CCFCD designs imported
from Minitab; pure-NumPy generators are also provided for the most common
designs so the framework can run without that dependency.
"""

from .provider import (  # noqa: F401
    DesignArtifact,
    DesignKind,
    DesignProvider,
    DesignSpec,
    FactorMeta,
    ValidationReport,
    build,
    validate_for_model,
)

__all__ = [
    "DesignArtifact",
    "DesignKind",
    "DesignProvider",
    "DesignSpec",
    "FactorMeta",
    "ValidationReport",
    "build",
    "validate_for_model",
]
