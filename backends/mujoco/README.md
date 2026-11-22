# Robot and material backend

Current UR10e/retained UR5e robot profiles, tensor FDCC and packed DiSECt
are used by the root training/task adapters. The native BladeMaster adapter lives
in cutting_world_model/blademaster_backend.py and uses the external source there configured.

Use the root environment and explicit current generated configs:
[Operations](../../docs/OPERATIONS.md), [Method](../../docs/METHOD.md).
Do not use the old public-prism benchmark defaults as a five-food launch recipe.
[Historical source/probes](../../docs/reference/SOURCES.md#historical-backend-probes)
are provenance, not current runtime configuration.
[Licenses](../../NOTICE.md).
