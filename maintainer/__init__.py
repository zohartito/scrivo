"""The maintainer-only sealed adaptive lane (workers, silver store, teachers,
calibration, deployment and sealed-release tooling).

The alpha imports nothing from here except through gated edges (``--adaptive``,
the lane's learning-* commands, or a data folder the lane has written), and
the alpha source archive leaves this package out. Every file here is in
``runtime_source_manifest.RUNTIME_SOURCE_FILES``, so the sealed boundary still
covers it.
"""
