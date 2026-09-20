"""Model evaluation harness (SPEC §2, §11 M1).

Deliberately outside ``tests/``: it needs a real GPU and a real checkpoint, and SPEC §9
says no test may require a GPU. Nothing in CI runs this.

What it produces is the evidence behind a ``[profiles.*]`` block in
``config/farmhub.toml``: `weights_gb` and `kv_cache_gb` are measured here, and
``measured_by`` names the run that measured them.
"""
