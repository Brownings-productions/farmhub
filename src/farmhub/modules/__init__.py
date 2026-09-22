"""Pluggable modules (SPEC §8).

Modules never import each other. Cross-module needs go through ``AppContext`` or the
event bus, and ``core`` never imports a module — the explicit list lives in ``app.py``.
"""
