"""Tool search for LLM agents, ranked by System-1 decision models."""

from importlib.metadata import version as _version

__all__ = ["__version__"]

__version__ = _version("toolhunch")
