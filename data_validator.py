#!/usr/bin/env python3
"""Backward-compatible entry point: ``python3 data_validator.py`` still works.

The implementation now lives in the ``data_validator/`` package. This file only
forwards to ``python3 -m data_validator`` so existing commands, docs and
scheduled jobs keep working. ``import data_validator`` resolves to the package,
not to this script.
"""
import runpy

if __name__ == "__main__":
    runpy.run_module("data_validator", run_name="__main__")
