# ISyntaxWSI Integration Specification

## Goal

Add support for Philips iSyntax files in TRIDENT by implementing an `ISyntaxWSI` class. This class should integrate with TRIDENT’s existing WSI interface structure and be selectable via factory functions.

## Context

- Based on `pyisyntax` (already in `pyproject.toml`)
- Existing iSyntax handling is tested in `test_isyntaxwsi.py`
- Analogous to existing WSI classes (e.g., `OpenSlideWSI`)
- Must support tile access, metadata access, and image region reading.

## Class: `ISyntaxWSI`
