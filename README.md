# Work in Progress!

# PyChemelt

[![Tests](https://github.com/osvalB/pychemelt/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/osvalB/pychemelt/actions/workflows/ci-cd.yml)
[![codecov](https://codecov.io/gh/osvalB/pychemelt/graph/badge.svg)](https://codecov.io/gh/osvalB/pychemelt)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://osvalb.github.io/pychemelt)


# Install from Source - Development

Run these commands in your shell:

```bash
git clone https://github.com/osvalB/pychemelt.git
cd pychemelt
uv sync --extra dev
```

# Verify Installation

Run the following to verify the installation:

```bash
uv run pytest
uv run build_docs.py
```
