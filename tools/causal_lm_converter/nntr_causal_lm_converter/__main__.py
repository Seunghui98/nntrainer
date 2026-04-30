"""Allow ``python -m nntr_causal_lm_converter ...``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
