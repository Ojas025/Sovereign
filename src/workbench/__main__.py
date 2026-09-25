"""Allow ``python -m workbench`` as an alias for the console script."""

from workbench.cli import main

raise SystemExit(main())
