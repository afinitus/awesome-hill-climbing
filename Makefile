# Common tasks for lastmile. Run `make help` to list them.

.DEFAULT_GOAL := help
.PHONY: help setup test validate leaderboard smoke awesome links

help:  ## list the targets
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  make %-12s %s\n", $$1, $$2}'

# The repo often lives on a Mac Desktop, which iCloud may sync. iCloud skips any folder whose name
# ends in ".nosync", so the virtualenv goes in .venv.nosync (thousands of small files that should
# never be uploaded), and .venv is a symlink to it so `uv run` and editors still find it.
setup:  ## create the virtualenv with the dev and BoTorch extras
	UV_PROJECT_ENVIRONMENT=.venv.nosync uv sync --extra dev --extra bo
	@if [ ! -e .venv ]; then ln -s .venv.nosync .venv; fi
	@if [ ! -L .venv ]; then echo "note: .venv is a real folder, not a symlink to .venv.nosync; remove it and rerun"; fi

test:  ## run the fast test suite
	uv run pytest -q

validate:  ## check the environment against the design contract
	uv run python tools/validate_env.py

leaderboard:  ## rebuild the README table and media/leaderboard/ plots from results/
	uv run python tools/leaderboard.py

# Every algorithm file must finish with --quick in under 2 minutes; CI runs the same loop.
# Quick runs write results files, but the leaderboard leaves them out (config.quick is true).
smoke:  ## run every algos/[0-9]*.py with --quick
	@files=$$(ls algos/[0-9]*.py 2>/dev/null); \
	if [ -z "$$files" ]; then echo "no algos/[0-9]*.py files yet; nothing to smoke-test"; exit 0; fi; \
	for f in $$files; do echo "== $$f --quick"; uv run python $$f --quick || exit 1; done

awesome:  ## rebuild README.md and papers/*.md from data/ (the Awesome list)
	uv run python tools/build_awesome.py

links:  ## validate every arXiv link, title and date in data/papers.csv against the arXiv API
	uv run python tools/arxiv_meta.py --dry-run
