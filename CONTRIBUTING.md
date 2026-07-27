# Contributing to NeuroPager

Thank you for your interest in contributing to NeuroPager. This project is a
research-grade effort to build an operating-system-inspired memory
architecture for lifelong LLM agents, and we welcome contributions from
researchers, engineers, and enthusiasts alike.

> **Project status:** NeuroPager is in its early scaffolding stage. Core
> algorithms are not yet implemented — see [ROADMAP](docs/roadmap.md) for
> what's planned and where you can help.

## Code of Conduct

This project and everyone participating in it is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to
uphold this code.

## Ways to Contribute

- **Bug reports** — file an issue using the bug report template.
- **Feature proposals** — file an issue using the feature request template.
- **Documentation** — improvements to `docs/`, docstrings, or examples.
- **Research** — design discussion around memory architectures, page
  replacement policies, or retrieval strategies (see
  [docs/research.md](docs/research.md)).
- **Code** — once core interfaces stabilize, implementation PRs against the
  package modules in `src/neuropager/`.

## Development Setup

```bash
git clone https://github.com/kavimugilsk/neuropager.git
cd neuropager
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
pre-commit install
```

## Project Layout

See [docs/architecture.md](docs/architecture.md) for a full architectural
overview and [docs/design.md](docs/design.md) for module-level design notes.

## Coding Standards

- **Python version:** 3.11+
- **Formatting:** [Black](https://github.com/psf/black) (line length 100)
- **Linting:** [Ruff](https://github.com/astral-sh/ruff)
- **Type checking:** `mypy --strict`
- **Docstrings:** [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- **Tests:** [pytest](https://docs.pytest.org/)

Run the full local check suite before opening a PR:

```bash
ruff check .
black --check .
mypy src
pytest
```

Or simply:

```bash
make check
```

## Commit Messages

We loosely follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add LRU page replacement policy interface
fix: correct page fault handler docstring
docs: expand architecture.md with retrieval flow
test: add fixtures for working memory
chore: update pre-commit hooks
```

## Pull Request Process

1. Fork the repository and create your branch from `main`.
2. If you've added an interface or module, update the relevant `docs/` page.
3. Ensure `make check` passes locally.
4. Open a PR with a clear description of the motivation and design.
5. Link any related issues.
6. A maintainer will review and may request changes before merging.

## Design Discussions

Because NeuroPager aims to be research-grade, non-trivial design changes
(new memory subsystems, new page replacement policies, changes to the page
fault handling model) should be proposed as an issue or discussion **before**
a PR, so the approach can be validated against the project's
[research vision](docs/research.md).

## License

By contributing, you agree that your contributions will be licensed under
the project's [MIT License](LICENSE).
