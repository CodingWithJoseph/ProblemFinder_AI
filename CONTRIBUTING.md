# Contributing Guidelines

Thank you for your interest in improving ProblemFinder! This document outlines the expectations for contributors to keep the codebase healthy and maintainable.

## Development Workflow
1. **Fork and branch** – Create feature branches using the pattern `feature/<topic>` or `fix/<issue>`.
2. **Environment** – Use Python 3.10+ with the dependencies pinned in `requirements.txt`. Install development tooling (`black`, `ruff`, `pytest`).
3. **Coding standards**
   - Follow PEP 8/484 and maintain full type annotations.
   - Use Google-style docstrings for all public functions, classes, and dataclasses.
   - Keep modules deterministic when possible; surface randomness via explicit seeds.
4. **Formatting & linting**
   - Run `black .` and `ruff .` before submitting a PR.
   - Ensure imports remain sorted and unused symbols removed.
5. **Testing**
   - Add or update unit tests in `tests/` for each new feature or bug fix.
   - Execute `pytest` locally and confirm all tests pass.
6. **Commit messages**
   - Write descriptive commits using the format `<type>: <summary>` (e.g., `feat: add cache TTL configuration`).
   - Group related changes into a single commit when practical.
7. **Pull request checklist**
   - Update documentation (`README.md`, `docs/architecture.md`, notebooks) when behaviour changes.
   - Include screenshots or logs for user-facing updates when relevant.
   - Reference related issues and provide a concise summary of changes in the PR description.

We appreciate bug reports, test coverage improvements, and refactors that increase clarity. Thank you for helping build a reliable, enterprise-grade classification pipeline!
