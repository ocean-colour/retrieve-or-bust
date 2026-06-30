# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

retrieve-or-bust is our last best effort at solving the IOP (inherent optical
properties) inversion problem with AI. The goal is to develop and evaluate
methods that retrieve IOPs from ocean color / remote-sensing reflectance.

## Package layout

- `robust/` — the Python package source (**R**etrieve **O**r **BUST**).
- `claude_prompts/` — prompts and task definitions that drive this work.

## Environment

- Use the `ocean14` conda environment for running Python.

## Conventions

- Write clear, well-documented Python with docstrings.

## Git

- **The user (J. Xavier Prochaska) performs all git commands.** Do not run
  `git add`, `git commit`, `git push`, or any other git command that changes
  repository state unless explicitly asked. You may run read-only git commands
  (e.g. `git status`, `git diff`, `git log`) when helpful.
