# Repository Guidelines

This repository is documentation-first. The current contents live under `docs/` and describe the parameter-golf context. Use this guide to keep contributions clear, consistent, and easy to review.

## Project Structure & Module Organization
- `docs/`: Primary content (e.g., `golf-param-competition.md`, `plan.md`).
- Future code or assets should live alongside a brief `README` in their folder.
- Prefer small, focused directories with self-explanatory names (kebab-case).

## Build, Test, and Development Commands
- No build step is required for Markdown changes.
- Optional checks (recommended):
  - `prettier -w docs/**/*.md`: Format Markdown consistently.
  - `markdownlint docs/**/*.md`: Lint headings, lists, and links.
  - If you add scripts, include a `make check` target to run linters/tests.

## Coding Style & Naming Conventions
- Markdown: use `#`-based headings, sentence-case titles, and fenced code blocks.
- Line width: aim for 80–100 chars; wrap lists and paragraphs thoughtfully.
- Filenames: kebab-case (e.g., `evaluation-notes.md`).
- Links: prefer relative paths within `docs/`.

## Testing Guidelines
- Docs: validate links, code blocks, and commands you include.
- Scripts (if added): provide a minimal smoke test and document usage.
- Keep examples runnable; include exact commands and expected outputs when feasible.

## Commit & Pull Request Guidelines
- Commits: short, imperative subject (≤72 chars), body for context when needed.
  - Examples: `Update competition rules`, `Add link checks to CI`.
- Pull Requests: include purpose, key changes, and how to review.
  - Reference related issues, attach screenshots for docs that include diagrams, and list testing steps.

## Security & Configuration Tips (Optional)
- Do not commit secrets or tokens; use environment variables and `.env.local` (gitignored).
- If adding code, pin tool versions and document prerequisites in the folder `README`.

## Getting Started
- Edit or add files in `docs/`.
- Validate with formatting/linting tools if available, then open a PR with a concise summary and review steps.

