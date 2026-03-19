# Repository Guidelines

## Project Structure & Module Organization
- Primary content lives in `docs/` (e.g., `golf-param-competition.md`, `plan.md`).
- Add future code or assets in focused, kebab-case folders (e.g., `tools/link-checker/`) with a brief `README.md` in that folder.
- Prefer small, self-contained modules; keep related examples and diagrams alongside their doc.
- Use relative links within `docs/` to reference files and sections.

## Build, Test, and Development Commands
- No build step is required for Markdown-only changes.
- Format Markdown:
  - `prettier -w docs/**/*.md`
- Lint Markdown:
  - `markdownlint docs/**/*.md`
- If you add scripts, provide a `Makefile` with `make check` that runs linters/tests.

## Coding Style & Naming Conventions
- Markdown: `#`-based headings; sentence-case titles; fenced code blocks with language hints.
- Line width: aim for 80–100 characters; wrap paragraphs and lists thoughtfully.
- Filenames and directories: kebab-case (e.g., `evaluation-notes.md`, `scenario-examples/`).
- Links: prefer relative paths (e.g., `../docs/plan.md`).

## Testing Guidelines
- Docs: validate links and commands you include; keep examples runnable with exact invocations and expected snippets of output.
- Scripts (if added): include a minimal smoke test and document usage in the folder `README.md`.
- Centralize checks under `make check` when present.

## Commit & Pull Request Guidelines
- Commits: short, imperative subject (≤72 chars) with optional body for context.
  - Examples: `Update competition rules`, `Add link checks to CI`.
- Pull requests: state purpose, key changes, and review steps; reference related issues; include screenshots for diagrams and list testing steps.

## Security & Configuration Tips
- Do not commit secrets or tokens. Use environment variables and a gitignored `.env.local` when needed.
- If adding code, pin tool versions and document prerequisites in the folder `README.md`.

## Getting Started
- Edit or add files in `docs/`.
- Run `prettier` and `markdownlint` (or `make check` if available).
- Open a PR with a concise summary and clear review steps.
