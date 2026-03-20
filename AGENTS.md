# Repository Guidelines

This guide summarizes how to contribute effectively to this docs‑first repository.

## Project structure & module organization

- Primary content lives in `docs/` (e.g., `golf-param-competition.md`, `plan.md`).
- Add code or assets in focused, kebab-case folders under `tools/` (e.g.,
  `tools/link-checker/`) with a short `README.md` in that folder.
- Keep examples and diagrams alongside their doc; use relative links within
  `docs/` (e.g., `../docs/plan.md`).

## Build, test, and development commands

- No build step is required for Markdown-only changes.
- Format Markdown: `prettier -w docs/**/*.md`
- Lint Markdown: `markdownlint docs/**/*.md`
- If you add scripts, provide a `Makefile` with `make check` that runs all
  linters/tests.

## Coding style & naming conventions

- Use `#`-based headings; sentence-case titles.
- Aim for 80–100 character lines; wrap thoughtfully.
- Use fenced code blocks with language hints (e.g., ```bash).
- Filenames and directories use kebab-case (e.g., `evaluation-notes.md`,
  `scenario-examples/`).
- Prefer relative links between docs and examples.

## Testing guidelines

- Docs: validate links and commands; keep examples runnable with exact
  invocations and expected output snippets.
- Scripts (if added): include a minimal smoke test and document usage in the
  folder `README.md`.
- Centralize checks under `make check` when present.

## Commit & pull request guidelines

- Commits: short, imperative subject (≤72 chars) with optional body.
  Examples: `Update competition rules`, `Add link checks to CI`.
- Pull requests: state purpose, key changes, and review steps; link related
  issues; include screenshots for diagrams and list testing steps.

## Security & configuration tips

- Do not commit secrets or tokens. Use environment variables and a gitignored
  `.env.local` when needed.
- If adding code, pin tool versions and document prerequisites in the folder
  `README.md`.

