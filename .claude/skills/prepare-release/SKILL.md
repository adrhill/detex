---
name: prepare-release
description: Prepare a new release by updating the changelog and version
disable-model-invocation: true
---

Prepare a new release for asdex. Follow the instructions in CLAUDE.md under "Releases".

## Context

- Current version: !`grep '^version' pyproject.toml`
- Latest git tag: !`git describe --tags --abbrev=0`
- Commits since last tag: (run `git log <latest-tag>..HEAD --oneline` using the tag above)

## Instructions

1. Determine the version bump using [semver](https://semver.org/) (MAJOR.MINOR.PATCH).
   The highest bump among all commits wins.
   While the version has a leading 0, the public API is not stable,
   so shift bump levels down: breaking → MINOR, everything else → PATCH.
   After 1.0.0: breaking → MAJOR, `feat:` → MINOR, `fix:` → PATCH.

2. For commits that also fix bugs or add features beyond what the commit type suggests
   (check PR descriptions with `gh pr view`), include those as separate changelog entries.

3. Map commit types to changelog badge types:
   - `badge-breaking` ← any `!` (e.g. `feat!:`, `fix!:`)
   - `badge-feature` ← `feat:`
   - `badge-bugfix` ← `fix:`
   - `badge-maintenance` ← `refactor:`, `test:`, `chore:`
   - `badge-docs` ← `docs:`

4. Update `CHANGELOG.md`:
   - Add a new `## Version vX.Y.Z` section above the previous release
   - Order entries by badge type: breaking, feature, enhancement, bugfix, maintenance, docs
   - Each entry links to its PR using the existing badge format
   - Add PR link references in numerical order

5. Update `version` in `pyproject.toml`

6. Show the user a summary of changes and commit as `` asdex `vX.Y.Z` ``.

7. Push the commit, then tag with `git tag vX.Y.Z` and push the tag with `git push --tags`.
   `vX.Y.Z` is the new version number from `pyproject.toml`.
