---
name: pr
description: Create a pull request for the current branch
allowed-tools: Bash(git:*), Bash(gh:*)
---

# Create a PR

Create a pull request for the current branch against main.

## Workflow

Follow these steps:

### 1. Understand the Conventional Commits syntax

The commit contains the following structural elements, to communicate intent to the
consumers of your library:

1. **fix:** a commit of the _type_ `fix` patches a bug in your codebase
   (this correlates with `PATCH` in Semantic Versioning).
2. **feat:** a commit of the _type_ `feat` introduces a new feature to the codebase
   (this correlates with `MINOR`in Semantic Versioning).
3. **BREAKING CHANGE:** a commit that has a footer `BREAKING CHANGE:`, or appends a `!` after the type/scope, introduces a breaking API change   
   (correlating with `MAJOR` in Semantic Versioning).
   A BREAKING CHANGE can be part of commits of any _type_.
4. _types_ other than `fix:` and `feat:` are allowed, e.g. `build:`, `chore:`,
  `ci:`, `docs:`, `style:`, `refactor:`, `perf:`, `test:`, and others.
5. _footers_ other than `BREAKING CHANGE: <description>` may be provided and follow a convention similar to the git trailer format.

Additional types are not mandated by the Conventional Commits specification, and have no implicit effect in Semantic Versioning (unless they include a `BREAKING CHANGE`).
A scope may be provided to a commit's type, to provide additional contextual information and is contained within parenthesis, e.g., `feat(parser): add ability to parse arrays`.

### 2. Understand the PR contents

Run in parallel:
- `git status` to see untracked files (never use -uall flag)
- `git diff` to see staged and unstaged changes
- Check if the current branch tracks a remote branch and is up to date
- `git log --oneline main..HEAD` and `git diff main...HEAD` to understand all commits on this branch

If there are uncommitted changes, warn the user before proceeding.

### 3. Draft a PR title and body

Analyze ALL commits on the branch (not just the latest) and draft 
- A short PR title (under 70 characters) using Conventional Commits syntax:
  `<type>[optional scope]: <description>`
- A PR body summarizing the changes made in the PR. Use bullet points. 

Afterwards, ask for feedback regarding the title and body. 

### 4. Create the PR 

Push the branch if needed, then create the PR using:

```
gh pr create --title "<type>[optional scope]: <description>" --body "$(cat <<'EOF'
<bullet points>

EOF
)"
```

### 5. Return the PR URL when done
