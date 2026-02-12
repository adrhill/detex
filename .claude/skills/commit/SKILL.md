---
name: commit
description: Create a commit using Conventional Commits syntax
allowed-tools: Bash(git:*)
---

# Create a commit

Create a commit for the current changes using Conventional Commits syntax.

## Workflow

Follow these steps:

### 1. Understand the Conventional Commits syntax

The commit contains the following structural elements, to communicate intent to the
consumers of your library:

1. **fix:** a commit of the _type_ `fix` patches a bug in your codebase
   (this correlates with `PATCH` in Semantic Versioning).
2. **feat:** a commit of the _type_ `feat` introduces a new feature to the codebase
   (this correlates with `MINOR`in Semantic Versioning).
3. **BREAKING CHANGE:** a commit that has a footer BREAKING CHANGE:, or appends a ! after the type/scope, introduces a breaking API change
   (correlating with MAJOR in Semantic Versioning).
   A BREAKING CHANGE can be part of commits of any _type_.
4. _types_ other than `fix:` and `feat:` are allowed, e.g. `build:`, `chore:`,
  `ci:`, `docs:`, `style:`, `refactor:`, `perf:`, `test:`, and others.
5. _footers_ other than `BREAKING CHANGE: <description>` may be provided and follow a convention similar to the git trailer format.

Additional types are not mandated by the Conventional Commits specification, and have no implicit effect in Semantic Versioning (unless they include a `BREAKING CHANGE`).
A scope may be provided to a commit's type, to provide additional contextual information and is contained within parenthesis, e.g., `feat(parser): add ability to parse arrays`.

### 2. Understand the changes

Run in parallel:
- `git status` to see untracked files (never use -uall flag)
- `git diff` to see both staged and unstaged changes
- `git log --oneline -5` to see recent commit messages for style reference

### 3. Draft a commit message

Analyze all staged and unstaged changes and draft a commit message:
- Use Conventional Commits syntax: `<type>[optional scope]: <description>`
- Keep the first line under 70 characters
- Focus on the "why" rather than the "what"
- Include a body when the change is non-trivial.
  The body should explain context, motivation, or list key changes.

Afterwards, ask for feedback regarding the commit message.

### 4. Create the commit

Stage the relevant files by name (avoid `git add -A` or `git add .`) and create the commit:

```
git add <files> && git commit -m "$(cat <<'EOF'
<type>[optional scope]: <description>

<optional body>

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

Do not push to the remote unless explicitly asked.

### 5. Confirm success

Run `git status` after the commit to verify it succeeded.
