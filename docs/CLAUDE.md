# Documentation Guide for AI Agents

## Diataxis Framework

The docs follow the [Diataxis](https://diataxis.fr/) framework.
Each page belongs to one of four categories:

- **Tutorials** (`getting-started.md`): Learning-oriented.
  Walk the reader through steps to complete a task.
  Don't explain concepts — just guide them.
- **How-To Guides** (`how-to/`): Task-oriented.
  Show how to solve specific problems.
  Assume the reader knows the basics.
- **Explanation** (`explanation/`): Understanding-oriented.
  Explain concepts, design decisions, and theory.
  No step-by-step instructions.
- **Reference** (`reference/`): Information-oriented.
  Auto-generated from docstrings via mkdocstrings.
  Keep docstrings accurate and complete.

## Semantic Line Breaks

All markdown content uses **semantic line breaks**:
one sentence or clause per line.
This makes diffs cleaner and is a firm requirement.

Good:
```markdown
Graph coloring assigns colors to vertices such that adjacent vertices get different colors.
This allows computing multiple rows in a single AD pass.
```

Bad:
```markdown
Graph coloring assigns colors to vertices such that adjacent
vertices get different colors. This allows computing multiple
rows in a single AD pass.
```

## MkDocs Conventions

### Autodoc Directives

Reference pages use mkdocstrings autodoc syntax:

```markdown
::: asdex.jacobian
```

This pulls the docstring from the source code.
Keep docstrings in Google style.

### Admonitions

Use admonitions for callouts:

```markdown
!!! tip "Title"

    Content here.

!!! warning

    Content here.
```

### Math

Use MathJax for LaTeX:

- Inline: `\(f: \mathbb{R}^n \to \mathbb{R}^m\)`
- Display: `\[J \in \mathbb{R}^{m \times n}\]`

## Local Preview

When making major changes to the docs
(adding pages, restructuring nav, changing MkDocs config),
serve the site locally and verify the result before finishing:

```bash
uv run mkdocs serve
```

This starts a live-reloading server at `http://127.0.0.1:8000`.
Use `uv run mkdocs build --strict` to catch broken links and warnings.

## Navigation Structure

The nav in `mkdocs.yml` maps to Diataxis categories:

- **Home** tab → tutorials (index, getting-started)
- **How-To Guides** tab → task-oriented guides
- **Explanation** tab → concept explanations
- **Reference** tab → auto-generated API docs
- **Contributing** → development guide
- **Benchmarks** → external link to benchmark dashboard
