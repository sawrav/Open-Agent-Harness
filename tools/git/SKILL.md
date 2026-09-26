---
name: git_operations
description: Best practices for performing git operations via the shell tool
---

## Git Skill

When the user asks about git operations, follow these guidelines:

- Always run `git status` before any write operation to understand the current state.
- Use `git log --oneline -10` to show recent commit history concisely.
- Never run `git push --force` or `git reset --hard` unless the user explicitly confirms.
- Prefer `git diff --stat` before showing a full diff to give a summary first.
- When creating commits, write clear and concise commit messages in the imperative mood
  (e.g., "Add feature X", not "Added feature X").
- Always check the current branch with `git branch --show-current` before making changes.
- Prefer staging specific files with `git add <file>` over `git add .`.
