---
name: code_analysis
description: Guidelines for analyzing and understanding codebases
---

## Code Analysis Skill

When analyzing code or understanding a codebase:

- Start with search_files to find relevant files by pattern
- Use grep_files to find specific code patterns, functions, or classes
- Use file_info to understand file sizes and modification times
- Use batch_read to examine multiple related files at once
- Use path_resolve to understand file locations and symlinks
- Look for entry points, main functions, and configuration files first
- Identify the project structure: tests, docs, config, source
- Check for README files and documentation
- Examine dependencies and requirements files
- Understand the module structure and imports

Analysis workflow:
1. Map the directory structure with list_directory
2. Find key files with search_files
3. Identify entry points and main modules
4. Examine imports and dependencies
5. Look for tests to understand expected behavior
6. Check documentation and comments
7. Identify patterns and conventions used in the codebase
