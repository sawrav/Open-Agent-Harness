---
name: debugging
description: Guidelines for debugging code and troubleshooting issues
---

## Debugging Skill

When debugging code or troubleshooting issues:

- Start by reading the relevant files to understand the current state
- Use grep_files to find error messages, warnings, or specific code patterns
- Check file_info for timestamps and sizes to understand recent changes
- Look at git log to see recent commits that might have introduced issues
- Use run_shell to execute diagnostic commands
- Search for test files and run them if available
- Check configuration files for misconfigurations
- Create minimal reproduction cases when possible
- Document findings and solutions clearly

Debugging workflow:
1. Reproduce the issue consistently
2. Identify the relevant code files
3. Search for related error patterns
4. Examine recent changes via git
5. Test hypotheses with small experiments
6. Verify the fix works
