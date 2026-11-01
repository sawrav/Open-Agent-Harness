---
name: development_workflow
description: Guidelines for development workflows using the harness tools
---

## Development Workflow Skill

When working on development tasks:

- Use code_analysis skill for understanding codebases
- Use project_management skill for repository understanding
- Use file_analysis skill for file and directory analysis
- Use debugging skill for troubleshooting
- Use file_operations skill for file management

Development workflow:
1. **Explore**: Use list_directory, dir_stats, search_files
2. **Understand**: Read relevant files, use batch_read for multiple files
3. **Analyze**: Use grep_files to find patterns, compare_files for versions
4. **Modify**: Write files with proper encoding, use write_file
5. **Verify**: Test changes, compare before/after with compare_files
6. **Document**: Update documentation, use summarize skill

Tool combinations:
- search_files + grep_files = find code patterns
- file_info + watch_file = understand file history
- batch_read + read_file = examine multiple files
- dir_stats + list_directory = understand project structure

Always follow git best practices from git_operations skill.
