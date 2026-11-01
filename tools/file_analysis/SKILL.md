---
name: file_analysis
description: Guidelines for analyzing file relationships and directory structures
---

## File Analysis Skill

When analyzing files and directories:

- Use dir_stats to get an overview of directory contents and sizes
- Use compare_files to check differences between files
- Use path_resolve to understand file locations
- Use watch_file to monitor file changes over time
- Combine multiple tools for comprehensive analysis

Analysis patterns:
1. **Directory exploration**: Start with dir_stats for overview
2. **File comparison**: Use compare_files for duplicates or versions
3. **Change tracking**: Use watch_file to see modification patterns
4. **Structure understanding**: Combine list_directory with dir_stats

Best practices:
- Always check file existence before operations
- Use recursive analysis for comprehensive understanding
- Compare files before making changes
- Track file modification times to understand development patterns
- Analyze directory size distributions to identify large files
