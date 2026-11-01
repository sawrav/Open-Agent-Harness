---
name: file_operations
description: Best practices for file operations using the harness tools
---

## File Operations Skill

When working with files, follow these guidelines:

- Always check if a file exists before reading it using the file_info tool
- Use search_files to find files by pattern instead of guessing paths
- Use grep_files to search for content within files
- For large files, read them in chunks if possible or summarize them
- When writing files, prefer base64 encoding for source code to avoid JSON escaping issues
- Always create parent directories before writing files using write_file
- Use list_directory to explore unfamiliar directories
- Be careful with destructive operations - check file contents first
- Use relative paths when possible, but absolute paths are fine too

Example workflow:
1. search_files for relevant files
2. file_info to check file details
3. read_file to examine contents
4. write_file with base64 encoding for code
