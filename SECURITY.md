# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please email the repository owner directly instead of opening a public issue.

## Secure Development Practices

This repository follows these security practices:

1. **No Credentials in Code**: Never commit API keys, passwords, or credentials
2. **Environment Variables**: Use `.env` files for configuration (added to `.gitignore`)
3. **Pre-commit Hooks**: Automated checks prevent committing sensitive files
4. **SSH Authentication**: Use SSH keys instead of passwords for Git operations

## Protected Files

The following files should NEVER be committed:
- `credentials.json` - Google Cloud credentials
- `token.json` - OAuth tokens
- `.env` - Environment variables
- Any files with API keys or passwords

## Git Security Features Enabled

- ✅ `.gitignore` configured to exclude sensitive files
- ✅ Pre-commit hook to scan for secrets
- ✅ SSH authentication configured
- ✅ No username/password exposure in Git operations
