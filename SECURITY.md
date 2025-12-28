# Security Policy

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in ArgusNexus, please report it responsibly.

### How to Report

1. **DO NOT** open a public GitHub issue for security vulnerabilities
2. Email security concerns to the repository maintainers
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Any suggested fixes (optional)

### What to Expect

- Acknowledgment within 48 hours
- Regular updates on the fix progress
- Credit in the security advisory (if desired)

### Scope

This security policy applies to:
- The ArgusNexus codebase
- Official documentation
- Example configurations

### Out of Scope

- Third-party dependencies (report to their maintainers)
- Your own deployment configurations
- Trading losses (this is not financial advice)

## Security Best Practices

When deploying ArgusNexus:

1. **Never commit secrets** - Use `.env` files (gitignored) or environment variables
2. **Rotate API keys regularly** - Especially after any suspected exposure
3. **Use HTTPS in production** - Set `PRODUCTION=true` for secure cookies
4. **Limit API access** - Configure CORS origins appropriately
5. **Monitor your deployment** - Use the built-in watchdog and alerts

## Disclaimer

ArgusNexus is provided for educational purposes. Use at your own risk. The maintainers are not responsible for any financial losses incurred through use of this software.
