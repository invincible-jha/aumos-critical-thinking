# Security Policy

## Reporting a Vulnerability

Report security vulnerabilities to **security@aumos.ai** with:
- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested remediation (if known)

Do NOT file public GitHub issues for security vulnerabilities.

## Response Timeline

- Acknowledgment within 48 hours
- Initial assessment within 5 business days
- Patch release target: within 30 days for critical issues

## Security Controls

- All database queries use parameterised SQLAlchemy expressions — no raw SQL string concatenation
- Tenant isolation enforced via PostgreSQL RLS (`SET app.current_tenant`) on every DB session
- Non-root Docker user (`aumos`) — no privilege escalation in containers
- Secrets managed via environment variables — never hardcoded in source
- Rate limiting applied at the auth-gateway layer upstream of this service
- JWT validation delegated to `aumos-auth-gateway` — this service trusts `X-Tenant-ID` header
