# Security Policy

## Supported Versions

NeuroPager is currently in pre-alpha, structural-scaffold status. There are
no tagged releases yet. Once versioned releases begin, this table will track
which versions receive security fixes.

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability within NeuroPager, please report it
privately rather than opening a public issue.

- **Email:** skkavi4618@gmail.com
- **Subject line:** `[SECURITY] NeuroPager - <short description>`

Please include:

- A description of the vulnerability and its potential impact
- Steps to reproduce, or a proof-of-concept if available
- Any suggested remediation, if known

You should expect an initial response within **5 business days**. We will
work with you to understand and validate the issue, and will credit
reporters (unless anonymity is requested) once a fix is released.

## Scope Notes

NeuroPager is a research library intended to be embedded into larger agent
systems. As the project matures to include:

- Persistent disk-backed page stores
- Vector store and knowledge graph integrations
- Network-facing retrieval components

...the attack surface will grow, and this policy will be expanded
accordingly (e.g., dependency scanning, supply-chain provenance, sandboxing
guidance for untrusted memory content).
