# assets/

Static assets for documentation and the project README.

## Purpose

Images, diagrams, and other binary assets referenced from `README.md` or
`docs/`. Source diagrams that can be expressed as Mermaid (e.g., the
architecture diagram in the root README) should stay as Mermaid code
blocks rather than rendered images, so they remain diffable and
editable — reserve this directory for assets that cannot be
(logo, screenshots, plots).

## Conventions

- Use descriptive, kebab-case filenames (e.g., `architecture-overview.png`).
- Keep file sizes reasonable; large binaries should be avoided or tracked
  via Git LFS if the repository adopts it in the future.

*No assets exist yet — this is a structural placeholder.*
