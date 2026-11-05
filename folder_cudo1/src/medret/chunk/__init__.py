"""Stage 3 - split parsed documents into retrieval units with character offsets.

Retrieval granularity and submission granularity are decoupled: units are retrieved small and can be
merged into larger output spans at submission time. See docs/design/architecture.md section 2.3.
"""
