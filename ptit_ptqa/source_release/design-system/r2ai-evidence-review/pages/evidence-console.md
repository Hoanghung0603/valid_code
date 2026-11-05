# Evidence Console — page override

This override replaces the automatically generated neumorphic/orange direction. The product is a technical medical evidence console, not a consumer wellness app.

## Visual direction

- Scientific editorial dashboard; restrained, audit-oriented and high trust.
- Background `#EEF3F4`, surface `#FFFFFF`, application shell `#071B2A`.
- Teal `#0A9182` is the only strong accent. Amber appears only for disclosure or warning.
- Segoe UI Variable/Aptos locally, with Figtree/Noto Sans as optional fallbacks.
- Radius scale 10 / 16 / 24 px. Use borders before shadows; one strong shell shadow only.
- Compact result lists, spacious narrative headings.

## Interaction

- Query selection, tabs, pagination and evidence cards remain keyboard operable.
- Every focused control has a visible 2 px teal outline with 3 px offset.
- Result cards support Enter/Space and expose an accessible action name.
- Async replay uses a busy state plus an aria-live toast.
- Motion stays within 180–220 ms and stops under `prefers-reduced-motion`.

## Responsive priorities

- 1440+: three-column review console.
- 1024–1249: query plus results first; evidence moves below.
- 375–800: one column; query, results and evidence keep the same order.
- No horizontal scroll. Hashes and URLs use `overflow-wrap:anywhere`.

## Avoid

Neumorphism, orange operation palettes, decorative medical symbols, gradients on data cards, emoji icons, hidden focus rings, color-only states and layout-shifting hover transforms.
