# Research Presentation: Hallucination Detection in Mixture-of-Experts Models

This directory contains a reveal.js HTML presentation about the MoE uncertainty estimation research project.

## Files

- `index.html` - The main presentation file (reveal.js HTML)
- `content.md` - Markdown outline of the presentation content
- `README.md` - This file

## Viewing the Presentation

Open Directly in Browser. Simply open `index.html` in any modern web browser:

```bash
# On Linux
xdg-open index.html

# On macOS
open index.html

# On Windows
start index.html

# Or just double-click the file in your file explorer
```

The presentation uses CDN-hosted reveal.js libraries, so it works without any local installation.

## Technical Details

- **Framework**: [reveal.js](https://revealjs.com/) v4.5.0
- **CDN**: CloudFlare CDN (no local dependencies)
- **Code Highlighting**: Monokai theme
- **Responsive**: Works on mobile and desktop
- **Offline**: Works offline once loaded (CDN assets cached)

## Content Updates

To update the content, edit `index.html` directly. The structure is:

```html
<section>  <!-- New slide -->
    <h2>Slide Title</h2>
    <p>Content...</p>
</section>

<section>  <!-- Vertical slides (use down arrow) -->
    <section><h2>Parent Slide</h2></section>
    <section><h2>Child Slide 1</h2></section>
    <section><h2>Child Slide 2</h2></section>
</section>
```
