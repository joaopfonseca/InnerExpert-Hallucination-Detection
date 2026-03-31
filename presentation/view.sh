#!/bin/bash
# Quick launcher for the presentation

cd "$(dirname "$0")"

echo "🎬 Opening presentation..."
echo ""
echo "Navigation:"
echo "  → / ← : Next/Previous slide"
echo "  Space : Next slide"
echo "  ESC   : Overview mode"
echo "  F     : Fullscreen"
echo ""

# Try to open the presentation in default browser
if command -v xdg-open &> /dev/null; then
    xdg-open index.html
elif command -v open &> /dev/null; then
    open index.html
elif command -v start &> /dev/null; then
    start index.html
else
    echo "Could not detect system browser opener."
    echo "Please open index.html manually in your browser."
    echo ""
    echo "Or start a local server:"
    echo "  python3 -m http.server 8000"
    echo "Then open http://localhost:8000 in your browser"
fi
