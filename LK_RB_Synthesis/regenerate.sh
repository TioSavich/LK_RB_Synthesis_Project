#!/bin/bash
# Quick regeneration script for EPLE analysis and diagrams

echo "🚀 EPLE Quick Regeneration Script"
echo "=================================="
echo ""

# Navigate to project directory
cd "$(dirname "$0")"

# Run analysis
echo "📊 Step 1: Running automated analysis..."
python3 main.py analyze
echo ""

# Generate markdown report
echo "📝 Step 2: Generating markdown report..."
python3 main.py report --format markdown > output/analysis_report.md
echo "   ✓ Saved to output/analysis_report.md"
echo ""

# Compile PDF if LaTeX is available
if command -v pdflatex &> /dev/null; then
    echo "📄 Step 3: Compiling PDF diagrams..."
    cd output
    pdflatex -interaction=nonstopmode all_muds.tex > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "   ✓ PDF compiled: output/all_muds.pdf"
    else
        echo "   ✗ PDF compilation failed (check all_muds.log)"
    fi
    cd ..
else
    echo "📄 Step 3: Skipping PDF compilation (pdflatex not found)"
fi

echo ""
echo "✅ Complete! View results:"
echo "   • PDF diagrams:    open output/all_muds.pdf"
echo "   • Text analysis:   cat output/analysis_report.md"
echo "   • Raw data:        cat output/mud_diagrams.json"
echo ""
echo "Or explore interactively:"
echo "   python3 main.py explore"
echo ""
