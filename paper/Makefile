# ============================================
# MAKEFILE FOR LATEX PAPER COMPILATION
# ============================================

# Main document name (without extension)
MAIN = main

# LaTeX compiler
LATEX = pdflatex
BIBTEX = bibtex

# Flags
LATEX_FLAGS = -interaction=nonstopmode -halt-on-error

# Build directory for auxiliary files (optional, comment out if not needed)
# BUILD_DIR = build

# ============================================
# TARGETS
# ============================================

.PHONY: all pdf bib clean cleanall view watch help

# Default target: build the PDF
all: pdf

# Build PDF with bibliography
pdf: $(MAIN).tex references.bib
	@echo "=== First LaTeX pass ==="
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo "=== Running BibTeX ==="
	$(BIBTEX) $(MAIN)
	@echo "=== Second LaTeX pass ==="
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo "=== Third LaTeX pass ==="
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	@echo "=== Build complete: $(MAIN).pdf ==="

# Quick build (single pass, no bibliography update)
quick: $(MAIN).tex
	@echo "=== Quick LaTeX build ==="
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex

# Only update bibliography
bib: $(MAIN).tex references.bib
	$(LATEX) $(LATEX_FLAGS) $(MAIN).tex
	$(BIBTEX) $(MAIN)

# Clean auxiliary files
clean:
	@echo "=== Cleaning auxiliary files ==="
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.fls *.fdb_latexmk *.synctex.gz

# Clean everything including PDF
cleanall: clean
	@echo "=== Cleaning PDF ==="
	rm -f $(MAIN).pdf

# View PDF (macOS)
view: pdf
	open $(MAIN).pdf

# Watch for changes and rebuild (requires fswatch on macOS)
# Install with: brew install fswatch
watch:
	@echo "=== Watching for changes... Press Ctrl+C to stop ==="
	fswatch -o *.tex *.bib | xargs -n1 -I{} make quick

# Help
help:
	@echo "Available targets:"
	@echo "  make          - Build PDF with full compilation (default)"
	@echo "  make pdf      - Build PDF with full compilation"
	@echo "  make quick    - Quick build (single pass, no bibliography)"
	@echo "  make bib      - Update bibliography only"
	@echo "  make clean    - Remove auxiliary files"
	@echo "  make cleanall - Remove all generated files including PDF"
	@echo "  make view     - Build and open PDF (macOS)"
	@echo "  make watch    - Watch for changes and auto-rebuild"
	@echo "  make help     - Show this help message"
