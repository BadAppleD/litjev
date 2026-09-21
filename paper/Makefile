.PHONY: all clean
all: main.pdf

main.pdf: main.tex fig_arch.tex references.bib curve_head.dat curve_stats.dat
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

clean:
	latexmk -C
	rm -f *.png *.bbl *.blg build.log
