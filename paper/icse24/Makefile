NAME := main
SUP := sup
TEXS := $(wildcard *.tex)
TABLES := $(wildcard tables/*.tex)
FIGS := $(wildcard figs/*.tex) $(wildcard code/*)
PLOTS := $(wildcard figs/*.eps)
BIBS := $(wildcard *.bib)

LATEX = pdflatex
BIBTEX = bibtex

all: ${NAME}.pdf

sup: ${SUP}.pdf

${NAME}.pdf: ${TEXS} ${TABLES} ${BIBS} ${FIGS} ${PLOTS} bib.bib
	-rm -f ${NAME}.aux
	-rm -f ${NAME}.bbl
	-$(LATEX) $(NAME)
	-$(BIBTEX) $(NAME)
	-$(LATEX) $(NAME)
	-$(LATEX) $(NAME)
	@echo '****************************************************************'
# vvvvvv For `latex` ONLY. Comment out for `pdflatex`
#	@dvips -j0 -t letter -o $(NAME).ps $(NAME).dvi
#	@ps2pdf -dPDFSETTINGS=/prepress $(NAME).ps $(NAME).pdf
# ^^^^^^^^^^^^^^^^^^^^^^^^
	@echo '******** Did you spell-check the paper? ********'


${SUP}.pdf: ${TEXS} ${TABLES} ${BIBS} ${FIGS} ${PLOTS} bib.bib
	-rm -f ${SUP}.aux
	-rm -f ${SUP}.bbl
	-$(LATEX) $(SUP)
	-$(BIBTEX) $(SUP)
	-$(LATEX) $(SUP)
	-$(LATEX) $(SUP)
	@echo '****************************************************************'
# vvvvvv For `latex` ONLY. Comment out for `pdflatex`
#	@dvips -j0 -t letter -o $(SUP).ps $(SUP).dvi
#	@ps2pdf -dPDFSETTINGS=/prepress $(SUP).ps $(SUP).pdf
# ^^^^^^^^^^^^^^^^^^^^^^^^
	@echo '******** Did you spell-check the paper? ********'

clean:
	ls $(NAME)* | grep -v ".tex" | xargs rm -f
	ls $(SUP)* | grep -v ".tex" | xargs rm -f
	rm -f *.bak *~
#	find -name "*.eps" | xargs rm -f
