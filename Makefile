.PHONY: pretty precommit_install

# if BIN not provided, try to detect the binary from the environment
PYTHON_INSTALL := $(shell python3 -c 'import sys;print(sys.executable)')
BIN ?= $(shell [ -e .venv/bin ] && echo `pwd`/'.venv/bin' || dirname $(PYTHON_INSTALL))/

CODE = app

help:  ## This help dialog.
	@IFS=$$'\n' ; \
	help_lines=(`fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##/:/'`); \
	printf "%-15s %s\n" "target" "help" ; \
	printf "%-15s %s\n" "------" "----" ; \
	for help_line in $${help_lines[@]}; do \
		IFS=$$':' ; \
		help_split=($$help_line) ; \
		help_command=`echo $${help_split[0]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		help_info=`echo $${help_split[2]} | sed -e 's/^ *//' -e 's/ *$$//'` ; \
		printf '\033[36m'; \
		printf "%-15s %s" $$help_command ; \
		printf '\033[0m'; \
		printf "%s\n" $$help_info; \
	done

lint:
	$(BIN)flake8
	$(BIN)isort -c . && isort .

precommit_install:
	echo '#!/bin/sh' >  .git/hooks/pre-commit
	echo "exec make lint BIN=$(BIN)" >> .git/hooks/pre-commit
	chmod +x .git/hooks/pre-commit
