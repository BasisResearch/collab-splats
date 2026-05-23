lint: FORCE
	bash ./scripts/lint.sh

lint-notebooks:
	bash ./scripts/lint_notebooks.sh

format:
	bash ./scripts/clean.sh

format-notebooks:
	bash ./scripts/clean_notebooks.sh

test: lint FORCE
	bash ./scripts/test.sh

test-notebooks: lint FORCE
	bash ./scripts/test_notebooks.sh

docs: FORCE
	LC_ALL=C.UTF-8 LANG=C.UTF-8 sphinx-build -b html docs/source docs/_build/html --keep-going

docs-serve: FORCE
	python -m http.server 8080 --directory docs/_build/html

FORCE: