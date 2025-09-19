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

FORCE: