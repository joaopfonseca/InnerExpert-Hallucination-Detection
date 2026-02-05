PROJECT_NAME = moeuncert

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "lightning_logs" -exec rm -rf {} +
	rm -rf *.egg-info
	rm -rf dist
	rm -rf build
	rm -rf coverage.xml
	rm -rf .coverage
	rm -rf .coverage.*
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf docs/_build
	rm -rf docs/_generated

## Lint using black, flake8 and pylint
code-analysis:
	black --check --diff .
	mypy $(PROJECT_NAME) --ignore-missing-imports --no-strict-optional
	flake8 $(PROJECT_NAME) --ignore=E203,W503
	pylint -E $(PROJECT_NAME) -d E1103,E0611,E1101,E0601

## Format code using Black
code-format:
	black $(PROJECT_NAME)
