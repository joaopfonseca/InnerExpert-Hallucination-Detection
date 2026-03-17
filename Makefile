PROJECT_NAME = moeuncert

## Generate answers using an LLM on the RealtimeQA dataset.
## Override with: make realtimeqa-answers ARGS="--model <name> --years <y> --month <m>"
realtimeqa-answers:
	python experiments/3.0-generate-answers.py $(ARGS)

## Analyze generation metrics and produce plots.
## Override with: make analyze-metrics ANALYZE_ARGS="--model <name> --years <y> --month <m>"
realtimeqa-metrics:
	python experiments/3.1-analyze-metrics.py $(ANALYZE_ARGS)

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

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
