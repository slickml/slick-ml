  
PROJECT = "slick-ml" 
SHELL := /bin/bash

format:
	@${EXEC} black .

format-check:
	@${EXEC} black . --check

lint:
	@${EXEC} flake8 .

checks:
	make format-check
	make lint