install: 
	@echo "Installing..."
	pipenv install --dev
	pipenv run pre-commit install

activate:
	@echo "Activating virtual environment..."
	pipenv shell
