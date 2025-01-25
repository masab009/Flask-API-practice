install:
		pip install -r requirements.txt &&\
		pip install --upgrade pip

test:
	python -m pytest -vv tests/test_app.py
format:
	black *.py

lint:
	pylint --disable=R,C webapp/app.py

all: install lint test format