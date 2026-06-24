.PHONY: install lint test serve demo docker run

install:
	pip install -r requirements-dev.txt

lint:
	ruff check src app tests

test:
	PYTHONPATH=src pytest

serve:        ## run the API locally on :8080
	PYTHONPATH=src MODEL_DIR=models uvicorn app.api.main:app --reload --port 8080

demo:         ## run the Gradio demo locally on :7860
	MODEL_DIR=models python app/demo/app.py

docker:
	docker build -t echonext-shd:local .

run: docker
	docker run --rm -p 8080:8080 echonext-shd:local
