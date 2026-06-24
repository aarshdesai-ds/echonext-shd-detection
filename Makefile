.PHONY: install lint test serve serve-all demo docker run

install:
	pip install -r requirements-dev.txt

lint:
	ruff check src app tests

test:
	PYTHONPATH=src pytest

serve:        ## run the pure API locally on :8080
	PYTHONPATH=src MODEL_DIR=models uvicorn app.api.main:app --reload --port 8080

serve-all:    ## run the combined API + demo (what deploys to Cloud Run) on :8080
	PYTHONPATH=src:. MODEL_DIR=models uvicorn app.serve:app --port 8080

demo:         ## run the Gradio demo standalone on :7860
	MODEL_DIR=models python app/demo/app.py

docker:
	docker build -t echonext-shd:local .

run: docker
	docker run --rm -p 8080:8080 echonext-shd:local
