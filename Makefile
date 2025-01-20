build-docker:
	docker ps -a
	docker stop $(docker ps -a -q)
	docker rm $(docker ps -a -q)
	docker compose -f docker-compose.yml up -d

run-train-docker:
	docker exec cicd_docker_ml_demo-core-1 python train.py

stop-docker:
	docker compose -f docker-compose.yml down

install:
	pip install --upgrade pip && pip install -r requirements.txt

format:
	black *.py

train:
	python train.py

eval:
	echo "## Model Metrics" > report.md
	cat ./Results/metrics.txt >> report.md
	echo '\n## Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](./Results/model_results.png)' >> report.md
	cml comment create report.md

hf-login:
	git pull origin main
	git switch main
	git config --global credential.helper store
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HF) --add-to-git-credential

push-hub:
	huggingface-cli upload SiraH/Drug-classification-docker-fastapi Dockerfile --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload SiraH/Drug-classification-docker-fastapi requirements.txt --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload SiraH/Drug-classification-docker-fastapi main.py --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload SiraH/Drug-classification-docker-fastapi ./routes /routes --repo-type=space --commit-message="Sync App files"
	huggingface-cli upload SiraH/Drug-classification-docker-fastapi ./Model /Model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload SiraH/Drug-classification-docker-fastapi ./Results /Metrics --repo-type=space --commit-message="Sync Model"

deploy: hf-login push-hub