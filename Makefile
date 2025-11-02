rd:
	DOCKER_BUILDKIT=1 docker compose up -d --build axora-rag 

run:
	DOCKER_BUILDKIT=1 docker compose up -d

all:
	DOCKER_BUILDKIT=1 docker compose up -d --build 

ins:
	go mod tidy && go mod vendor

model:
	docker exec -it axora-ollama ollama pull mistral:7b-instruct-q4_0

stop:
	docker compose down