#!/bin/bash

COMPOSE_API_VERSION=1.40 \
docker-compose --env-file project.env -f data-storage.yaml build --pull

COMPOSE_API_VERSION=1.40 \
docker-compose --env-file project.env -f data-storage.yaml up
