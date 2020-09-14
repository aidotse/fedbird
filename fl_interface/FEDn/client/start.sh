#!/bin/bash

export ALLIANCE_UID=ac435faef-c2df-442e-b349-7f633d3d5523
export CLIENT_NAME_BASE=is-it-important
export MDBUSR=qamcom
export MDBPWD=qamcom
COMPOSE_API_VERSION=1.40 HOSTS_SOURCE=$1 CLIENT_NAME=$2 DOCKERFILE=$3 docker-compose -f start.yaml up 
