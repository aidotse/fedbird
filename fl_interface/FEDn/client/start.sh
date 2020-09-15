#!/bin/bash

export ALLIANCE_UID=ac435faef-c2df-442e-b349-7f633d3d5523
export CLIENT_NAME_BASE=client-fedn1-
export MDBUSR=qamcom
export MDBPWD=qamcom


COMPOSE_API_VERSION=1.40 HOSTS_SOURCE=$1 CLIENT_NAME=$2 DOCKERFILE=$3 docker-compose -f start.yaml build
COMPOSE_API_VERSION=1.40 HOSTS_SOURCE=$1 CLIENT_NAME=$2 DOCKERFILE=$3 docker-compose -f start.yaml up 


# export ALLIANCE_UID=ac435faef-c2df-442e-b349-7f633d3d5523
#CLIENT_NAME_BASE=client-fedn1- MDBUSR=qamcom MDBPWD=qamcom COMPOSE_API_VERSION=1.40 HOSTS_SOURCE=dns CLIENT_NAME=client-fedn1-1 DOCKERFILE=./nvidia_platform.dockerfile docker-compose -f start.yaml up 
