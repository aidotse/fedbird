#!/bin/bash

ARGUMENT_LIST=(
    "component-path"
)

opts=$(getopt \
    --longoptions "$(printf "%s:," "${ARGUMENT_LIST[@]}")" \
    --name "$(basename "$0")" \
    --options "" \
    -- "$@"
)

eval set --$opts

while [[ $# -gt 0 ]]; do
    case "$1" in
	--component-path)
            argComponentPath=$2
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

echo "======================================="
echo "COMPONENT_PATH = $argComponentPath"
echo "======================================="

COMPOSE_API_VERSION=1.40 \
COMPONENT_PATH=$argComponentPath \
docker-compose --env-file project.env -f dashboard.yaml build

COMPOSE_API_VERSION=1.40 \
COMPONENT_PATH=$argComponentPath \
docker-compose --env-file project.env -f dashboard.yaml up #--remove-orphans
