#!/bin/bash

ARGUMENT_LIST=(
    "platform"
    "component-path"
    "config"
    "port"
    "certificate"
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
        --platform)
            argPlatform=$2
            shift 2
            ;;

	--component-path)
            argComponentPath=$2
            shift 2
            ;;

	--config)
            argConfig=$2
            shift 2
            ;;

	--port)
            argPort=$2
            shift 2
            ;;

	--certificate)
            argCertificate=$2
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

echo "======================================="
echo "PLATFORM = $argPlatform"
echo "COMPONENT_PATH = $argComponentPath"
echo "CONFIG = $argConfig"
echo "PORT = $argPort"
echo "CERTIFICATE = $argCertificate"
echo "======================================="

COMPOSE_API_VERSION=1.40 \
PLATFORM=$argPlatform \
COMPONENT_PATH=$argComponentPath \
CONFIG=$argConfig \
PORT=${argPort} \
CERTIFICATE=${argCertificate} \
bash -c \
'docker-compose --env-file project.env -f combiner.yaml build && \
docker-compose --env-file project.env -f combiner.yaml up'
#--remove-orphans
