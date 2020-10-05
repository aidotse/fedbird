#!/bin/bash

ARGUMENT_LIST=(
    "client-name"
    "platform"
    "component-path"
    "config"
    "certificate"
    "data-path"
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
        --client-name)
            argClientName=$2
            shift 2
            ;;

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

	--certificate)
            argCertificate=$2
            shift 2
            ;;

	 --data-path)
            argDataPath=$2
	    shift 2
	    ;;
        *)
            break
            ;;
    esac
done

echo "======================================="
echo "CLIENT_NAME = $argClientName"
echo "PLATFORM = $argPlatform"
echo "COMPONENT_PATH = $argComponentPath"
echo "CONFIG = $argConfig"
echo "DATA_PATH = $argDataPath"
echo "======================================="

COMPOSE_API_VERSION=1.40 \
DATA_PATH=$argDataPath \
CLIENT_NAME=$argClientName \
PLATFORM=$argPlatform \
COMPONENT_PATH=$argComponentPath \
CONFIG=$argConfig \
CERTIFICATE=${argCertificate} \
bash -c \
'docker-compose --env-file project.env -f client.yaml build && \
docker-compose --env-file project.env -f client.yaml up'
