#!/bin/bash

pushd base-images
docker build -f nvidia_platform.dockerfile -t fedbird:latest .
popd
