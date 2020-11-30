#!/bin/bash
# File              : generate-base-image.sh
# Author            : Sheetal Reddy <sheetal.reddy@ai.se>
# Date              : 11.11.2020
# Last Modified Date: 11.11.2020
# Last Modified By  : Sheetal Reddy <sheetal.reddy@ai.se>

pushd base-images
docker build -f nvidia_platform.dockerfile -t fedbird:latest .
popd
