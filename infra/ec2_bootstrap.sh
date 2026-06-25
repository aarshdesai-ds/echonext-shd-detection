#!/bin/bash
# EC2 bootstrap: install Docker, fetch the model bundle (public HF repo), build
# and run the combined API+demo image. Invoked from instance user-data.
set -eux

dnf install -y docker git python3-pip
systemctl enable --now docker

cd /root
rm -rf echonext-shd-detection
git clone https://github.com/aarshdesai-ds/echonext-shd-detection.git
cd echonext-shd-detection

python3 -m pip install -q "huggingface_hub<1.0"
mkdir -p models
huggingface-cli download aarshdesai04/echonext-shd-models \
  --include "v1-cnn-ens5/*" --local-dir /tmp/hfdl
cp /tmp/hfdl/v1-cnn-ens5/* models/

docker build -t echonext .
docker run -d -p 80:8080 --restart unless-stopped --name echonext echonext
