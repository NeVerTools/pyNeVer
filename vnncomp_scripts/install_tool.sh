#!/bin/bash

# NeVer2
# install_tool.sh script for VNN-COMP 2024

TOOL_NAME=NeVer2
VERSION_STRING=v1

# check arguments
if [ "$1" != ${VERSION_STRING} ]; then
	echo "Expected first argument (version string) '$VERSION_STRING', got '$1'"
	exit 1
fi

echo "Installing $TOOL_NAME"
DIR=$(dirname $(dirname $(realpath $0)))

apt-get update &&
apt-get install -y python3 python3-pip &&
apt-get install -y psmisc && # for killall, used in prepare_instance.sh script
pip3 install -r "$DIR/vnncomp_scripts/requirements.txt"