#!/bin/bash

TEMPLATE_DIR=$(dirname "$0")
NEW_PROJECT=$1

if [ -z "$NEW_PROJECT" ]; then
  echo "Usage: $0 new_project_name"
  exit 1
fi

cp -r "$TEMPLATE_DIR" "../$NEW_PROJECT"
cd "../$NEW_PROJECT"

git init
echo "Initialized Git repo in $NEW_PROJECT"

python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
echo "Virtual environment set up and packages installed."

code .
