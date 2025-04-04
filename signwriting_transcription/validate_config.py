# check config.yaml is a valid yaml file

import yaml

with open("config.yaml", 'r') as stream:
    try:
        yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)