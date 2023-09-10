import subprocess
from pathlib import Path

import addict
import yaml


def get_data_from_yaml(filename):
    with open(filename, 'r') as stream:
        data = yaml.safe_load(stream)
    data = addict.Dict(data)
    return data


def exec_cmd(cmd_template, *args, **kwargs):
    cmd_template.format(args, kwargs)
    print('Run cmd: ' + cmd_template)
    ret = subprocess.run(cmd_template, shell=True)
    return ret.returncode
