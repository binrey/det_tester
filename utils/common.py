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


def set_cpp_defines(filepath, tokens):
    file = Path(filepath)
    if file.exists():
        with file.open(mode='r') as f:
            lines = f.readlines()
        with file.open(mode='w') as f:
            new_lines = []
            for line in lines:
                new_line = line
                for k in tokens:
                    if line.startswith('#define') and k in line:
                        if tokens[k][1]:
                            new_line = '#define ' + str(tokens[k][0]) + '\n'
                        else:
                            new_line = '#define ' + str(k) + ' ' + str(tokens[k][0]) + '\n'
                new_lines.append(new_line)
            f.writelines(new_lines)
    else:
        raise FileNotFoundError('{} is not found'.format(file))


def set_cpp_vars(filepath, tokens):
    file = Path(filepath)
    if file.exists():
        with file.open(mode='r') as f:
            lines = f.readlines()
        with file.open(mode='w') as f:
            new_lines = []
            for line in lines:
                new_line = line
                for k in tokens:
                    if k in line:
                        line_parts = new_line.split(' ')
                        ind = line_parts.index(k)
                        new_line = ' '.join(line_parts[0:ind + 1]) + ' = {};\n'.format(str(tokens[k][0]))
                new_lines.append(new_line)
            f.writelines(new_lines)
    else:
        raise FileNotFoundError('{} is not found'.format(file))
