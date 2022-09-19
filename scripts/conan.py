#!/usr/bin/env python3

import subprocess
import argparse
import platform
from pathlib import Path

# Example:
# python scripts/conan.py ninja

generators = [ "ninja" ]
configs = [ "Release", "Debug", 'RelWithDebInfo' ]
def multi_config_generator(generator):
    return False

parser = argparse.ArgumentParser(description="Run conan generation.")
parser.add_argument("generator", help="generator name", choices=generators)
parser.add_argument('-c', '--config', help="build configuration", type=str, default='Release', choices=configs)
parser.add_argument('-d', '--debug', help="build in debug", action='store_const', const='Debug', dest='config')
parser.add_argument('-r', '--release', help="build in release", action='store_const', const='Release', dest='config')
parser.add_argument('--skip-cmake', help="skip cmake configuration step", action='store_true')
parser.add_argument('-bf', '--build-folder', help="build folder path", type=str, default="")
parser.add_argument('-s', '--debug-stream', help="enable debug stream", action='store_true')
parser.add_argument("extra_options", nargs=argparse.REMAINDER, help="additional options")
args = parser.parse_args()

generator = args.generator
folder_name = "".join(generator)
build_folder = args.build_folder if args.build_folder else f"_build/{folder_name}"
if not multi_config_generator(generator):
    if not args.build_folder:
        build_folder += f"/{args.config}"
    configs = [ args.config ]

conan_os = platform.system()
arch = "x86_64"

conan_command = [ "conan", "install", "-g", "virtualbuildenv", "-g", "virtualenv", "-if", f"{build_folder}/conan", "--profile:build=default" ]

conan_command += [ "-s", f"os={conan_os}" ]
conan_command += [ "-s", f"arch={arch}" ]

if generator == "ninja":
    conan_command += [ "-e", "CONAN_CMAKE_GENERATOR=Ninja" ]
else:
    raise ValueError(f"Unknown generator: {generator}")

if conan_os == "Linux":
    conan_command += [ "-s", "compiler.libcxx=libstdc++11" ]
else:
    raise ValueError(f"Unknown OS: {conan_os}")

for config in configs:
    conan_command_config = conan_command + [ "-s", f"build_type={config}" ]
    conan_command_config += [ "." ]
    conan_command_config += [ "-o", f"cmake_extra_options=-DDEBUG_STREAM={'on' if args.debug_stream else 'off'}" ]
    conan_command_config += args.extra_options
    print(' '.join(conan_command_config))
    subprocess.run(' '.join(conan_command_config), shell=True, check=True)

if not args.skip_cmake:
    configure_command = [ "conan", "build", ".", "-c", "-if", f"{build_folder}/conan", "-bf", build_folder ]
    print(' '.join(configure_command))
    subprocess.run(' '.join(configure_command), shell=True, check=True)

# Fix environment scripts for p10k
environments = [ "", "build" ]
for env in environments:
    activate_script = Path(f"{build_folder}/conan/activate{'_' if env else ''}{env}.sh")
    if not activate_script.is_file():
        continue
    with activate_script.open("a") as file:
        file.write('\nexport CONAN_OLD_CONAN_VENV="$CONAN_VENV"')
        file.write(f'\nexport CONAN_VENV="(conan{env}env) $CONAN_VENV"\n')

    deactivate_script = Path(f"{build_folder}/conan/deactivate{'_' if env else ''}{env}.sh")
    if not deactivate_script.is_file():
        continue
    with deactivate_script.open("a") as file:
        file.write('\nexport CONAN_VENV="$CONAN_OLD_CONAN_VENV"')
        file.write('\nunset CONAN_OLD_CONAN_VENV\n')
