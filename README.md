# Quantum++ Examples

## About

Quantum++ Examples are examples of Quantum Computing in C++ using the [Quantum++](https://github.com/softwareQinc/qpp)
library. Most examples illustrate the equations, examples and exercises from
Nielsen, M., & Chuang, I. (2010). *Quantum Computation and Quantum Information: 10th Anniversary Edition*. Cambridge: Cambridge University Press.
[doi:10.1017/CBO9780511976667](https://doi.org/10.1017/CBO9780511976667)

---

## Installation instructions

Build has only been tested under Ubuntu 20.04 with g++-10

Install g++-10, as root
```bash
apt install g++-10
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 100
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 90
```
Install pip and Conan, as root
```bash
apt install python3-pip
python3 -m pip install conan
```
Install QPP with Conan
```bash
conan create third_party/conanfile.qpp.py qpp/main@local/snapshot
```

You can now build and run the examples
```bash
./scripts/conan.py ninja
cd _build/ninja/Release
source conan/activate.sh; source conan/activate_build.sh
ninja
ninja test
```
