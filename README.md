# Quantum++ Examples

## About

Quantum++ Examples are examples of Quantum Computing in C++ using the [Quantum++](https://github.com/softwareQinc/qpp)
library. Most examples illustrate the equations, examples and exercises from
Nielsen, M., & Chuang, I. (2010). *Quantum Computation and Quantum Information: 10th Anniversary Edition*. Cambridge: Cambridge University Press.
[doi:10.1017/CBO9780511976667](https://doi.org/10.1017/CBO9780511976667)

---

## Installation instructions

Build has only been tested under Ubuntu 22.04 with g++-12

Install g++-12, as root
```bash
apt install g++-12
update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 120
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
