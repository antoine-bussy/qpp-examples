# Quantum++ Examples

## About

Quantum++ Examples are examples of Quantum Computing in C++ using the [Quantum++](https://github.com/softwareQinc/qpp)
library. Most examples illustrate the equations, examples and exercises from
Nielsen, M., & Chuang, I. (2010). *Quantum Computation and Quantum Information: 10th Anniversary Edition*. Cambridge: Cambridge University Press.
[doi:10.1017/CBO9780511976667](https://doi.org/10.1017/CBO9780511976667)

---

## Installation instructions

Build has only been tested with the current pixi lock file, and the custom cmake provider.
This makes the build process easier.

### With pixi

Install `pixi` from [https://prefix.dev/](https://prefix.dev/).

Run a pixi shell in the project directory
```bash
pixi shell
```

You can now build and run the examples
```bash
cmake --preset default
cd _build/ninja/Release
ninja
ninja test
```

Alternatively, you can use the preset `default` directly in `VSCode`.

### Without pixi (untested)

You need to have a compiler, `cmake`, `OpenMP` installed. The versions required by the project are pretty high, although probably higher than really necessary. This is why using `pixi` is recommended.

Build and run the examples (or use `VSCode`)
```bash
cmake --preset no-pixi
cd _build/ninja/Release
ninja
ninja test
```
This will handle the dependencies (e.g `Eigen`, `googletest`) automatically by fetching them in the build directory.

Alternatively, you might ignore the custom provider and install the dependencies yourself (not recommended).
