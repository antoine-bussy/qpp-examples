from conans import ConanFile, CMake, tools
import os

class QppExamplesConan(ConanFile):

    name = "qpp-examples"
    version = "0.0.0"
    license = "MIT"
    author = "Antoine Bussy bussyantoine@gmail.com"
    description = "QPP Examples"
    topics = ("Quantum Computing", "Examples")
    settings = "os", "compiler", "build_type", "arch"
    options = {
        "cmake_extra_options": "ANY",
        }
    default_options = {
        "cmake_extra_options": None,
        }
    generators = "cmake_find_package_multi"
    short_paths = True

    def requirements(self):
        self.requires("tbb/2020.3@")

    def build_requirements(self):
        self.build_requires("qpp/main@local/snapshot", force_host_context=True)
        self.build_requires("cmake/3.20.1@")
        if "Ninja" in os.environ.get('CONAN_CMAKE_GENERATOR', ""):
            self.build_requires("ninja/1.10.2@")
        self.build_requires("gtest/cci.20210126@", force_host_context=True)

    default_user = "a-bussy"
    @property
    def default_channel(self):
        return "stable"

    scm = {
         "type": "git",
         "revision": "auto",
     }

    def _configure_cmake(self):
        cmake = CMake(self)
        cmake.definitions["QPP_EXAMPLES_VERSION"] = self.version
        cmake_extra_options = str(self.options.cmake_extra_options).split() if self.options.cmake_extra_options else None
        cmake.configure(args=cmake_extra_options)
        return cmake

    def build(self):
        cmake = self._configure_cmake()
        cmake.build()
        cmake.install()
        cmake.test(output_on_failure=True)

    def package_id(self):
        del self.info.options.cmake_extra_options
