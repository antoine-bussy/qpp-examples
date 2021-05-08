import os
from conans import ConanFile, tools, CMake


class QppConan(ConanFile):
    name = "qpp"
    url = "https://github.com/softwareQinc/qpp"
    homepage = "https://github.com/softwareQinc/qpp"
    description = "Quantum++ is a modern C++11 general purpose quantum computing library."
    license = ("MIT")
    topics = ("quantum computing")
    settings = "os", "compiler", "build_type", "arch"
    no_copy_source = True

    def requirements(self):
        self.requires("eigen/3.3.9@")

    def source(self):
        git = tools.Git()
        git.clone(self.url, self.version)

    def package(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.install()

    def package_id(self):
        self.info.header_only()
