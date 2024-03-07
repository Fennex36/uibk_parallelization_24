# UIBK VU HPC implementations A: Parallelisation / PR Methods in Computational Physics B: Parallelisation

This repository contains material required to complete exercises for the
VU/PR in the 2024 summer semester, including assignment sheets and any
associated materials. Note that some of this material is based on previous
years.

**Make sure to read this document carefully, as it contains vital information
regarding the grading of exercises.**

## Topics

Generally, topics discussed here include

- parallel programming concepts
- shared memory parallelization using OpenMP
- performance-oriented programming
- proper experiment orchestration and benchmarking
- parallel programming tools

## General Information

All programming exercise solutions you hand in **must** compile and run on the
LCC3 cluster as described in the [LCC3 tutorial](lcc3_tutorial.md). One way
way of ensuring this is by developing on LCC3 directly, either through a
terminal or using an editor/IDE with support for remote development via SSH
(such as [Visual Studio Code](https://code.visualstudio.com/docs/remote/ssh)).

Note that the default version of the GCC compiler present on LCC3 is very
outdated. You may load a never version by calling `module load gcc/12.2.0-gcc-8.5.0-p4pe45v`.

**As the tutorial states, use job scripts to run your applications on LCC3
compute nodes, do NOT run applications directly on the LCC3 login node.**

### Connecting to LCC3 via SSH

**NOTE**: You have to be inside the University network to be able to connect via
SSH. See [here](https://www.uibk.ac.at/zid/netz-komm/vpn/) for more information.

The hostname of LCC3 is `login.lcc3.uibk.ac.at`.

### Coding Guidelines

All programming exercises are conducted in C or C++.

Your code should always compile without warnings, when passing the flags `-Wall
-Wextra -Werror -std=gnu11`. Error handling is your discretion, but wherever you
do or do not include error handling you need to be able to justify it.

Make sure your code is properly formatted using either your IDE/Text editor of
choice, or by using a tool such as `clang-format`. You can find an example
[.clang-format](.clang-format) file in this repository. **Failure to
consistently format code may result in lower scores.**

Try to write _self-documenting code_ by choosing descriptive variable and
function names. While you may want to add comments to certain sections of your
code, try to avoid trivial comments such as `fopen(...); // open file`. The best
source code comments are the ones you do NOT need to write. **All names and
comments should be written in English**.

Finally, all submitted code must be accompanied by a GNU Makefile. That is, your
solution must be able to be compiled with a single call to `make`. You can find
an example Makefile and more information [here](example_makefile).
