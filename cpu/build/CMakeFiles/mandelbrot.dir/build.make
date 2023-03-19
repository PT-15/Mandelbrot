# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Disable VCS-based implicit rules.
% : %,v


# Disable VCS-based implicit rules.
% : RCS/%


# Disable VCS-based implicit rules.
% : RCS/%,v


# Disable VCS-based implicit rules.
% : SCCS/s.%


# Disable VCS-based implicit rules.
% : s.%


.SUFFIXES: .hpux_make_needs_suffix_list


# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/morgana/projects/mandelbrot

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/morgana/projects/mandelbrot/build

# Include any dependencies generated for this target.
include CMakeFiles/mandelbrot.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/mandelbrot.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/mandelbrot.dir/flags.make

CMakeFiles/mandelbrot.dir/src/main.cpp.o: CMakeFiles/mandelbrot.dir/flags.make
CMakeFiles/mandelbrot.dir/src/main.cpp.o: ../src/main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/morgana/projects/mandelbrot/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/mandelbrot.dir/src/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/mandelbrot.dir/src/main.cpp.o -c /home/morgana/projects/mandelbrot/src/main.cpp

CMakeFiles/mandelbrot.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/mandelbrot.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/morgana/projects/mandelbrot/src/main.cpp > CMakeFiles/mandelbrot.dir/src/main.cpp.i

CMakeFiles/mandelbrot.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/mandelbrot.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/morgana/projects/mandelbrot/src/main.cpp -o CMakeFiles/mandelbrot.dir/src/main.cpp.s

# Object files for target mandelbrot
mandelbrot_OBJECTS = \
"CMakeFiles/mandelbrot.dir/src/main.cpp.o"

# External object files for target mandelbrot
mandelbrot_EXTERNAL_OBJECTS =

mandelbrot: CMakeFiles/mandelbrot.dir/src/main.cpp.o
mandelbrot: CMakeFiles/mandelbrot.dir/build.make
mandelbrot: CMakeFiles/mandelbrot.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/morgana/projects/mandelbrot/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable mandelbrot"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/mandelbrot.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/mandelbrot.dir/build: mandelbrot

.PHONY : CMakeFiles/mandelbrot.dir/build

CMakeFiles/mandelbrot.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/mandelbrot.dir/cmake_clean.cmake
.PHONY : CMakeFiles/mandelbrot.dir/clean

CMakeFiles/mandelbrot.dir/depend:
	cd /home/morgana/projects/mandelbrot/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/morgana/projects/mandelbrot /home/morgana/projects/mandelbrot /home/morgana/projects/mandelbrot/build /home/morgana/projects/mandelbrot/build /home/morgana/projects/mandelbrot/build/CMakeFiles/mandelbrot.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/mandelbrot.dir/depend

