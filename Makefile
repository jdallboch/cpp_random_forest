# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -std=c++14 -Wall -Wextra -pedantic -g

# Source files
SRCS = main.cpp random_forest.cpp tree.cpp node.cpp data_helpers.cpp metrics.cpp

# Header files
HDRS = random_forest.hpp tree.hpp node.hpp data_helpers.hpp metrics.hpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Executable name
EXEC = rf_program

# Default target
all: $(EXEC)

# Link the object files to create the executable
$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile source files into object files
%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up the build files
clean:
	rm -f $(OBJS) $(EXEC)

# Phony targets
.PHONY: all clean
