# Makefile for compiling and running matrix_generator and convolution

# Compiler
CC = gcc

# Compiler flags
CFLAGS = -Wall

# Executable names
MATRIX_GENERATOR = matrix_generator
CONVOLUTION = convolution

# Source files
MATRIX_SOURCE = matrix_generator.c
CONVOLUTION_SOURCE = convolution.c

# Target to run all
all: $(MATRIX_GENERATOR) $(CONVOLUTION)

# Compile matrix_generator
$(MATRIX_GENERATOR): $(MATRIX_SOURCE)
	$(CC) $(CFLAGS) $(MATRIX_SOURCE) -o $(MATRIX_GENERATOR)

# Compile convolution
$(CONVOLUTION): $(CONVOLUTION_SOURCE)
	$(CC) $(CFLAGS) $(CONVOLUTION_SOURCE) -lOpenCL -o $(CONVOLUTION)

# Run matrix_generator if matrix.txt does not exist, then run convolution
run: $(MATRIX_GENERATOR) $(CONVOLUTION)
	@if [ ! -f matrix.txt ]; then \
		echo "matrix.txt not found. Running matrix_generator..."; \
		./$(MATRIX_GENERATOR); \
	else \
		echo "matrix.txt exists. Skipping matrix_generator."; \
	fi
	./$(CONVOLUTION)

# Clean up generated files
clean:
	rm -f $(MATRIX_GENERATOR) $(CONVOLUTION) matrix.txt

.PHONY: all run clean
