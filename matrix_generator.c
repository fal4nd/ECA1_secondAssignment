#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

#define ROWS 1300
#define COLS 1300
#define MIN_VAL 0
#define MAX_VAL 10

int main() {
    FILE *file;
    uint32_t **matrix;  // Pointer to store the matrix (array of pointers)
    int i, j;

    // Initialize random number generator
    srand(time(NULL));

    // Allocate memory for the matrix (dynamic allocation of a 2D array)
    matrix = (uint32_t **)malloc(ROWS * sizeof(uint32_t *));
    if (matrix == NULL) {
        perror("Memory allocation failed");
        return EXIT_FAILURE;
    }

    for (i = 0; i < ROWS; i++) {
        matrix[i] = (uint32_t *)malloc(COLS * sizeof(uint32_t));
        if (matrix[i] == NULL) {
            perror("Memory allocation failed");
            return EXIT_FAILURE;
        }
    }

    // Fill the matrix with random values between MIN_VAL and MAX_VAL
    for (i = 0; i < ROWS; i++) {
        for (j = 0; j < COLS; j++) {
            if(rand() % 2){
                matrix[i][j] = rand() % INT32_MAX;  // Generate random value
            }else{
                matrix[i][j] = (-1) * rand() % INT32_MAX;
            }
        }
    }

    // Open the file to write the matrix
    file = fopen("matrix.txt", "w");
    if (file == NULL) {
        perror("Error opening file");
        return EXIT_FAILURE;
    }

    // Write the matrix to the file
    for (i = 0; i < ROWS; i++) {
        for (j = 0; j < COLS; j++) {
            fprintf(file, "%d", matrix[i][j]);  // Write each value
            if (j < COLS - 1) {
                fprintf(file, " ");  // Add space between columns
            }
        }
        fprintf(file, "\n");  // Newline after each row
    }

    // Close the file
    fclose(file);

    // Free the allocated memory
    for (i = 0; i < ROWS; i++) {
        free(matrix[i]);
    }
    free(matrix);

    printf("Matrix saved to matrix.txt\n");
    return 0;
}
