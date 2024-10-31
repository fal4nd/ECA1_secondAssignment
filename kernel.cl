__kernel void convolution(global int* input, global long* output,
    constant int* kernel_matrix, int size){

    const long INT64_MAX = 9223372036854775807;
    const long INT64_MIN = -9223372036854775808;
    const int element = get_global_id(0);
    const int row = element / size + 1;
    const int col = element % size + 1;
    const int row_size = size + 2;
    long sum = 0;

    // we store values of the input in a vector
    long16 input_vector = (long16)(
        (long)input[(row - 1) * row_size + col - 1],
        (long)input[(row - 1) * row_size + col],
        (long)input[(row - 1) * row_size + col + 1],
        (long)input[row * row_size + col - 1],
        (long)input[row * row_size + col],
        (long)input[row * row_size + col + 1],
        (long)input[(row + 1) * row_size + col - 1],
        (long)input[(row + 1) * row_size + col],
        (long)input[(row + 1) * row_size + col + 1],
        0L, 0L, 0L, 0L, 0L, 0L, 0L // padding remaining elements
    );

    //we store kernel values in a vector
    long16 kernel_vector = (long16)(
        (long)kernel_matrix[0],
        (long)kernel_matrix[1],
        (long)kernel_matrix[2],
        (long)kernel_matrix[3],
        (long)kernel_matrix[4],
        (long)kernel_matrix[5],
        (long)kernel_matrix[6],
        (long)kernel_matrix[7],
        (long)kernel_matrix[8],
        0L, 0L, 0L, 0L, 0L, 0L, 0L // padding remaining elements
    );

    // Perform element-wise multiplication and store it in product_vector
    long16 product_vector = input_vector * kernel_vector;

    int overflow_positive = 0;
    int overflow_negative = 0;

    for(int i = 0; i < 9; i++){
        //if overflow is detected, we use a variable to remember it
        overflow_positive |= ((sum > 0) & (product_vector[i] > 0) & (sum > INT64_MAX - product_vector[i]));
        overflow_negative |= ((sum < 0) & (product_vector[i] < 0) & (sum < INT64_MIN - product_vector[i]));

        sum += product_vector[i];
    }

    // it might happen that both overflows happen, we don't care which value is going to be used
    // both INT64_MIN and INT64_MAX values indicates an overflow
    sum = overflow_positive ? INT64_MAX : sum;
    sum = overflow_negative ? INT64_MIN : sum;

    output[element] = sum;
}
