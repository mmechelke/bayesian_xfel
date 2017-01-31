#ifndef MYMATH_H
#define MYMATH_H

#define PI 3.141592653589793115997963468544
#define TWO_PI 6.283185307179586231995926937088
#define MAX_EXP 709.
#define MIN_EXP -709.

typedef double vector_double[3];
typedef vector_double matrix[3];

#define vector vector_double

#define array vector

typedef double matrix66[6][6];

PyObject *PyArray_CopyFromDimsAndData(int n_dimensions, int *dimensions, 
				      int type_num, char *data);

void matrix_dyadicproduct(matrix dest, vector a, vector b);
void matrix_iadd_dyadicproduct(matrix dest, vector a, vector b);
void matrix_print(matrix x);
void matrix_add(matrix dest, matrix a, matrix b);
void matrix_iadd(matrix dest, matrix a);
void matrix_mul(matrix dest, matrix a, matrix b);
void matrix_scale(matrix dest, matrix a, double b);
void matrix_dot(vector, matrix, vector);
void matrix_set(matrix, double);
void matrix_identity(matrix);
void matrix_copy(matrix, matrix);
void matrix_transpose(matrix, matrix);


void vector_print(vector x);
void vector_scale(vector, vector, double);
double vector_dot(vector a, vector b);
double vector_norm(vector);
void vector_add(vector dest, vector a, vector b);
void vector_iadd(vector dest, vector a);
void vector_imul(vector x, double c);
void vector_sub(vector dest, vector a, vector b);
void vector_normalize(vector v);
void vector_set(vector, double);
void vector_average(vector dest, vector *v, int n);
int vector_less(vector a, vector b);
int vector_greater(vector a, vector b);
void vector_cross(vector dest, vector a, vector b);
double vector_dihedral(vector, vector, vector, vector);
void vector_copy(vector, vector);
void vector_transform(vector, matrix, vector, vector);




PyObject *PyArray_FromVector(vector x);
PyObject *PyArray_FromMatrix(matrix x);
PyObject *PyArray_FromVectors(vector *x, int n);
PyObject *PyArray_FromMatrices(matrix *x, int n);

#endif
