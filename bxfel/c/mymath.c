/*
##
## Math utility functions
##
## Author: Michael Habeck
##         MPI fuer Entwicklungsbiologie, Tuebingen
##        
##         Copyright (C) 2007 Michael Habeck
##         All rights reserved.
##         No warranty implied or expressed.
##
*/

#include "xfel.h"

void matrix_print(matrix x) {
  int i,j;

  for (i=0;i<3;i++) {
    for (j=0;j<3;j++)
      printf("%e ",x[i][j]);
    printf("\n");
  }
}

void vector_print(vector x) {
  int i;
  
  printf("[ ");
  for (i=0;i<3;i++) printf("%e ",x[i]);
  printf("]\n");
}

void matrix_dyadicproduct(matrix dest, vector a, vector b) {

  int i, j;

  for (i = 0; i < 3; i++) for (j = 0; j < 3; j++)
      dest[i][j] = a[i] * b[j];
}

void matrix_iadd_dyadicproduct(matrix dest, vector a, vector b) {

  int i, j;

  for (i = 0; i < 3; i++) for (j = 0; j < 3; j++)
      dest[i][j] += a[i] * b[j];
}

void matrix_dot(vector b, matrix A, vector a) {

  b[0] = A[0][0] * a[0] + A[0][1] * a[1] + A[0][2] * a[2];
  b[1] = A[1][0] * a[0] + A[1][1] * a[1] + A[1][2] * a[2];
  b[2] = A[2][0] * a[0] + A[2][1] * a[1] + A[2][2] * a[2];

}

void matrix_add(matrix dest, matrix a, matrix b) {

  int i, j;

  for (i = 0; i < 3; i++) for (j = 0; j < 3; j++)
    dest[i][j] = a[i][j] + b[i][j];
}

void matrix_iadd(matrix dest, matrix a) {

  int i, j;

  for (i = 0; i < 3; i++) for (j = 0; j < 3; j++)
    dest[i][j] += a[i][j];
}

void matrix_mul(matrix dest, matrix a, matrix b) {
  
  int i, j, k;

  for (i = 0; i < 3; i++) 
    for (j = 0; j < 3; j++) {
      dest[i][j] = 0.;
      for (k = 0; k < 3; k++)
	dest[i][j] += a[i][k] * b[k][j];
    }
}

void matrix_scale(matrix dest, matrix a, double scale) {

  int i, j;

  for (i = 0; i < 3; i++) for (j = 0; j < 3; j++)
    dest[i][j] = scale * a[i][j];
}

void matrix_set(matrix dest, double val) {

  int i, j;

  for (i = 0; i < 3; i++) for (j = 0; j < 3; j++) 
    dest[i][j] = val;
}

void matrix_identity(matrix dest) {

  dest[0][0] = 1.; dest[0][1] = 0., dest[0][2] = 0.;
  dest[1][0] = 0.; dest[1][1] = 1., dest[1][2] = 0.;
  dest[2][0] = 0.; dest[2][1] = 0., dest[2][2] = 1.;

}

void matrix_copy(matrix dest, matrix src) {

  int i, j;
  
  for (i = 0; i < 3; i++) for (j = 0; j < 3; j++) 
      dest[i][j] = src[i][j];
}

void matrix_transpose(matrix dest, matrix src) {

  int i, j;
  
  for (i = 0; i < 3; i++) for (j = 0; j < 3; j++) 
      dest[j][i] = src[i][j];
}

void vector_scale(vector b, vector a, double scale) {

  int i;

  for (i = 0; i < 3; i++) b[i] = scale * a[i];
}

double vector_dot(vector a, vector b) {

  int i;
  double d;

  d = 0.;

  for (i = 0; i < 3; i++)
    d += a[i] * b[i];

  return d;
}

void vector_imul(vector x, double c) {
  
  int i;

  for (i = 0; i < 3; i++) x[i] *= c;
}

void vector_add(vector dest, vector a, vector b) {
  
  int i;

  for (i = 0; i < 3; i++)
    dest[i] = a[i] + b[i];
}

void vector_iadd(vector dest, vector a) {
  
  int i;

  for (i = 0; i < 3; i++) dest[i] += a[i];
}

void vector_sub(vector dest, vector a, vector b) {
  
  int i;

  for (i = 0; i < 3; i++)
    dest[i] = a[i] - b[i];
}

void vector_normalize(vector v) {
  int i;
  double sum;

  sum = 0.;

  for (i = 0; i < 3; i++) sum += v[i] * v[i];

  sum = sqrt(sum);

  if (sum != 0.0)
    for (i = 0; i < 3; i++) v[i] /= sum;
}

double vector_norm(vector v) {
  int i;
  double sum;

  sum = 0.;

  for (i = 0; i < 3; i++) sum += v[i] * v[i];

  return sqrt(sum);
}

void vector_copy(vector v, vector w) {
  int i;

  for (i = 0; i < 3; i++) v[i] = w[i];
}

void vector_cross(vector c, vector a, vector b) {

  /* Cross product between two 3D-vectors. */

  c[0] = a[1]*b[2] - a[2]*b[1];
  c[1] = a[2]*b[0] - a[0]*b[2];
  c[2] = a[0]*b[1] - a[1]*b[0]; 
}

void vector_transform(vector b, matrix A, vector a, vector c) {

  b[0] = A[0][0] * a[0] + A[0][1] * a[1] + A[0][2] * a[2] + c[0];
  b[1] = A[1][0] * a[0] + A[1][1] * a[1] + A[1][2] * a[2] + c[1];
  b[2] = A[2][0] * a[0] + A[2][1] * a[1] + A[2][2] * a[2] + c[2];

}

void vector_set(vector v, double val) {
  int i;

  for (i = 0; i < 3; i++) v[i] = val;
}

void vector_average(vector dest, vector *v, int n) {
  int i;

  vector_set(dest, 0.);

  for (i = 0; i < n; i++) 
    vector_add(dest, dest, v[i]);

  for (i = 0; i < 3; i++) dest[i] /= n;
}

int vector_less(vector a, vector b) {

  /* return 1 if a[i] < b[i] for all i */

  if (a[0] > b[0] || a[1] > b[1] || a[2] > b[2])
    return 0;
  else
    return 1;
}

int vector_greater(vector a, vector b) {

  /* return 1 if a[i] > b[i] for all i */

  if (a[0] < b[0] || a[1] < b[1] || a[2] < b[2])
    return 0;
  else
    return 1;
}

PyObject *PyArray_CopyFromDimsAndData(int n_dimensions, int *dimensions, 
				      int type_num, char *data) {
  /*
    This method is similar to PyArray_FromDimAndData. It creates a new
    PyArrayObject, but instead of referencing 'data', it returns a
    copy of it.
   */

  PyObject *a1, *a2;

  a1 = PyArray_FromDimsAndData(n_dimensions, dimensions, type_num, data);
  a2 = PyArray_Copy((PyArrayObject*) a1);

  Py_DECREF(a1);

  return PyArray_Return((PyArrayObject*) a2);
}

PyObject *PyArray_FromVector(vector x) {
 
  int dims[1];

  PyArrayObject *y;
  
  dims[0] = 3;

  y = (PyArrayObject*) PyArray_FromDims(1, dims, PyArray_DOUBLE);

  vector_copy(((vector*)y->data)[0], x);
  
  return (PyObject*) y;
}

PyObject *PyArray_FromMatrix(matrix x) {
 
  int dims[2];

  PyArrayObject *y;
  
  dims[0] = dims[1] = 3;

  y = (PyArrayObject*) PyArray_FromDims(2, dims, PyArray_DOUBLE);

  matrix_copy(((matrix*)y->data)[0], x);

  return (PyObject*) y;
}

PyObject *PyArray_FromVectors(vector *x, int n) {

  int i, dims[2];

  PyArrayObject *y;

  dims[0] = n;
  dims[1] = 3;

  y = (PyArrayObject*) PyArray_FromDims(2, dims, PyArray_DOUBLE);

  for (i = 0; i < n; i++) vector_copy(((vector*)y->data)[i], x[i]);

  return (PyObject*) y;

}

PyObject *PyArray_FromMatrices(matrix *x, int n) {

  int i, dims[3];

  PyArrayObject *y;

  dims[0] = n;
  dims[1] = dims[2] = 3;

  y = (PyArrayObject*) PyArray_FromDims(3, dims, PyArray_DOUBLE);

  for (i = 0; i < n; i++) matrix_copy(((matrix*)y->data)[i], x[i]);

  return (PyObject*) y;

}
