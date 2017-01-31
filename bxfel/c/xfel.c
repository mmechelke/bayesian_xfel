#include "xfel.h"


void vector_add(vector dest, vector a, vector b) {
  
  int i;

  for (i = 0; i < 3; i++)
    dest[i] = a[i] + b[i];
}


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


double logsumexp(double *x, int n) {

  int i;
  double xmax, z, y=0.;
  
  xmax = x[0];
  for(i=1; i<n; i++) 
    if (x[i] > xmax) xmax = x[i];

  for(i=0; i<n; i++) {
    z = x[i] - xmax;
    if (z > -709) 
      y += exp(z);
  }

  return xmax + log(y);
}


PyObject * py_logsumexp(PyObject *self, PyObject *args) {

  double result;
  PyArrayObject *x;

  import_array();

  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &x)) 
    RAISE(PyExc_TypeError, "error in py_logsumexp_vec", NULL);

  result = logsumexp((double*) x->data, x->dimensions[0]);

  return Py_BuildValue("d", result);
}

PyObject * py_logsumexp2d(PyObject *self, PyObject *args) {

  int i, j;
  double result, *z;
  PyArrayObject *x, *y;

  import_array();

  if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &x, &PyArray_Type, &y))
    RAISE(PyExc_TypeError, "error in py_logsumexp2d", NULL);

  z = MALLOC(x->dimensions[0], double);

  for (i=0; i < y->dimensions[0]; i++) {
    for (j=0; j < x->dimensions[0]; j++) 
      z[j] = *(double*) (x->data + j * x->strides[0] + i * x->strides[1]);
    result = logsumexp(z, x->dimensions[0]);
    *(double*) (y->data + i * y->strides[0]) = result;
  }
  free(z);
  return Py_BuildValue("d", result);
}


static PyMethodDef mod_methods[] = {
  {"grid", (PyCFunction) PyGrid_New, 1},
  {"logsumexp", (PyCFunction) py_logsumexp, 1},
  {"logsumexp2d", (PyCFunction) py_logsumexp2d, 1},
  {NULL}  /* Sentinel */
};

PyMODINIT_FUNC
init_bxfel(void) 
{
  
  Py_InitModule3("_bxfel", mod_methods, "C extensions for xfel");
  import_array();
  PyGrid_Type.tp_new = PyType_GenericNew;
  
  Py_INCREF(&PyGrid_Type);
}

