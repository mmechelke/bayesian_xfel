#ifndef __XFEL_H__
#define __XFEL_H__

#include "Python.h"
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

#define RAISE(a,b,c) {PyErr_SetString(a, b); return c;}
#define HERE {printf("%s: %d\n",__FILE__,__LINE__);}
#define MALLOC(n, t) ((t*) malloc((n) * sizeof(t)))
#define REALLOC(p, n, t) ((t*) realloc((void*) (p), (n) * sizeof(t)))
#define MEMCPY(a,b,n,t) memcpy((void*) (a), (void*) (b), (n) * sizeof(t))
#define RETURN_PY_NONE {Py_INCREF(Py_None);return Py_None;}
#define MIN_EXP -709.
#define DUMP_ERROR(a) dump_error(a);
#define INC_AND_RETURN(a) {Py_INCREF((PyObject*) (a)); return (PyObject*) (a);}

#define SET_PYOBJECT(op, dest) {if (dest) Py_DECREF(dest); if (op) Py_INCREF(op); dest = (op);}
#define SET_PYARRAY(dest, op) {if (dest) Py_DECREF(dest); if (op == Py_None) dest = NULL; else {Py_INCREF(op); dest = (PyArrayObject*) (op);}}

#define PI 3.141592653589793115997963468544
#define TWO_PI 6.283185307179586231995926937088
#define MAX_EXP 709.
#define MIN_EXP -709.

typedef double vector_double[3];
typedef vector_double matrix[3];
typedef double matrix66[6][6];

#define vector vector_double
#define array vector

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

double logsumexp(double *x, int n);


typedef struct {

  PyObject_HEAD 
  
  int nx, ny, nz;   /* number of cell in each dimension */
  double spacing;   /* spacing */
  double width;     /* kernel width of density */
  vector origin;    /* lower left corner */

  int n_shells;     /* number of shells taken into account
		       in density calculation */

  double *values;   /* density, generally: scalar property
		       assigned to grid point */

  double *neighbor_density; 
  int *neighbors;
  int *neighbors_ijk; 
  int n_neighbors;

} PyGridObject;

PyObject * PyGrid_New(PyObject *self, PyObject *args);

extern PyTypeObject PyGrid_Type;


#endif
