#include "xfel.h"
#include <stdio.h>

#define WEIGHT_CUTOFF -100
#define NORMALIZE_CC 1

static int create_shifts(PyGridObject *self, int **s) {

  int i, j, index, d3;

  d3 = pow(3, 3);

  for (i = 0; i < d3; i++) {
    index = i;
    for (j = 2; j > 0; j--) {
      s[i][j] = index % 3 - 1;
      index /= 3;
    }
    s[i][j] = index - 1;
  }
  return 0;
}

static int grid_index(PyGridObject *self, int *grid_coords) {

  int i, grid_index;

  grid_index = grid_coords[0];
  grid_index = grid_index * self->ny + grid_coords[1];
  grid_index = grid_index * self->nz + grid_coords[2];

  return grid_index;
}

static int assign_point(PyGridObject *self, double *x, int *coords) {

  int i;
  double y;

  for (i = 0; i < 3; i++) {
    y = (x[i] - self->origin[i]) / self->spacing;
    coords[i] = floor(y);
    if (y - coords[i] > 0.5) coords[i]++;
  }
  
  return grid_index(self, coords);
}

static int grid_coordinates(PyGridObject *self, int grid_index, int *coord) {
  /*
    only works for points inside the grid boundaries, i.e. points with
    non-negative grid-coordinates and smaller than self->shape
  */

  coord[2] = grid_index % self->nz;
  grid_index /= self->nz;

  coord[1] = grid_index % self->ny;
  grid_index /= self->ny;

  coord[0] = grid_index;

  return 0;
}	

static int in_grid(PyGridObject *self, int *coords) {

  if (coords[0] < 0)
    return 0;
  else if (coords[0] >= self->nx)
    return 0;
  else if (coords[1] < 0)
    return 0;
  else if (coords[1] >= self->ny)
    return 0;
  else if (coords[2] < 0)
    return 0;
  else if (coords[2] >= self->nz)
    return 0;
  return 1;
}

static int setup_neighbors(PyGridObject *self) {

  int i, j, k, index, n_neighbors, n = 0;
  int coords[3];
  double rho;

  if (self->neighbors) free(self->neighbors);
  if (self->neighbors_ijk) free(self->neighbors_ijk);
  if (self->neighbor_density) free(self->neighbor_density);

  n_neighbors = 8 * self->n_shells * self->n_shells * self->n_shells;

  rho = exp(-self->spacing*self->spacing/self->width/self->width);

  if (!(self->neighbors = MALLOC(n_neighbors, int)))
    return -1;
  if (!(self->neighbors_ijk = MALLOC(3 * n_neighbors, int)))
    return -1;
  if (!(self->neighbor_density = MALLOC(n_neighbors, double)))
    return -1;

  for (i = -self->n_shells; i <= self->n_shells; i++) {

    coords[0] = i;

    for (j = -self->n_shells; j <= self->n_shells; j++) {
      
      //      if (fabs(i) + fabs(j) > self->n_shells) continue;
      if (i*i + j*j > self->n_shells * self->n_shells) continue;
      coords[1] = j;

      for (k = -self->n_shells; k <= self->n_shells; k++) {
	
	//	if (fabs(i) + fabs(j) + fabs(k) > self->n_shells) continue;
	if (i*i + j*j + k*k > self->n_shells * self->n_shells) continue;
	coords[2] = k;
	index = grid_index(self, coords);	
	self->neighbors[n] = index;
	self->neighbors_ijk[3 * n + 0] = i;
	self->neighbors_ijk[3 * n + 1] = j;
	self->neighbors_ijk[3 * n + 2] = k;
	self->neighbor_density[n] = pow(rho, i*i+j*j+k*k);
	n++;
      }
    }
  }

  self->n_neighbors = n;

  if (!(self->neighbors = REALLOC(self->neighbors, n, int)))
    return -1;
  if (!(self->neighbors_ijk = REALLOC(self->neighbors_ijk, 3 * n, int)))
    return -1;
  if (!(self->neighbor_density = REALLOC(self->neighbor_density, n, double)))
    return -1;
  return 0;
}

static int set_density(PyGridObject *self, double value) {
  /*
    set density to some constant value
  */
  int i;

  for (i = 0; i < self->nx * self->ny * self->nz; i++)
    self->values[i] = value;
  return 0;
}

static int convolve(PyGridObject *self, double *rho, double cutoff) {

  int i, j, k, a, b, c, index, radius, coords[3];
  double rho_ijk, s, w, log_Z;

  w = 2. * self->width * self->width;
  s = self->spacing * self->spacing;
  radius = self->n_shells * self->n_shells;

  log_Z = 1.5 * log(PI * w);

  set_density(self, 0.);

  for (i = 0; i < self->nx; i++) {
    coords[0] = i;

    for (j = 0; j < self->ny; j++) {
      coords[1] = j;

      for (k = 0; k < self->nz; k++) {
	coords[2] = k;

	rho_ijk = rho[grid_index(self, coords)];

	if (rho_ijk < cutoff) continue;

	for (a = -self->n_shells; a <= self->n_shells; a++) {

	  // in grid?
  
	  if (i + a < 0) continue;
	  else if (i + a >= self->nx) continue;

	  // calc contribution

	  for (b = -self->n_shells; b <= self->n_shells; b++) {

	    // in grid and sphere?
      
	    if (j + b < 0) continue;
	    else if (j + b >= self->ny) continue;
	    if (a*a + b*b > radius) continue;

	    // calc contribution

	    for (c = -self->n_shells; c <= self->n_shells; c++) {

	      // in grid and sphere?

	      if (k + c < 0) continue;
	      else if (k + c >= self->nz) continue;
	      if (a*a + b*b + c*c > radius) continue;

	      // calc contribution

	      index = self->nz * (self->ny * (i+a) + j+b) + k+c;

	      self->values[index] += rho_ijk * exp(-(a*a+b*b+c*c)*s/w - log_Z);
	    }
	  }
	}
      }	
    }
  }

  return 0;
}

static int add_density(PyGridObject *self, double *x, double width, 
		       double weight) {

  int i, j, k, a, b, c, index, radius, coords[3];
  double rho_ijk, rho_a, rho_b, rho_c, delta_rho, w;
  vector delta;

  index = assign_point(self, x, coords);

  w = 2. * width * width;

  rho_ijk = - 3 * log(width) + log(weight);

  for (i = 0; i < 3; i++) {
    
    delta[i] = x[i] - self->origin[i] - coords[i] * self->spacing;
    rho_ijk -= delta[i] * delta[i] / w;
  }  

  delta_rho = -self->spacing*self->spacing / w;

  radius = self->n_shells * self->n_shells;
  
  i = coords[0];
  j = coords[1];
  k = coords[2];
  
  for (a = -self->n_shells; a <= self->n_shells; a++) {

    // in grid?
  
    if (i + a < 0) continue;
    else if (i + a >= self->nx) continue;

    // calc contribution

    rho_a = a * (a * delta_rho + 2 * self->spacing * delta[0] / w);

    for (b = -self->n_shells; b <= self->n_shells; b++) {

      // in grid and sphere?
      
      if (j + b < 0) continue;
      else if (j + b >= self->ny) continue;
      if (a*a + b*b > radius) continue;

      // calc contribution

      rho_b = b * (b * delta_rho + 2 * self->spacing * delta[1] / w);

      for (c = -self->n_shells; c <= self->n_shells; c++) {

	// in grid and sphere?

	if (k + c < 0) continue;
	else if (k + c >= self->nz) continue;
	if (a*a + b*b + c*c > radius) continue;

	// calc contribution

	rho_c = c * (c * delta_rho + 2 * self->spacing * delta[2] / w);

	index = self->nz * (self->ny * (i+a) + j+b) + k+c;

	self->values[index] += exp(rho_ijk + rho_a + rho_b + rho_c);
      }
    }
  }

  return 0;
}

static int add_mask(PyGridObject *self, double *x, double alpha, double beta, 
		    int m) {

  int i, j, k, a, b, c, index, radius, coords[3];
  double d, d_a, d_b, d_c, delta_rho;
  vector r;

  index = assign_point(self, x, coords);

  d = 0;

  for (i = 0; i < 3; i++) {
    
    r[i] = x[i] - self->origin[i] - coords[i] * self->spacing;
    d += r[i] * r[i];
  }  

  delta_rho = - self->spacing * self->spacing;

  radius = m * m;
  
  i = coords[0];
  j = coords[1];
  k = coords[2];
  
  for (a = -m; a <= m; a++) {

    // in grid?
  
    if (i + a < 0) continue;
    else if (i + a >= self->nx) continue;

    // calc contribution

    d_a = a * (a * delta_rho + 2 * self->spacing * r[0]);

    for (b = -m; b <= m; b++) {

      // in grid and sphere?
      
      if (j + b < 0) continue;
      else if (j + b >= self->ny) continue;
      if (a*a + b*b > radius) continue;

      // calc contribution

      d_b = b * (b * delta_rho + 2 * self->spacing * r[1]);

      for (c = -m; c <= m; c++) {

	// in grid and sphere?

	if (k + c < 0) continue;
	else if (k + c >= self->nz) continue;
	if (a*a + b*b + c*c > radius) continue;

	// calc contribution

	d_c = c * (c * delta_rho + 2 * self->spacing * r[2]);

	index = self->nz * (self->ny * (i+a) + j+b) + k+c;

	self->values[index] += log(1-exp(alpha * (d + d_a + d_b + d_c - beta)));
      }
    }
  }

  return 0;
}

static int update_gradient(PyGridObject *self, double *x, double width, 
			   double weight, double *grad, double *grad_xyz) {

  int i, j, k, a, b, c, index, radius, coords[3];
  double rho_ijk, rho_a, rho_b, rho_c, delta_rho, w, value, dx, dy, dz;
  vector delta, dd;

  index = assign_point(self, x, coords);

  w = 2. * width * width;

  rho_ijk = 0.; 

  for (i = 0; i < 3; i++) {    
    delta[i] = x[i] - self->origin[i] - coords[i] * self->spacing;
    rho_ijk -= delta[i] * delta[i] / w;
    grad_xyz[i] = 0.;
    dd[i] = 2 * self->spacing / w * delta[i];
  }  

  rho_ijk = exp(rho_ijk);

  delta_rho = - self->spacing * self->spacing / w;

  radius = self->n_shells * self->n_shells;
  
  i = coords[0];
  j = coords[1];
  k = coords[2];

  for (a = -self->n_shells; a <= self->n_shells; a++) {

    // in grid?
  
    if (i + a < 0) continue;
    else if (i + a >= self->nx) continue;

    // calc contribution

    rho_a = a * (a * delta_rho + dd[0]);

    dx = delta[0] - self->spacing * a;

    for (b = -self->n_shells; b <= self->n_shells; b++) {

      // in grid and sphere?
      
      if (j + b < 0) continue;
      else if (j + b >= self->ny) continue;
      if (a*a + b*b > radius) continue;

      // calc contribution

      rho_b = b * (b * delta_rho + dd[1]);
    
      dy = delta[1] - self->spacing * b;

      for (c = -self->n_shells; c <= self->n_shells; c++) {

	// in grid and sphere?

	if (k + c < 0) continue;
	else if (k + c >= self->nz) continue;
	if (a*a + b*b + c*c > radius) continue;

	// calc contribution

	rho_c = c * (c * delta_rho + dd[2]);

	dz = delta[2] - self->spacing * c;

	index = self->nz * (self->ny * (i+a) + j+b) + k+c;

	value = exp(rho_a + rho_b + rho_c) * grad[index];

	grad_xyz[0] -= value * dx;
	grad_xyz[1] -= value * dy;
	grad_xyz[2] -= value * dz;
      }
    }
  }

  for (i = 0; i < 3; i++) grad_xyz[i] *= rho_ijk;

  return 0;
}

static int mean_position(PyGridObject *self, double *x, double width, 
			 double weight, double *rho_obs, double *pos) {

  int i, j, k, a, b, c, index, radius, coords[3];
  double rho_ijk, rho_a, rho_b, rho_c, delta_rho, w, Z, value, norm;
  vector delta, rho, y;

  index = assign_point(self, x, coords);

  w = 2. * width * width;
  Z = pow(PI*w, 1.5);

  rho_ijk = weight/Z;
  for (i = 0; i < 3; i++) {
    
    y[i] = self->origin[i] + coords[i] * self->spacing;
    delta[i] = x[i] - y[i];
    
    rho[i] = exp(-delta[i] * delta[i] / w);
    rho_ijk *= rho[i];
  }  

  delta_rho = exp(-self->spacing*self->spacing / w);

  radius = self->n_shells * self->n_shells;
  
  i = coords[0];
  j = coords[1];
  k = coords[2];

  pos[0] = pos[1] = pos[2] = pos[3] = 0.;

  norm = 1e-100;
  
  for (a = -self->n_shells; a <= self->n_shells; a++) {

    // in grid?
  
    if (i + a < 0) continue;
    else if (i + a >= self->nx) continue;

    // calc contribution

    rho_a = pow(delta_rho, a*a);
    rho_a *= exp(2 * a * self->spacing * delta[0] / w);

    for (b = -self->n_shells; b <= self->n_shells; b++) {

      // in grid and sphere?
      
      if (j + b < 0) continue;
      else if (j + b >= self->ny) continue;
      if (a*a + b*b > radius) continue;

      // calc contribution

      rho_b = pow(delta_rho, b*b);
      rho_b *= exp(2 * b * self->spacing * delta[1] / w);
    
      for (c = -self->n_shells; c <= self->n_shells; c++) {

	// in grid and sphere?

	if (k + c < 0) continue;
	else if (k + c >= self->nz) continue;
	if (a*a + b*b + c*c > radius) continue;

	// calc contribution

	rho_c = pow(delta_rho, c*c);
	rho_c *= exp(2 * c * self->spacing * delta[2] / w);

	index = self->nz * (self->ny * (i+a) + j+b) + k+c;

	value = rho_ijk * rho_a * rho_b * rho_c * rho_obs[index];
	norm += value;

	pos[0] += value * (y[0] + self->spacing * a);
	pos[1] += value * (y[1] + self->spacing * b);
	pos[2] += value * (y[2] + self->spacing * c);
      }
    }
  }

  pos[0] /= norm;
  pos[1] /= norm;
  pos[2] /= norm;
  pos[3]  = norm;

  return 0;
}

static int add(PyGridObject *self, double *x, double weight) {
  /*
    add point source to grid, equivalent to adding density with 
    zero width
  */
  int i, coords[3];

  i = assign_point(self, x, coords);

  if (in_grid(self, coords))
    self->values[i] += weight;

  return 0;
}

static int add_density_slow(PyGridObject *self, double *x, double weight) {

  int i, j, k, n, m, index, coords[3], max_index;
  double rho_ijk, delta_rho, w;
  vector delta, rho;

  index = assign_point(self, x, coords);
  w = self->width;
  w *= w;

  rho_ijk = 1.;

  for (i = 0; i < 3; i++) {
    
    delta[i] = x[i] - self->origin[i] - coords[i] * self->spacing;
    
    rho[i] = exp(- delta[i]*delta[i] / w);
    rho_ijk *= rho[i];
  }  

  max_index = self->nx * self->ny * self->nz - 1;

  for (n = 0; n < self->n_neighbors; n++) {

    m = index + self->neighbors[n];
    if ((m < 0) || (m > max_index)) continue;
    
    i = self->neighbors_ijk[3*n + 0];
    j = self->neighbors_ijk[3*n + 1];
    k = self->neighbors_ijk[3*n + 2];

    delta_rho = self->neighbor_density[n];
    delta_rho *= exp(2 * i * self->spacing * delta[0] / w);
    delta_rho *= exp(2 * j * self->spacing * delta[1] / w);
    delta_rho *= exp(2 * k * self->spacing * delta[2] / w);

    self->values[m] += weight * rho_ijk * delta_rho;
  }

  return 0;
}

static int transform_and_interpolate(PyGridObject *self, matrix R, vector t, 
				     PyGridObject *other) {
  /*
    rotate and translate grid and interpolate on other grid
  */

  int i, j, k, a, b, c, ijk[3], abc[3];
  double rho;
  vector x, y, dt;
  matrix Rt;

  matrix_transpose(Rt, R);

  vector_sub(dt, other->origin, t);

  for (i = 0; i < other->nx; i++) {
    for (j = 0; j < other->ny; j++) {
      for (k = 0; k < other->nz; k++) {

	ijk[0] = i;
	ijk[1] = j;
	ijk[2] = k;

	/* translate and rotate new grid point */

	for (a = 0; a < 3; a++)
	  x[a] = other->spacing * ijk[a];

	vector_iadd(x, dt);
	matrix_dot(y, Rt, x);

	/* coordinates on this grid */

	for (a = 0; a < 3; a++) {
	  x[a] = (y[a] - self->origin[a]) / self->spacing;
	  abc[a] = floor(x[a]);
	  x[a] -= abc[a];
	}
	
	/* trilinear interpolation */

	rho = 0.;

	for (a = 0; a <= 1; a++) {

	  abc[0] += a;
	  if ((abc[0] < 0) || (abc[0] >= self->nx)) {
	    abc[0] -= a;
	    continue;
	  }
	    
	  for (b = 0; b <= 1; b++) {

	    abc[1] += b;
	    if ((abc[1] < 0) || (abc[1] >= self->ny)) {
	      abc[1] -= b;
	      continue;
	    }
	    
	    for (c = 0; c <= 1; c++) {

	      abc[2] += c;
	      if ((abc[2] < 0) || (abc[2] >= self->nz)) {
		abc[2] -= c;
		continue;
	      }
	    
	      rho += self->values[grid_index(self, abc)] * 
		(a * x[0] + (1-a) * (1-x[0])) * 
		(b * x[1] + (1-b) * (1-x[1])) * 
		(c * x[2] + (1-c) * (1-x[2]));

	      abc[2] -= c;
	    }

	    abc[1] -= b;

	  }
	  abc[0] -= a;
	}
	      
	other->values[grid_index(other, ijk)] += rho;
      }
    }
  }

  return 0;
}

static double cc(PyGridObject *self, matrix R, vector t, PyGridObject *other) {
  /*
    correlation coefficient of rotated and translated grid
    evaluated on another grid
  */

  int i, j, k, a, b, c, n, m, ijk[3], abc[3];
  double rho, stats[5];
  vector x, y, dt;
  matrix Rt;

  matrix_transpose(Rt, R);

  vector_sub(dt, other->origin, t);

  for (i = 0; i < 5; i++) stats[i] = 0.;

  n = self->nx * self->ny * self->nz;
  m = other->nx * other->ny * other->nz;

  /* calculate first and second order statistics */

  for (i = 0; i < m; i++) {
    rho = other->values[i];
    stats[2] += rho;
    stats[3] += rho*rho;
  }
  
  stats[2]/= m;
  stats[2] = 0.;
  stats[3]-= m * stats[2] * stats[2];
  stats[3] = stats[3] < 0. ? 0 : sqrt(stats[3]);

  /* cross correlation */

  for (i = 0; i < other->nx; i++) {
    for (j = 0; j < other->ny; j++) {
      for (k = 0; k < other->nz; k++) {

	ijk[0] = i;
	ijk[1] = j;
	ijk[2] = k;

	/* translate and rotate new grid point */

	for (a = 0; a < 3; a++)
	  x[a] = other->spacing * ijk[a];

	vector_iadd(x, dt);
	matrix_dot(y, Rt, x);

	/* coordinates on this grid */

	for (a = 0; a < 3; a++) {
	  x[a] = (y[a] - self->origin[a]) / self->spacing;
	  abc[a] = floor(x[a]);
	  x[a] -= abc[a];
	}
	
	/* trilinear interpolation */

	rho = 0.;

	for (a = 0; a <= 1; a++) {

	  abc[0] += a;
	  if ((abc[0] < 0) || (abc[0] >= self->nx)) {
	    abc[0] -= a;
	    continue;
	  }
	    
	  for (b = 0; b <= 1; b++) {

	    abc[1] += b;
	    if ((abc[1] < 0) || (abc[1] >= self->ny)) {
	      abc[1] -= b;
	      continue;
	    }
	    
	    for (c = 0; c <= 1; c++) {

	      abc[2] += c;
	      if ((abc[2] < 0) || (abc[2] >= self->nz)) {
		abc[2] -= c;
		continue;
	      }
	    
	      rho += self->values[grid_index(self, abc)] * 
		(a * x[0] + (1-a) * (1-x[0])) * 
		(b * x[1] + (1-b) * (1-x[1])) * 
		(c * x[2] + (1-c) * (1-x[2]));

	      abc[2] -= c;
	    }

	    abc[1] -= b;

	  }
	  abc[0] -= a;
	}

	stats[0] += rho;
	stats[1] += rho * rho;
	stats[4] += other->values[grid_index(other, ijk)] * rho;
      }
    }
  }

  stats[0]/= m;
  stats[0] = 0.;
  stats[1]-= m * stats[0] * stats[0];
  stats[1] = stats[1] < 0. ? 0 : sqrt(stats[1]);

  stats[1]+= 1e-300;
  stats[3]+= 1e-300;

  if (!NORMALIZE_CC) {
    stats[1] = 1.;
    stats[3] = 1.;
  }

  return (stats[4] - m * stats[0] * stats[2]) / stats[1] / stats[3];
}

static double cc_translation(PyGridObject *self,vector t,PyGridObject *other) {
  /*
    correlation coefficient of a translated grid evaluated on another grid
  */

  int i, j, k, a, b, c, n, m, ijk[3], abc[3];
  double rho, stats[5];
  vector x, dt;

  vector_sub(dt, other->origin, t);
  vector_sub(dt, dt, self->origin);

  for (i = 0; i < 5; i++) stats[i] = 0.;

  n = self->nx * self->ny * self->nz;
  m = other->nx * other->ny * other->nz;

  /* calculate first and second order statistics */

  for (i = 0; i < m; i++) {
    rho = other->values[i];
    stats[2] += rho;
    stats[3] += rho*rho;
  }

  stats[2] /= m;
  stats[3] -= m * stats[2] * stats[2];
  stats[3] = stats[3] < 0. ? 0 : sqrt(stats[3]);

  /* cross correlation */

  for (i = 0; i < other->nx; i++) {

    ijk[0] = i;
    x[0] = (i * other->spacing + dt[0]) / self->spacing;
    abc[0] = floor(x[0]);
    x[0]-= abc[0];

    for (j = 0; j < other->ny; j++) {

      ijk[1] = j;
      x[1] = (j * other->spacing + dt[1]) / self->spacing;
      abc[1] = floor(x[1]);
      x[1]-= abc[1];

      for (k = 0; k < other->nz; k++) {

	ijk[2] = k;
	x[2] = (k * other->spacing + dt[2]) / self->spacing;
	abc[2] = floor(x[2]);
	x[2]-= abc[2];

	/* trilinear interpolation */

	rho = 0.;

	for (a = 0; a <= 1; a++) {

	  abc[0] += a;
	  if ((abc[0] < 0) || (abc[0] >= self->nx)) {
	    abc[0] -= a;
	    continue;
	  }
	    
	  for (b = 0; b <= 1; b++) {

	    abc[1] += b;
	    if ((abc[1] < 0) || (abc[1] >= self->ny)) {
	      abc[1] -= b;
	      continue;
	    }
	    
	    for (c = 0; c <= 1; c++) {

	      abc[2] += c;
	      if ((abc[2] < 0) || (abc[2] >= self->nz)) {
		abc[2] -= c;
		continue;
	      }
	    
	      rho += self->values[grid_index(self, abc)] * 
		(a * x[0] + (1-a) * (1-x[0])) * 
		(b * x[1] + (1-b) * (1-x[1])) * 
		(c * x[2] + (1-c) * (1-x[2]));

	      abc[2] -= c;
	    }

	    abc[1] -= b;

	  }
	  abc[0] -= a;
	}

	stats[0] += rho;
	stats[1] += rho * rho;
	stats[4] += other->values[grid_index(other, ijk)] * rho;
      }
    }
  }

  stats[0]/= m;
  stats[0] = 0.;
  stats[1]-= m * stats[0] * stats[0];
  stats[1] = stats[1] < 0. ? 0 : sqrt(stats[1]);
  stats[1]+= 1e-300;
  stats[3]+= 1e-300;

  if (!NORMALIZE_CC) {
    stats[1] = 1.;
    stats[3] = 1.;
  }

  return (stats[4] - m * stats[0] * stats[2]) / stats[1] / stats[3];
}

static double* cc_tensor(PyGridObject *self, PyGridObject *other, int n) {
 
  int i, j, k, a, b, c, m, N, index, coord[3];
  double *cc, *stats, x, y, s[6];

  m = 2*n + 1;
  N = m*m*m;

  if (!(stats = MALLOC(6*N, double)))
    return NULL;
  if (!(cc = MALLOC(N, double)))
    return NULL;

  for (i = 0; i < 6*N; i++)
    stats[i] = 0.;

  for (i = 0; i < N; i++)
    cc[i] = 0.;

  // calculate coorelations for various shifts on the grid

  for (i = 0; i < self->nx; i++) {
    for (j = 0; j < self->ny; j++) {
      for (k = 0; k < self->nz; k++) {

	coord[0] = i;
	coord[1] = j;
	coord[2] = k;
	
	x = self->values[grid_index(self, coord)];

	if (x < 1e-100) continue;

	for (a = -n; a <= n; a++) {

	  coord[0] += a;
	  if ((coord[0] < 0) || (coord[0] >= other->nx)) {
	    coord[0] -= a;
	    continue;
	  }

	  for (b = -n; b <= n; b++) {

	    coord[1] += b;
	    if ((coord[1] < 0) || (coord[1] >= other->ny)) {
	      coord[1] -= b;
	      continue;
	    }

	    for (c = -n; c <= n; c++) {

	      coord[2] += c;
	      if ((coord[2] < 0) || (coord[2] >= other->nz)) {
		coord[2] -= c;
		continue;
	      }
	      
	      y = other->values[grid_index(other, coord)];
	      
	      index = (((a + n) * m + (b + n)) * m) + c + n;
	      index*= 6;

	      stats[index + 0] += x;
	      stats[index + 1] += x*x;
	      stats[index + 2] += y;
	      stats[index + 3] += y*y;
	      stats[index + 4] += x*y;
	      stats[index + 5] += 1;

	      coord[2] -= c;
	    }
	  
	  coord[1] -= b;
	  
	  }
	
	  coord[0] -= a;
	}
      }
    }
  }

  for (i = 0; i < m; i++)
    for (j = 0; j < m; j++)
      for (k = 0; k < m; k++) {

	index = (i * m + j) * m + k;

	// number of counts

	s[5] = stats[6 * index + 5];

	if (s[5] < 2)
	  continue;

	// means

	s[0] = stats[6 * index + 0] / s[5];
	s[2] = stats[6 * index + 2] / s[5];

	s[0] = 0.;
	s[2] = 0.;

	// standard deviations

	s[1] = stats[6 * index + 1] - s[5] * s[0]*s[0];
	s[1] = s[1] < 0. ? 1e-300 : sqrt(s[1]);

	s[3] = stats[6 * index + 3] - s[5] * s[2]*s[2];
	s[3] = s[3] < 0. ? 1e-300 : sqrt(s[3]);

	if (!NORMALIZE_CC) {
	  s[1] = 1.;
	  s[3] = 1.;
	}

	// correlation

	s[4] = stats[6 * index + 4] - s[5] * s[0]*s[2];
	cc[index] = s[4] / s[1] / s[3];
      }

  free(stats);

  return cc;
}

static double evaluate(PyGridObject *self, vector y) {
  /*
    probability of a point in 3D space
  */

  int i, j, k, coords[3];
  double rho = 0.;
  vector x;

  /* get nearest neighbors */

  for (i = 0; i < 3; i++) {
    x[i] = (y[i] - self->origin[i]) / self->spacing;
    coords[i] = floor(x[i]);
    x[i]-= coords[i];
  }
  
  for (i = 0; i <= 1; i++) {
    
    coords[0] += i;
    if ((coords[0] < 0) || (coords[0] >= self->nx)) {
      coords[0] -= i;
      continue;
    }
	    
    for (j = 0; j <= 1; j++) {
      
      coords[1] += j;
      if ((coords[1] < 0) || (coords[1] >= self->ny)) {
	coords[1] -= j;
	continue;
      }
      
      for (k = 0; k <= 1; k++) {
	
	coords[2] += k;
	if ((coords[2] < 0) || (coords[2] >= self->nz)) {
	  coords[2] -= k;
	  continue;
	}	
	rho += self->values[grid_index(self, coords)] * 
	  (i * x[0] + (1-i) * (1-x[0])) * 
	  (j * x[1] + (1-j) * (1-x[1])) * 
	  (k * x[2] + (1-k) * (1-x[2]));

	coords[2] -= k;
      }      
      coords[1] -= j;
    }
    coords[0] -= i;
  }

  return rho;
}

static double* em_statistics(PyGridObject *self, double *X, double *rho_obs, int N, 
			     double rho_cutoff) {

  int i, j, k, n, d, ijk[3];
  double *F, *G, *H, *weights, *stats, rho_ijk, norm, max_weight, w, Z, dx, dy, dz, weight;
  vector delta, rho;

  /* create helper arrays */

  if (!(F = MALLOC(self->nx * N, double)))
    return NULL;
  if (!(G = MALLOC(self->ny * N, double)))
    return NULL;
  if (!(H = MALLOC(self->nz * N, double)))
    return NULL;
  if (!(stats = MALLOC(7 * N, double)))
    return NULL;
  if (!(weights = MALLOC(N, double)))
    return NULL;
  
  for (i = 0; i < self->nx * N; i++) F[i] = 0.;
  for (j = 0; j < self->ny * N; j++) G[j] = 0.;
  for (k = 0; k < self->nz * N; k++) H[k] = 0.;
  for (n = 0; n < 7 * N;        n++) stats[n] = 0.;

  /* fill arrays */

  w = self->width;
  w*= 2 * w;  
  Z = sqrt(PI*w);

  for (n = 0; n < N; n++) {

    assign_point(self, &X[3*n], ijk);

    for (i = 0; i < 3; i++) {
      delta[i] = X[3*n + i] - self->origin[i] - ijk[i] * self->spacing;
      rho[i]   = - delta[i] * delta[i] / w;
    }

    for (i = 0; i < self->nx; i++) {
      dx = (ijk[0] - i) * self->spacing;
      F[i*N + n] = - dx * dx / w - rho[0] + 2 * delta[0] * dx / w;
    }

    for (j = 0; j < self->ny; j++) {
      dy = (ijk[1] - j) * self->spacing;
      G[j*N + n] = - dy * dy / w - rho[1] + 2 * delta[1] * dy / w;
    }

    for (k = 0; k < self->nz; k++) {
      dz = (ijk[2] - k) * self->spacing;
      H[k*N + n] = - dz * dz / w - rho[2] + 2 * delta[2] * dz / w;
    }
  }

  /* calculate weights */

  for (i = 0; i < self->nx; i++) {

    ijk[0] = i;
    delta[0] = i * self->spacing + self->origin[0];

    for (j = 0; j < self->ny; j++) {

      ijk[1] = j;
      delta[1] = j * self->spacing + self->origin[1];

      for (k = 0; k < self->nz; k++) {

	ijk[2] = k;

	rho_ijk = rho_obs[grid_index(self, ijk)];
	
	if (rho_ijk < rho_cutoff) continue;

	delta[2] = k * self->spacing + self->origin[2];

	/* calculate weights */

	max_weight = -1e308;

	for (n = 0; n < N; n++) {
	  
	  weights[n] = WEIGHT_CUTOFF;
	  weight = F[i*N + n] + G[j*N + n] + H[k*N + n];
	  
	  if ((weight - max_weight) < WEIGHT_CUTOFF) continue;
	  
	  max_weight = weight > max_weight ? weight : max_weight;

	  weights[n] = weight;
	}

	/* normalize weights */

	norm = 0.;

	for (n = 0; n < N; n++) {
	  
	  weight = weights[n] - max_weight;
	  if (weight < WEIGHT_CUTOFF)
	    weight = 0.;
	  else
	    weight = exp(weight);
	  norm += weight;
	  weights[n] = weight;
	}

	/* update statistics */

	for (n = 0; n < N; n++) {

	  weight = rho_ijk * weights[n] / norm;
	  if (weight < 1e-100) continue;
	  stats[7*n + 0] += weight;

	  for (d = 0; d < 3; d++) {	    
	    stats[7*n + 1 + d] += delta[d] * weight;
	    stats[7*n + 4 + d] += delta[d] * delta[d] * weight;
	  }
	}

      }
    }
  }

  free(F);
  free(G);
  free(H);
  free(weights);

  return stats;
}

static double* mean_structure(PyGridObject *self, double *X, int n_atoms) {

  int i, j, k, n, coords[3];
  double *x, *y, *z, *d, *D, *s, d_min, rho, norm;

  if (!(x = MALLOC(n_atoms, double))) return NULL;
  if (!(y = MALLOC(n_atoms, double))) return NULL;
  if (!(z = MALLOC(n_atoms, double))) return NULL;
  if (!(d = MALLOC(n_atoms, double))) return NULL;
  if (!(D = MALLOC(n_atoms, double))) return NULL;
  if (!(s = MALLOC(4*n_atoms, double))) return NULL;

  /* store distance to origin */

  for (i = 0; i < n_atoms; i++) {
    
    x[i] = X[3*i + 0] - self->origin[0];
    y[i] = X[3*i + 1] - self->origin[1];
    z[i] = X[3*i + 2] - self->origin[2];

    D[i] = x[i] * x[i] + y[i] * y[i] + z[i] * z[i];

  }

  /* loop through grid */

  for (i = 0; i < self->nx; i++) {

    coords[0] = i;

    for (j = 0; j < self->ny; j++) {

      coords[1] = j;

      for (k = 0; k < self->nz; k++) {

	coords[2] = k;

	d_min = -1e300;

	for (n = 0; n < n_atoms; n++) {

	  d[n] = D[n] + 2 * self->spacing * (i * x[n] + j * y[n] + k * z[n]);
	  d_min = d_min < d[n] ? d_min : d[n];
	}

	norm = 0.;

	for (n = 0; n < n_atoms; n++) {
	  
	  d[n] = exp(-0.5 * (d[n] - d_min) / self->width / self->width);
	  norm += d[n];

	}

	rho = self->values[grid_index(self, coords)];
	
	for (n = 0; n < n_atoms; n++) {
	  s[4 * n + 0] += rho * d[n] * i / norm;
	  s[4 * n + 1] += rho * d[n] * j / norm;
	  s[4 * n + 2] += rho * d[n] * k / norm;
	  s[4 * n + 3] += rho * d[n] / norm;
	}
      }
    }
  }  

  free(d);
  free(D);
  free(x);
  free(y);
  free(z);

  return s;
} 

static PyObject* py_coordinates(PyGridObject *self, PyObject *args) {
  
  int i, grid_index, grid_coords[3];
  PyObject *t;
    
  if (!PyArg_ParseTuple(args, "i", &grid_index)) 
    RAISE(PyExc_TypeError, "int expected.", NULL);
  
  grid_coordinates(self, grid_index, grid_coords);

  t = PyTuple_New(3);
  for (i = 0; i < 3; i++)
    PyTuple_SET_ITEM(t, i, Py_BuildValue("i", grid_coords[i]));

  return t;
}

static PyObject* py_index(PyGridObject *self, PyObject *args) {

  int i, grid_coords[3];

  PyObject *t;

  if (!PyArg_ParseTuple(args, "O!", &PyTuple_Type, &t)) 
    RAISE(PyExc_TypeError, "tuple expected.", NULL);
  
  if (3 != PyTuple_Size(t))
    RAISE(PyExc_ValueError, "wrong size", NULL);
    
  for (i = 0; i < 3; i++)
    grid_coords[i] = (int) PyInt_AsLong(PyTuple_GetItem(t, i));

  i = grid_index(self, grid_coords);

  return Py_BuildValue("i", i);
}

static PyObject* py_assign(PyGridObject *self, PyObject *args) {

  int i, n, *indices, coords[3];
  vector *x;
  PyArrayObject *X;
  PyObject *a;

  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &X)) 
    RAISE(PyExc_TypeError, "numeric array expected", NULL);
  
  x = (vector*) X->data;
  n = X->dimensions[0];

  if (!(indices = MALLOC(n, int)))
    RAISE(PyExc_MemoryError, "malloc failed", NULL);

  for (i = 0; i < n; i++)
    indices[i] = assign_point(self, x[i], coords);

  a = PyArray_SimpleNewFromData(1,&n, PyArray_INT, (char*)indices);
  PyArray_FLAGS(a) |= NPY_OWNDATA;


  return a;
}

static PyObject* py_add_density(PyGridObject *self, PyObject *args) {

  int i, n;
  vector *x;
  PyArrayObject *X = NULL;

  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &X)) 
    RAISE(PyExc_TypeError, "numeric array expected", NULL);
  
  Py_INCREF(X);

  x = (vector*) X->data;
  n = X->dimensions[0];

  for (i = 0; i < n; i++) add_density(self, x[i], self->width, 1.);

  RETURN_PY_NONE;
}

static PyObject* py_eval(PyGridObject *self, PyObject *args) {

  int i, n;
  double rho=0.;
  vector *x;
  PyArrayObject *X;

  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &X)) 
    RAISE(PyExc_TypeError, "numeric array expected", NULL);
  
  x = (vector*) X->data;
  n = X->dimensions[0];

  for (i = 0; i < n; i++) rho += evaluate(self, x[i]);

  Py_BuildValue("d", rho);
}

static PyObject* py_update_gradient(PyGridObject *self, PyObject *args) {

  int i, n, dims[2];
  double *grad, grad_i[3], factor;
  vector *x;
  PyArrayObject *X, *g;
  PyObject *G;

  if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &X, &PyArray_Type, &g)) 
    RAISE(PyExc_TypeError, "numeric array expected", NULL);
  
  x = (vector*) X->data;
  n = X->dimensions[0];

  if (!(grad = MALLOC(3*n, double)))
    RAISE(PyExc_MemoryError, "malloc failed", NULL);

  factor = self->width;
  factor*= factor;
  factor*= factor * self->width;

  factor = 1. / factor;

  for (i = 0; i < n; i++) {
    update_gradient(self, x[i], self->width, 1., (double*) (g->data), grad_i);
    grad[3*i+0] = grad_i[0] * factor;
    grad[3*i+1] = grad_i[1] * factor;
    grad[3*i+2] = grad_i[2] * factor;
  }

  dims[0] = n;
  dims[1] = 3;

  G = PyArray_SimpleNewFromData(2, dims, PyArray_DOUBLE, (char*) grad);
  PyArray_FLAGS(G) |= NPY_OWNDATA;

  return G;
}

static PyObject* py_mean_structure(PyGridObject *self, PyObject *args) {

  int i, n, dims[2];
  double *y, xyz[4];
  vector *x;
  PyArrayObject *X, *g;
  PyObject *Y;

  if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &X, &PyArray_Type, &g)) 
    RAISE(PyExc_TypeError, "numeric array expected", NULL);
  
  x = (vector*) X->data;
  n = X->dimensions[0];

  if (!(y = MALLOC(4*n, double)))
    RAISE(PyExc_MemoryError, "malloc failed", NULL);

  for (i = 0; i < n; i++) {
    mean_position(self, x[i], self->width, 1., (double*) (g->data), xyz);
    y[4*i+0] = xyz[0];
    y[4*i+1] = xyz[1];
    y[4*i+2] = xyz[2];
    y[4*i+3] = xyz[3];
  }

  dims[0] = n;
  dims[1] = 4;
  
  Y = PyArray_SimpleNewFromData(2, dims, PyArray_DOUBLE, (char*) y);
  PyArray_FLAGS(Y) |= NPY_OWNDATA;

  return Y;
}

static PyObject* py_mean_structure2(PyGridObject *self, PyObject *args) {

  int i, n, dims[2];
  double *y, *x;
  PyArrayObject *X;
  PyObject *Y;

  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &X)) 
    RAISE(PyExc_TypeError, "numeric array expected", NULL);
  
  x = (double*) X->data;
  n = X->dimensions[0];

  if (!(y = mean_structure(self, x, n)))
    RAISE(PyExc_StandardError, "mean_structure failed", NULL);

  dims[0] = n;
  dims[1] = 4;

  Y = PyArray_SimpleNewFromData(2, dims, PyArray_DOUBLE, (char*) y);
  PyArray_FLAGS(Y) |= NPY_OWNDATA;


  return Y;
}

static PyObject* py_add_density_full(PyGridObject *self, PyObject *args) {
  /*
    set individual widths and weights
  */
  int i, n;
  vector *x;
  PyArrayObject *X, *sigma, *weights;

  if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &X, &PyArray_Type, 
			&sigma, &PyArray_Type, &weights)) 
    RAISE(PyExc_TypeError, "three numeric arrays expected", NULL);
  
  x = (vector*) X->data;
  n = X->dimensions[0];

  if (sigma->nd != 1)
    RAISE(PyExc_StandardError, "sigma must be rank 1", NULL);
  if (weights->nd != 1)
    RAISE(PyExc_StandardError, "weights must be rank 1", NULL);
    
  if (sigma->dimensions[0] != n)
    RAISE(PyExc_StandardError, "sigma must have len(X)", NULL);
  if (weights->dimensions[0] != n)
    RAISE(PyExc_StandardError, "weights must have len(X)", NULL);

  for (i = 0; i < n; i++) 
    add_density(self, x[i], *(double*) (sigma->data + i * sigma->strides[0]),
		*(double*) (weights->data + i * weights->strides[0]));

  RETURN_PY_NONE;
}

static PyObject* py_add_points(PyGridObject *self, PyObject *args) {

  int i, n;
  vector *x;
  PyArrayObject *X;

  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &X)) 
    RAISE(PyExc_TypeError, "numeric array expected", NULL);
  
  x = (vector*) X->data;
  n = X->dimensions[0];

  for (i = 0; i < n; i++) add(self, x[i], 1.);

  RETURN_PY_NONE;
}


static PyObject* py_add_weighted_points(PyGridObject *self, PyObject *args) {

  int i, n;
  vector *x;
  
  PyArrayObject *X, *w;

  if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &X, &PyArray_Type, &w)) 
    RAISE(PyExc_TypeError, "two numeric arrays expected", NULL);
  
  x = (vector*) X->data;
  n = X->dimensions[0];

  for (i = 0; i < n; i++) 
    add(self, x[i], *(double*) (w->data + i * w->strides[0]));

  RETURN_PY_NONE;
}


static PyObject* py_set_density(PyGridObject *self, PyObject *args) {
  /*
    set density to constant value
  */
  double value;

  if (!PyArg_ParseTuple(args, "d", &value)) 
    RAISE(PyExc_TypeError, "double expected", NULL);
  
  if (set_density(self, value))
    RAISE(PyExc_StandardError, "error in set_density", NULL);
  
  RETURN_PY_NONE;
}

static PyObject* py_set_rho(PyGridObject *self, PyObject *args) {
  /*
    set density from array
  */
  int i,j,k, grid_index;
  double elem;
  PyArrayObject *rho;
 
  
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &rho)) 
    RAISE(PyExc_TypeError, "numeric array expected", NULL);

  for (i = 0; i < self->nx; i++){
    for (j = 0; j < self->ny; j++){
      for (k =0; k < self->nz; k++){

        elem = *((double *)PyArray_GETPTR3(rho, i, j, k));
        grid_index = ((i * self->ny + j) * self->nz) + k;
        self->values[grid_index] = elem; 
        
      }     
    }
  }
  RETURN_PY_NONE;
}

static PyObject* py_add_density_slow(PyGridObject *self, PyObject *args) {

  int i, n;
  vector *x;
  PyArrayObject *X;

  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &X)) 
    RAISE(PyExc_TypeError, "numeric array expected", NULL);
  
  x = (vector*) X->data;
  n = X->dimensions[0];

  for (i = 0; i < n; i++) add_density_slow(self, x[i], 1.);

  RETURN_PY_NONE;
}

static PyObject* py_calc_density(PyGridObject *self, PyObject *args) {

  int i, n;
  vector *x;
  PyArrayObject *X;

  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &X)) 
    RAISE(PyExc_TypeError, "numeric array expected", NULL);
  
  x = (vector*) X->data;
  n = X->dimensions[0];

  if (set_density(self, 0.))
    RAISE(PyExc_StandardError, "error in set_density", NULL);

  for (i = 0; i < n; i++) add_density(self, x[i], self->width, 1.);

  RETURN_PY_NONE;
}

static PyObject* py_mask(PyGridObject *self, PyObject *args) {

  int i, n, m;
  double a, b;
  vector *x;
  PyArrayObject *X;

  if (!PyArg_ParseTuple(args, "O!ddi", &PyArray_Type, &X, &a, &b, &m)) 
    RAISE(PyExc_TypeError, "numeric array, two doubles and one int expected", 
	  NULL);
  
  x = (vector*) X->data;
  n = X->dimensions[0];

  if (set_density(self, 0.))
    RAISE(PyExc_StandardError, "error in set_density", NULL);

  for (i = 0; i < n; i++) add_mask(self, x[i], a, b, m);

  RETURN_PY_NONE;
}

static PyObject* py_calc_density_slow(PyGridObject *self, PyObject *args) {

  int i, n;
  vector *x;
  PyArrayObject *X;

  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &X)) 
    RAISE(PyExc_TypeError, "numeric array expected", NULL);
  
  x = (vector*) X->data;
  n = X->dimensions[0];

  if (set_density(self, 0.))
    RAISE(PyExc_StandardError, "error in set_density", NULL);

  for (i = 0; i < n; i++) add_density_slow(self, x[i], 1.);

  RETURN_PY_NONE;
}

static PyObject* py_setup_neighbors(PyGridObject *self, PyObject *args) {

  if (!PyArg_ParseTuple(args, "")) 
    RAISE(PyExc_TypeError, "no argument expected", NULL);
  
  if (setup_neighbors(self))
    RAISE(PyExc_StandardError, "error in setup_neighbors", NULL);

  RETURN_PY_NONE;
}

static PyObject* py_convolve(PyGridObject *self, PyObject *args) {

  double cutoff;
  PyArrayObject *rho;

  if (!PyArg_ParseTuple(args, "O!d", &PyArray_Type, &rho, &cutoff)) 
    RAISE(PyExc_TypeError, "double array expected", NULL);

  if (convolve(self, (double*) rho->data, cutoff))
    RAISE(PyExc_StandardError, "error in convolve", NULL);

  RETURN_PY_NONE;
}

static PyObject* py_transform_and_interpolate(PyGridObject *self,PyObject *args){

  int i,j;

  PyArrayObject *R, *t;
  PyGridObject *other;

  vector vt;
  matrix mR;

  if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &R, &PyArray_Type, &t,
			&PyGrid_Type, &other)) 
    RAISE(PyExc_TypeError, "two double array and grid expected", NULL);

  /* check dimensions of arrays */

  if (t->nd != 1)
    RAISE(PyExc_StandardError, "wrong dimension for t", NULL);

  if (t->dimensions[0] != 3)
    RAISE(PyExc_StandardError, "wrong dimension for t", NULL);
  
  if (R->nd != 2)
    RAISE(PyExc_StandardError, "wrong dimension for R", NULL);

  if ((R->dimensions[0] != 3) || (R->dimensions[1] != 3))
    RAISE(PyExc_StandardError, "wrong dimension for R", NULL);
  
  /* read out values */

  for (i = 0; i < 3; i++) {
    vt[i] = *(double*) (t->data + i * t->strides[0]);
    for (j = 0; j < 3; j++) 
      mR[i][j] = *(double*) (R->data + i * R->strides[0] + j * R->strides[1]);
  }
  
  /* rotate and interpolate */

  if (transform_and_interpolate(self, mR, vt, other))
    RAISE(PyExc_StandardError, "error in transform_and_interpolate", NULL);

  RETURN_PY_NONE;
}

static PyObject* py_cc(PyGridObject *self,PyObject *args){

  int i,j;
  double val;

  PyArrayObject *R, *t;
  PyGridObject *other;

  vector vt;
  matrix mR;

  if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &R, &PyArray_Type, &t,
			&PyGrid_Type, &other)) 
    RAISE(PyExc_TypeError, "two double array and grid expected", NULL);

  /* check dimensions of arrays */

  if (t->nd != 1)
    RAISE(PyExc_StandardError, "wrong dimension for t", NULL);

  if (t->dimensions[0] != 3)
    RAISE(PyExc_StandardError, "wrong dimension for t", NULL);
  
  if (R->nd != 2)
    RAISE(PyExc_StandardError, "wrong dimension for R", NULL);

  if ((R->dimensions[0] != 3) || (R->dimensions[1] != 3))
    RAISE(PyExc_StandardError, "wrong dimension for R", NULL);
  
  /* read out values */

  for (i = 0; i < 3; i++) {
    vt[i] = *(double*) (t->data + i * t->strides[0]);
    for (j = 0; j < 3; j++) 
      mR[i][j] = *(double*) (R->data + i * R->strides[0] + j * R->strides[1]);
  }
 /* rotate and interpolate */

  val = cc(self, mR, vt, other);

  return Py_BuildValue("d", val);
}

static PyObject* py_cc_translation(PyGridObject *self,PyObject *args){

  int i,j;
  double val;

  PyArrayObject *t;
  PyGridObject *other;

  vector vt;

  if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type,&t,&PyGrid_Type,&other)) 
    RAISE(PyExc_TypeError, "double array and grid expected", NULL);

  /* check dimensions of arrays */

  if (t->nd != 1)
    RAISE(PyExc_StandardError, "wrong dimension for t", NULL);

  if (t->dimensions[0] != 3)
    RAISE(PyExc_StandardError, "wrong dimension for t", NULL);
  
  /* read out values */

  for (i = 0; i < 3; i++) 
    vt[i] = *(double*) (t->data + i * t->strides[0]);
  
  /* rotate and interpolate */

  val = cc_translation(self, vt, other);

  return Py_BuildValue("d", val);
}

static PyObject* py_cc_tensor(PyGridObject *self, PyObject *args){

  int n, dims[3];
  double *cc;

  PyGridObject *other;
  PyObject *CC;

  if (!PyArg_ParseTuple(args, "O!i", &PyGrid_Type,&other,&n)) 
    RAISE(PyExc_TypeError, "grid and integer expected", NULL);

  if (!(cc = cc_tensor(self, other, n)))
    RAISE(PyExc_StandardError, "error in cc_tensor", NULL);    

  dims[0] = dims[1] = dims[2] = 2 * n + 1;

  CC = PyArray_SimpleNewFromData(3, dims, PyArray_DOUBLE, (char*) cc);
  PyArray_FLAGS(CC) |= NPY_OWNDATA;

  return CC;
}

static PyObject* project1D(PyGridObject *self, PyObject *args) {
  /*
    project the density onto a given axis (0,1,2) for (x,y,z)
    by summing over the other axes
  */

  int i,j,k, axis, coord[3];
  double *rho;
  npy_intp dimensions;

  PyObject *Rho;

  if (!PyArg_ParseTuple(args, "i", &axis)) 
    RAISE(PyExc_TypeError, "int expected", NULL);

  if (axis < 0 || axis > 2)
    RAISE(PyExc_ValueError, "axis must be in (0,1,2)", NULL);

  if (axis == 0) {

    if (!(rho = MALLOC(self->nx, double))) 
      RAISE(PyExc_MemoryError, "malloc failed", NULL);
    for (i = 0; i < self->nx; i++) rho[i] = 0.;

    for (i = 0; i < self->nx; i++) {
      coord[0] = i;
      for (j = 0; j < self->ny; j++) {
        coord[1] = j;
        for (k = 0; k < self->nz; k++) {
          coord[2] = k;
          rho[i] += self->values[grid_index(self, coord)];
        }
      }
    }
    npy_intp dimensions  = self->nx;

    Rho = PyArray_SimpleNewFromData(1, &dimensions, PyArray_DOUBLE,
                                    (char*) rho);
  }

  else if (axis == 1) {
  
    if (!(rho = MALLOC(self->ny, double))) 
      RAISE(PyExc_MemoryError, "malloc failed", NULL);

    for (i = 0; i < self->ny; i++) rho[i] = 0.;

    for (i = 0; i < self->nx; i++) {
      coord[0] = i;
      for (j = 0; j < self->ny; j++) {
	coord[1] = j;
	for (k = 0; k < self->nz; k++) {
	  coord[2] = k;
	  rho[j] += self->values[grid_index(self, coord)];
	}
      }
    }
    npy_intp dimensions  = self->ny;
    Rho = PyArray_SimpleNewFromData(1, &dimensions, PyArray_DOUBLE,
                                    (char*) rho);
  }

  else if (axis == 2) {

    if (!(rho = MALLOC(self->nz, double))) 
      RAISE(PyExc_MemoryError, "malloc failed", NULL);

    for (i = 0; i < self->nz; i++) rho[i] = 0.;

    for (i = 0; i < self->nx; i++) {
      coord[0] = i;
      for (j = 0; j < self->ny; j++) {
	coord[1] = j;
	for (k = 0; k < self->nz; k++) {
	  coord[2] = k;
	  rho[k] += self->values[grid_index(self, coord)];
	}
      }
    }

    npy_intp dimensions  = self->nz;
    Rho = PyArray_SimpleNewFromData(1, &dimensions, PyArray_DOUBLE,
                                    (char*) rho);
  }
  PyArray_FLAGS(Rho) |= NPY_OWNDATA;

  return Rho;
}

static PyObject* project2D(PyGridObject *self, PyObject *args) {
  /*
    sum density along given axis
  */

  int i,j,k, axis, coord[3], dims[2];
  double *rho;

  PyObject *Rho;

  if (!PyArg_ParseTuple(args, "i", &axis)) 
    RAISE(PyExc_TypeError, "int expected", NULL);

  if (axis == 0) {
    dims[0] = self->ny; dims[1] = self->nz;
  }
  else if (axis == 1) {
    dims[0] = self->nx; dims[1] = self->nz;
  }
  else if (axis == 2) {
    dims[0] = self->nx; dims[1] = self->ny;
  }

  if (!(rho = MALLOC(dims[0] * dims[1], double))) 
    RAISE(PyExc_MemoryError, "malloc failed", NULL);
  
  for (i = 0; i < dims[0] * dims[1]; i++) rho[i] = 0.;

  if (axis == 0) {

    for (i = 0; i < self->nx; i++) {
      coord[0] = i;
      for (j = 0; j < self->ny; j++) {
	coord[1] = j;
	for (k = 0; k < self->nz; k++) {
	  coord[2] = k;
	  rho[j*self->nz + k] += self->values[grid_index(self, coord)];
	}
      }
    }
  }

  else if (axis == 1) {

    for (i = 0; i < self->nx; i++) {
      coord[0] = i;
      for (j = 0; j < self->ny; j++) {
	coord[1] = j;
	for (k = 0; k < self->nz; k++) {
	  coord[2] = k;
	  rho[i*self->nz + k] += self->values[grid_index(self, coord)];
	}
      }
    }
  }

  else if (axis == 2) {

    for (i = 0; i < self->nx; i++) {
      coord[0] = i;
      for (j = 0; j < self->ny; j++) {
	coord[1] = j;
	for (k = 0; k < self->nz; k++) {
	  coord[2] = k;
	  rho[i*self->ny + j] += self->values[grid_index(self, coord)];
	}
      }
    }
  }

  Rho = PyArray_SimpleNewFromData(2, dims, PyArray_DOUBLE, (char*) rho);
  free(rho);

  return Rho;
}

static PyObject* apply_laplacian(PyGridObject *self, PyObject *args) {
  /*
    apply Laplacian filter to the density and return the resulting
    values
  */

  int i, j, k, coord[3], dims[2], index;
  double *values;

  PyObject *rho;

  if (!PyArg_ParseTuple(args, "")) 
    RAISE(PyExc_TypeError, "no arg expected", NULL);

  dims[0] = self->nx * self->ny * self->nz;

  if (!(values = MALLOC(dims[0], double))) 
    RAISE(PyExc_MemoryError, "malloc failed", NULL);
  
  for (i = 0; i < dims[0]; i++) values[i] = 0.;

  for (i = 0; i < self->nx; i++) {
    coord[0] = i;
    for (j = 0; j < self->ny; j++) {
      coord[1] = j;
      for (k = 0; k < self->nz; k++) {
	coord[2] = k;
	index = grid_index(self, coord);
	values[index] -= 6 * self->values[index];
	
	/* x direction */

	if (i >= 1) {
	  coord[0] -= 1;
	  values[index] += self->values[grid_index(self, coord)];
	  coord[0] += 1;
	}

	if (i+1 < self->nx) {
	  coord[0] += 1;
	  values[index] += self->values[grid_index(self, coord)];
	  coord[0] -= 1;
	}

	/* y direction */

	if (j >= 1) {
	  coord[1] -= 1;
	  values[index] += self->values[grid_index(self, coord)];
	  coord[1] += 1;
	}

	if (j+1 < self->ny) {
	  coord[1] += 1;
	  values[index] += self->values[grid_index(self, coord)];
	  coord[1] -= 1;
	}

	/* z direction */

	if (k >= 1) {
	  coord[2] -= 1;
	  values[index] += self->values[grid_index(self, coord)];
	  coord[2] += 1;
	}

	if (k+1 < self->nz) {
	  coord[2] += 1;
	  values[index] += self->values[grid_index(self, coord)];
	  coord[2] -= 1;
	}

      }
    }
  }

  rho = PyArray_SimpleNewFromData(1, dims, PyArray_DOUBLE, (char*) values);
  free(values);

  return rho;
}

static PyObject* py_em_stats(PyGridObject *self, PyObject *args) {

  int dims[2];
  double *stats, cutoff;
  PyArrayObject *X, *rho;
  PyObject *Stats;

  if (!PyArg_ParseTuple(args, "O!O!d", &PyArray_Type, &X, &PyArray_Type, &rho, &cutoff)) 
    RAISE(PyExc_TypeError, "two numeric arrays and double expected", NULL);

  dims[0] = X->dimensions[0];
  dims[1] = 7;

  if (!(stats = em_statistics(self, (double*) X->data, (double*) rho->data, dims[0], cutoff)))
    RAISE(PyExc_StandardError, "em_stats failed", NULL);
  
  Stats = PyArray_SimpleNewFromData(2, dims, PyArray_DOUBLE, (char*) stats);
  free(stats);

  return Stats;
}

static PyObject* py_density_in_sphere(PyGridObject *self, PyObject *args) {

  int i, j, k, a, b, c, n_max, ijk[3], dims[1], index;
  double radius, *dens, radius2, value, cutoff;
  PyObject *x;

  if (!PyArg_ParseTuple(args, "dd", &radius, &cutoff)) 
    RAISE(PyExc_TypeError, "double expected", NULL);

  dims[0] = self->nx * self->ny * self->nz;

  if (!(dens = MALLOC(dims[0], double)))
    RAISE(PyExc_MemoryError, "malloc failed", NULL);
    
  for (i = 0; i < dims[0]; i++) dens[i] = 0.;

  n_max = (int) ceil(radius / self->spacing);

  printf("n_max = %d\n", n_max);

  radius2 = radius / self->spacing;
  radius2*= radius2;

  for (i = 0; i < self->nx; i++) {
    for (j = 0; j < self->ny; j++) {
      for (k = 0; k < self->nz; k++) {

	ijk[0] = i;
	ijk[1] = j;
	ijk[2] = k;

	if (self->values[grid_index(self, ijk)] < cutoff)
	  continue;

	value = 0.;

	for (a = -n_max; a <= n_max; a++) {

	  // in grid?
  
	  if (i + a < 0) continue;
	  else if (i + a >= self->nx) continue;

	  // calc contribution

	  for (b = -n_max; b <= n_max; b++) {

	    // in grid and sphere?
      
	    if (j + b < 0) continue;
	    else if (j + b >= self->ny) continue;
	    if (a*a + b*b > radius2) continue;

	    // calc contribution

	    for (c = -n_max; c <= n_max; c++) {

	      // in grid and sphere?

	      if (k + c < 0) continue;
	      else if (k + c >= self->nz) continue;
	      if (a*a + b*b + c*c > radius2) continue;

	      // calc contribution

	      index = self->nz * (self->ny * (i+a) + j+b) + k+c;

	      value += self->values[index];

	    }
	  }
	}

	dens[grid_index(self, ijk)] = value;

      }
    }
  }

  x = PyArray_SimpleNewFromData(1, dims, PyArray_DOUBLE, (char*) dens);
  free(dens);

  return x;
}

static PyObject* py_cutout_sphere(PyGridObject *self, PyObject *args) {

  int i, j, k, a, b, c, n_max, ijk[3], dims[2], index;
  double radius, *dens, radius2, value, x, y, z;
  PyObject *X;

  if (!PyArg_ParseTuple(args, "id", &index, &radius)) 
    RAISE(PyExc_TypeError, "int and double expected", NULL);

  grid_coordinates(self, index, ijk);

  n_max = (int) ceil(radius / self->spacing);

  dims[0] = 2 * n_max + 1;
  dims[0]*= dims[0] * dims[0];
  dims[1] = 4;

  if (!(dens = MALLOC(dims[1] * dims[0], double)))
    RAISE(PyExc_MemoryError, "malloc failed", NULL);
    
  for (i = 0; i < dims[1] * dims[0]; i++) dens[i] = 0.;

  radius2 = radius / self->spacing;
  radius2*= radius2;

  value = 0.;

  i = ijk[0]; 
  j = ijk[1];
  k = ijk[2];

  x = i * self->spacing + self->origin[0];
  y = j * self->spacing + self->origin[1];
  z = k * self->spacing + self->origin[2];

  for (a = -n_max; a <= n_max; a++) {
    for (b = -n_max; b <= n_max; b++) {
      for (c = -n_max; c <= n_max; c++) {

	index = (2*n_max+1) * ((2*n_max+1) * (a + n_max) + (b + n_max)) + (c + n_max);

	dens[4 * index + 0] = x + a * self->spacing;
	dens[4 * index + 1] = y + b * self->spacing;
	dens[4 * index + 2] = z + c * self->spacing;

	if (i + a < 0) continue;
	else if (i + a >= self->nx) continue;
    
	if (j + b < 0) continue;
	else if (j + b >= self->ny) continue;
	if (a*a + b*b > radius2) continue;

	if (k + c < 0) continue;
	else if (k + c >= self->nz) continue;
	if (a*a + b*b + c*c > radius2) continue;

	dens[4 * index + 3] = self->values[self->nz*(self->ny*(i+a)+j+b)+k+c];

      }
    }
  }

  X = PyArray_SimpleNewFromData(2, dims, PyArray_DOUBLE, (char*) dens);
  free(dens);

  return X;
}

static int compare (const void * a, const void * b) {
  /*
    for sorting
   */
  
  double x, y;

  x = *(double*) a;
  y = *(double*) b;

  if (x < y) return -1;
  else if (x > y) return 1;
  else return 0;

}

static PyObject* py_median_filtering(PyGridObject *self, PyObject *args) {

  int i, j, k, a, b, c, n, m, index, radius, ijk[3], dims[1];
  double *values, *bubble;
  PyObject *x;

  if (!PyArg_ParseTuple(args, "i", &radius)) 
    RAISE(PyExc_TypeError, "int expected", NULL);

  n = 2 * radius + 1;
  n*= n*n;

  dims[0] = self->nx * self->ny * self->nz;

  if (!(bubble = MALLOC(n, double)))
    RAISE(PyExc_MemoryError, "py_median_filtering: malloc failed", NULL);
  if (!(values = MALLOC(dims[0], double)))
    RAISE(PyExc_MemoryError, "py_median_filtering: malloc failed", NULL);

  for (i=0; i < self->nx; i++) {
    for (j=0; j < self->ny; j++) {
      for (k=0; k < self->nz; k++) {

	m = 0;

	for (a=-radius; a <= radius; a++){

	  if ((i+a < 0) || (i+a >= self->nx)) continue;

	  ijk[0] = i+a;

	  for (b=-radius; b <= radius; b++) {

	    if ((j+b < 0) || (j+b >= self->ny)) continue;

	    ijk[1] = j+b;

	    for (c=-radius; c <= radius; c++) {

	      if ((k+c < 0) || (k+c) >= self->nz) continue;

	      ijk[2] = k+c;

	      index = grid_index(self, ijk);

	      bubble[m] = self->values[index];
	      m++;
	    }
	  }	
 	}

	qsort((void*) bubble, m, sizeof(double), compare);

	ijk[0] = i;
	ijk[1] = j;
	ijk[2] = k;

	values[grid_index(self, ijk)] = bubble[m/2];
      }
    }
  }

  free(bubble);

  x = PyArray_SimpleNewFromData(1, dims, PyArray_DOUBLE, (char*) values);
  free(values);

  return x;
}

static PyMethodDef methods[] = {
  {"cc", (PyCFunction) py_cc, 1},
  {"cc_translation", (PyCFunction) py_cc_translation, 1},
  {"cc_tensor", (PyCFunction) py_cc_tensor, 1},
  {"grid_coordinates", (PyCFunction) py_coordinates, 1},
  {"density_in_sphere", (PyCFunction) py_density_in_sphere, 1},
  {"cutout_sphere", (PyCFunction) py_cutout_sphere, 1},
  {"grid_index", (PyCFunction) py_index, 1},
  {"assign", (PyCFunction) py_assign, 1},  
  {"mask", (PyCFunction) py_mask, 1},  
  {"add_density", (PyCFunction) py_add_density, 1},  
  {"update_gradient", (PyCFunction) py_update_gradient, 1},  
  {"mean_structure", (PyCFunction) py_mean_structure, 1},  
  {"mean_structure2", (PyCFunction) py_mean_structure2, 1},  
  {"em_stats", (PyCFunction) py_em_stats, 1},  
  {"add_density_full", (PyCFunction) py_add_density_full, 1},  
  {"add_points", (PyCFunction) py_add_points, 1},  
  {"add_weighted_points", (PyCFunction) py_add_weighted_points, 1},  
  {"calc_density", (PyCFunction) py_calc_density, 1},  
  {"add_density_slow", (PyCFunction) py_add_density_slow, 1},  
  {"transform_and_interpolate", (PyCFunction)py_transform_and_interpolate,1},  
  {"calc_density_slow", (PyCFunction) py_calc_density_slow, 1},  
  {"setup_neighbors", (PyCFunction) py_setup_neighbors, 1},  
  {"set_density", (PyCFunction) py_set_density, 1},  
  {"set_rho", (PyCFunction) py_set_rho, 1},  
  {"project1D", (PyCFunction) project1D, 1},  
  {"project2D", (PyCFunction) project2D, 1},  
  {"convolve", (PyCFunction) py_convolve, 1},  
  {"apply_laplacian", (PyCFunction)apply_laplacian,1},  
  {"median_filtering", (PyCFunction)py_median_filtering,1},  
  {"evaluate", (PyCFunction)py_eval,1},  
  {NULL, NULL }
};

static void dealloc(PyGridObject *self) {
  if (self->values) free(self->values);
  if (self->neighbors) free(self->neighbors);
  if (self->neighbors_ijk) free(self->neighbors_ijk);
  if (self->neighbor_density) free(self->neighbor_density);
  PyObject_Del(self);
}


static PyObject *getattr(PyGridObject *self, char *name) {
  double *vec;
  int i, k, dims[2];
  npy_intp ret_dims[1];
  PyObject *t;
  PyObject *ret;
  
  if (!strcmp(name, "nx"))
    return Py_BuildValue("i", self->nx);

  else if (!strcmp(name, "ny"))
    return Py_BuildValue("i", self->ny);

  else if (!strcmp(name, "nz"))
    return Py_BuildValue("i", self->nz);

  else if (!strcmp(name, "n_shells"))
    return Py_BuildValue("i", self->n_shells);

  else if (!strcmp(name, "spacing"))
    return Py_BuildValue("d", self->spacing);

  else if (!strcmp(name, "width"))
    return Py_BuildValue("d", self->width);

  else if (!strcmp(name, "origin"))
    if (!self->origin) {
      RETURN_PY_NONE;
    }
    else {
      t = PyTuple_New(3);
      for (i = 0; i < 3; i++)
	PyTuple_SET_ITEM(t, i, Py_BuildValue("d", self->origin[i]));
      return t;
    }

  else if (!strcmp(name, "values"))
    if (!self->values) {
      RETURN_PY_NONE;
    }
    else {
      ret_dims[0] = self->nx * self->ny * self->nz;
      ret =  PyArray_SimpleNew(1, ret_dims, NPY_DOUBLE);
      vec = (double *) PyArray_DATA(ret);
      /*
       *  NOTE: Treating PyArray_DATA(ret) as if it were a contiguous one-dimensional C
       *  array is safe, because we just created it with PyArray_SimpleNew, so we know
       *  that it is, in fact, a one-dimensional contiguous array.
       */
      for (k = 0; k < ret_dims[0]; ++k) {
        vec[k] = self->values[k];
        // printf("value pair %4.2f %4.2f \n", vec[k], self->values[k]);
      }
      return ret;
    }

  else if (!strcmp(name, "neighbors"))
    if (!self->neighbors) {
      RETURN_PY_NONE;
    }
    else {
      return PyArray_SimpleNewFromData(1,&self->n_neighbors,PyArray_INT, 
					 (char*) self->neighbors);
    }

  else if (!strcmp(name, "neighbors_ijk"))
    if (!self->neighbors_ijk) {
      RETURN_PY_NONE;
    }
    else {
      dims[0] = self->n_neighbors;
      dims[1] = 3;
      return PyArray_SimpleNewFromData(2,dims,PyArray_INT, 
					 (char*) self->neighbors_ijk);
    }

  else if (!strcmp(name, "neighbor_density"))
    if (!self->neighbor_density) {
      RETURN_PY_NONE;
    }
    else {
      return PyArray_SimpleNewFromData(1,&self->n_neighbors,PyArray_DOUBLE, 
					 (char*) self->neighbor_density);
    }

  else
    return Py_FindMethod(methods, (PyObject*) self, name);

}

static int setattr(PyGridObject *self, char *name, PyObject *op) {

  int i, result = 0;

  if (!strcmp(name, "spacing")) {
    self->spacing = (double) PyFloat_AsDouble(op);
    //    result = setup_neighbors(self);
  }

  else if (!strcmp(name, "width")) {
    self->width = (double) PyFloat_AsDouble(op);
    //    result = setup_neighbors(self);
  }

  else if (!strcmp(name, "n_shells")) {
    self->n_shells = (int) PyInt_AsLong(op);
    //    result = setup_neighbors(self);
  }
  
  else if (!strcmp(name, "origin")) {
    self->origin[0] = (double) PyFloat_AsDouble(PyTuple_GetItem(op,0));
    self->origin[1] = (double) PyFloat_AsDouble(PyTuple_GetItem(op,1));
    self->origin[2] = (double) PyFloat_AsDouble(PyTuple_GetItem(op,2));
  }
  
  return result;
}

static char __doc__[] = "grid"; 

PyTypeObject PyGrid_Type = { 
	PyObject_HEAD_INIT(NULL)
	0,			  /*ob_size*/
	"grid",               /*tp_name*/
	sizeof(PyGridObject), /*tp_basicsize*/
	0,			                /*tp_itemsize*/
	(destructor)dealloc,                         /*tp_dealloc*/
	(printfunc)NULL,	                     /*tp_print*/
       	(getattrfunc)getattr,                        /*tp_getattr*/
	(setattrfunc)setattr,                        /*tp_setattr*/
	(cmpfunc)NULL,         	                     /*tp_compare*/
	(reprfunc)NULL,	                             /*tp_repr*/

	NULL,		                             /*tp_as_number*/
	NULL,	                                     /*tp_as_sequence*/
	NULL,		 	                     /*tp_as_mapping*/

	(hashfunc)0,		                     /*tp_hash*/
	(ternaryfunc)0,		                     /*tp_call*/
	(reprfunc)0,		                     /*tp_str*/
		
	0L,0L,0L,0L,
	__doc__            /* Documentation string */
};

 
static int init(PyGridObject *self, PyTupleObject *sizes) {

  int i;

  self->nx = (int)PyInt_AsLong(PyTuple_GetItem((PyObject*)sizes,0));
  self->ny = (int)PyInt_AsLong(PyTuple_GetItem((PyObject*)sizes,1));
  self->nz = (int)PyInt_AsLong(PyTuple_GetItem((PyObject*)sizes,2));
  self->values = MALLOC(self->nx * self->ny * self->nz, double);
  for (i = 0; i < self->nx * self->ny * self->nz; i++)
    self->values[i] = 0.;
  return 0;
} 

PyObject * PyGrid_New(PyObject *self, PyObject *args) {

  PyGridObject *ob;
  PyTupleObject *t;

  import_array();

  if (!PyArg_ParseTuple(args, "O!", &PyTuple_Type, &t)) 
    RAISE(PyExc_TypeError, "tuple expected.", NULL);
  ob = PyObject_NEW(PyGridObject, &PyGrid_Type);

  ob->nx = 0;
  ob->ny = 0;
  ob->nz = 0;

  ob->n_shells = 5;

  ob->spacing = 0.;
  ob->width = 1.;

  vector_set(ob->origin,0.);

  ob->values = NULL;
  ob->neighbors = NULL;
  ob->neighbors_ijk = NULL;
  ob->neighbor_density = NULL;
  ob->n_neighbors = 0;

  if (init(ob, t))
    RAISE(PyExc_StandardError, "error in init", NULL);

  return (PyObject*) ob;
}

