
// Parallel Conjugate Gradient method 
// By Max Reeves
// Parallel and Distributed Programming
// Uppsala University, Spring 2017


#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


typedef struct {
  // Struct representing local data in a system of linear equations Ax = b, 
  // where A is symmetric and positive definite
  int N;

  double *A;
  double *x;
  double *x_star; // vector x* for approximated solution
  double *b;
} equation_data;


typedef struct {
  int N;
  int Np;

  int rank;
  int coord; // Coordinate of this row in the 1D grid
  int displ; // The displacement of this row from the top
 
  int count; // Height of this row
  int count_max; // The maximum height of a row 
  int count_min; // The minimum height of a row

  MPI_Datatype block_trans_t; // Type for a block of columns in a row
  MPI_Datatype row_t; // Type for a full row

  int *ranks; // Rank of all processes, indexed by coordinate
  int *counts; // Dimension of all rows, indexed by rank
  int *displs; // Displacement of all rows, indexed by rank

  MPI_Comm comm;
} process_data;


process_data set_up_world(int Np, int N);
void print_matrix(double *A, int N, int M);
void print_matrix_transpose(double *A, int N, int M);
double *random_matrix(int N, int M);
equation_data random_linear_system(process_data row);
double *solve_conjugate_gradient(double *A, double *b, int N, int max_steps, double tol);
void solve_conjugate_gradient_par(process_data row, equation_data equation, int max_steps, double tol);
double max_error(double *real_x, double *approx_x, int N);
void malloc_test(void *ptr);
int is_symmetric(double *A, int N);


int main (int argc, char **argv) {
  MPI_Init(&argc, &argv);

  double tol, *A, *x, *x_star, *x_star_seq, *b, start_time, setup_time, solution_time;
  int Np, N, tol_digits, temp_rank;
  equation_data equation;
  process_data row;
  
  MPI_Comm_size(MPI_COMM_WORLD, &Np);
  MPI_Comm_rank(MPI_COMM_WORLD, &temp_rank);
  
  if (argc == 2)
    N = atoi(argv[1]);

  if (argc != 2 || N < 1 || Np > N) {
    if (temp_rank == 0)
      printf("Incorrect input argument. Expected and integer N, 0 < Np <= N\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
    return 0;
  }

  // Set the tolerance
  tol_digits = 10;
  tol = 1.0/pow(10.0, tol_digits + 1.0);

  // Set up the world and get a struct containing all process info needed
  row = set_up_world(Np, N);

  // Set up the linear system to be solved in parallel
  start_time = MPI_Wtime();
  srand((unsigned)start_time*row.rank + start_time);
  equation = random_linear_system(row);
  MPI_Barrier(row.comm); // For fairer timing
  setup_time = MPI_Wtime() - start_time;

  // Solve the linear system in parallel
  start_time = MPI_Wtime();
  solve_conjugate_gradient_par(row, equation, N, tol);
  MPI_Barrier(row.comm); // For fairer timing
  solution_time = MPI_Wtime() - start_time;

  // Gather and stuff
  if (row.rank == 0) {
#if defined(DEBUG) 
    A = malloc(N*N*sizeof(double));
    b = malloc(N*sizeof(double));
#endif
#if defined(DEBUG) || defined(VERIFY) 
    x = malloc(N*sizeof(double));
    x_star = malloc(N*sizeof(double));
#endif
  }

#if defined(DEBUG)
  MPI_Gatherv(equation.A, row.count, row.row_t, A, row.counts, row.displs, row.row_t, 0, row.comm); 
  MPI_Gatherv(equation.b, row.count, MPI_DOUBLE, b, row.counts, row.displs, MPI_DOUBLE, 0, row.comm);
#endif
#if defined(DEBUG) || defined(VERIFY)
  MPI_Gatherv(equation.x, row.count, MPI_DOUBLE, x, row.counts, row.displs, MPI_DOUBLE, 0, row.comm);
  MPI_Gatherv(equation.x_star, row.count, MPI_DOUBLE, x_star, row.counts, row.displs, MPI_DOUBLE, 0, row.comm);
#endif

  if (row.rank == 0) {
#if defined(DEBUG)
    x_star_seq = solve_conjugate_gradient(A, b, N, N, tol);

    if (N <= 20) {
      printf("A:\n"); print_matrix(A, N, N);
      printf("b:\n"); print_matrix(b, N, 1);
      printf("x:\n"); print_matrix(x, N, 1);
      printf("x*:\n"); print_matrix(x_star, N, 1);
      printf("x* seq:\n"); print_matrix(x_star_seq, N, 1);
    } else {
      printf("Warning: Attempts to print matrices larger than 20x20 are suppressed.\n");
    }
    printf("A symmetric: %s\n", (is_symmetric(A, N)) ? ("Yes") : ("No"));
    printf("Seq max error: %.*f\n", tol_digits + 5, max_error(x, x_star_seq, N));

    free(A);
    free(b);
    free(x_star_seq);
#endif
#if defined(DEBUG) || defined(VERIFY) 
    printf("Par max error: %.*f\n", tol_digits + 5, max_error(x, x_star, N));

    free(x);
    free(x_star);
#endif    

    printf("Generate time: %f\n", setup_time);
    printf("Solution time: %f\n", solution_time);
  }

  free(equation.A);
  free(equation.b);
  free(equation.x);
  free(equation.x_star);

  free(row.ranks);
  free(row.counts);
  free(row.displs);

  MPI_Finalize();

  return 0;
}


process_data set_up_world(int Np, int N) {
  process_data row;
  int period, size, large_count, col_cnt_dsp[3], buf[3*Np];
  MPI_Aint lb, extent;
  MPI_Datatype row_t;
  
  // Store number of processes Np and dimension N
  row.N = N;
  row.Np = Np;

  // Create 1D communicator and save ranks and coordinates
  period = 1;
  MPI_Cart_create(MPI_COMM_WORLD, 1, &Np, &period, 0, &row.comm);
  MPI_Comm_rank(row.comm, &row.rank);
  MPI_Cart_coords(row.comm, row.rank, 1, &row.coord);

  // Calculate the number of rows handled by each process
  large_count = N%Np;
  row.count_min = N/Np;
  row.count_max = (large_count == 0) ? (row.count_min) : (row.count_min + 1);
  row.count = (row.coord < large_count) ? (row.count_max) : (row.count_min);
  row.displ = row.coord*(row.count_min) + ((row.coord <= large_count) ? (row.coord) : (large_count));

  // Create types for a block within a row, a transposed block, and a full row
  MPI_Type_vector(row.count, 1, row.N, MPI_DOUBLE, &row_t); 
  MPI_Type_create_resized(row_t, 0, sizeof(double), &row.block_trans_t);

  MPI_Type_vector(1, row.N, 1, MPI_DOUBLE, &row.row_t);
  MPI_Type_commit(&row.block_trans_t);
  MPI_Type_commit(&row.row_t);

  // Gather rank, count, and displacement of each coordinate
  col_cnt_dsp[0] = row.coord; 
  col_cnt_dsp[1] = row.count; 
  col_cnt_dsp[2] = row.displ;
    
  MPI_Allgather(col_cnt_dsp, 3, MPI_INT, buf, 3, MPI_INT, row.comm);
  
  row.ranks = malloc(Np*sizeof(int));
  row.counts = malloc(Np*sizeof(int));
  row.displs = malloc(Np*sizeof(int));

  for (int i = 0; i < Np; ++i) {
    row.ranks[buf[3*i]] = i;
    row.counts[i] = buf[3*i + 1];
    row.displs[i] = buf[3*i + 2];
  }

  return row;
}


// Print a matrix
void print_matrix(double *A, int N, int M) {
  char buf[32];

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      snprintf(buf, 6, "%.10f", A[i*M + j]);
      printf("%s ", buf);
    }
    printf("\n");
  }
  printf("\n");
}


// Print the transpose of a matrix
void print_matrix_transpose(double *A, int N, int M) {
  for (int j = 0; j < M; ++j) {
    for (int i = 0; i < N; ++i) {
      printf("%f ", A[i*M + j]);
    }
    printf("\n");
  }
  printf("\n");
}


// Exits program with error message if ptr is NULL 
void malloc_test(void *ptr) {
  if (ptr == NULL) {
    printf("malloc failed\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
    exit(0);
  }
}


// Generate a random matrix with numbers in range (0,1)
double *random_matrix(int N, int M) {
  double *A = malloc(N*M*sizeof(double));
  malloc_test(A);

  for (int i = 0; i < N*M; ++i)
    A[i] = (double)rand() / (double)((unsigned)RAND_MAX + 1);

  return A;
}


// Generate a random positive definite equation with solution by 
// 1. Generating a random matrix A
// 2. Calculating A = A' + A
// 3. Adding N to each element on the diagonal of A
// 4. Generating a random solution x
// 5. Calculating the rhs in A * x = b
equation_data random_linear_system(process_data row) {
  equation_data equation;
  double *B, *a, *b, *x_recv, *x_send, *x_tmp;
  int B_size, coord_work, coord_send, coord_recv, rank_work, rank_send, rank_recv, 
    rank_up, rank_down, rank_block, displ_work, displ_send, count_work, count_send;
  MPI_Request send_req, send_reqs[row.Np];
  MPI_Status recv_stat;

  equation.N = row.N;
  equation.A = random_matrix(row.count, row.N);

  if (row.Np > 1) {
    B_size = row.count_max*row.count_max;
    B = malloc(B_size*sizeof(double));
    malloc_test(B);
  }

  // Calculate A = 0.5*(A' + A) and add N to the diagonal
  for (int n = 0; n < row.Np; ++n) {
    coord_work = (row.coord + n)%row.Np;
    coord_recv = (row.Np + row.coord - 1 - n)%row.Np;
    coord_send = (coord_work + 1)%row.Np;
    rank_work = row.ranks[coord_work];
    rank_recv = row.ranks[coord_recv];
    rank_send = row.ranks[coord_send];
    displ_work = row.displs[rank_work];
    count_work = row.counts[rank_work];
    
    if ((n < row.Np - 1) && (coord_work != coord_recv))
      MPI_Isend(&equation.A[row.displs[rank_recv]], row.counts[rank_recv], row.block_trans_t, rank_recv, n, row.comm, &send_req);	
        
    if (n == 0) {
      // Don't use the buffer, just calculate the diagonal block addition in-place
      for (int i = 0; i < row.count; ++i) {
	a = equation.A + (i*row.N + displ_work);
	for (int j = 0; j < count_work; ++j) {
	  b = equation.A + (j*row.N + displ_work);
	  if (j < i)
	    a[j] = b[i];
	  else 
	    a[j] = 0.5*(a[j] + b[i]);
	  if (j == i)
	    a[j] += row.N;
	}
      }
    } else if (n > row.Np/2) {
      // Just copy B
      for (int i = 0; i < row.count; ++i) {
	a = equation.A + (i*row.N + displ_work);
	b = B + (i*count_work);
	for (int j = 0; j < count_work; ++j)
	  a[j] = b[j]; 
      }
    } else {
      // Add B to A
      for (int i = 0; i < row.count; ++i) {
	a = equation.A + (i*row.N + displ_work);
	b = B + (i*count_work);
	for (int j = 0; j < count_work; ++j)
	  a[j] = 0.5*(a[j] + b[j]); 
      }
    }

    if (n < row.Np - 1) {
      if (coord_work != coord_recv) {
	MPI_Recv(&(B[0]), B_size, MPI_DOUBLE, rank_send, n, row.comm, MPI_STATUS_IGNORE);
	MPI_Wait(&send_req, MPI_STATUS_IGNORE);
      } else {
	MPI_Sendrecv(&equation.A[row.displs[rank_recv]], row.counts[rank_recv], row.block_trans_t, rank_recv, n,
		     &(B[0]), B_size, MPI_DOUBLE, rank_send, n, row.comm, MPI_STATUS_IGNORE);
      }
    }
  }

  if (row.Np > 1)
    free(B); // Useless memory, free before allocating more

  // Generate random solution x, zero matrix b, and memory for x*
  equation.x = random_matrix(row.count, 1);
  equation.x_star = calloc(row.count, sizeof(double)); 
  equation.b = calloc(row.count, sizeof(double));
  malloc_test(equation.x_star);
  malloc_test(equation.b);

  // Create tempory x vectors for sending and receiving
  x_recv = malloc(row.count_max*sizeof(double));
  x_send = malloc(row.count_max*sizeof(double));
  malloc_test(x_recv);
  malloc_test(x_send);

  rank_up = row.ranks[(row.Np + row.coord - 1)%row.Np];
  rank_down = row.ranks[(row.coord + 1 )%row.Np];
  
  // Initially store local x in x_recv
  memcpy(x_recv, equation.x, row.count*sizeof(double));

  // Perform matrix-vector multiplication to calculate rhs b
  for (int n = 0; n < row.Np; ++n) {
    x_tmp = x_recv;
    x_recv = x_send;
    x_send = x_tmp;

    if (n < row.Np-1) 
      MPI_Isend(x_send, row.count_max, MPI_DOUBLE, rank_up, 111, row.comm, &send_req);

    rank_block = row.ranks[(row.coord + n)%row.Np];
    displ_send = row.displs[rank_block];
    count_send = row.counts[rank_block];
    for (int i = 0; i < row.count; ++i) {
      a = equation.A + (i*row.N + displ_send);
      for (int j = 0; j < count_send; ++j) 
	equation.b[i] += x_send[j]*a[j];
    }

    if (n < row.Np-1) {
      MPI_Recv(x_recv, row.count_max, MPI_DOUBLE, rank_down, 111, row.comm, MPI_STATUS_IGNORE);
      MPI_Wait(&send_req, MPI_STATUS_IGNORE);
    }
  }

  free(x_recv);
  free(x_send);

  return equation;
} 


// Calculate max(abs(real_x-approx_x))
double max_error(double *real_x, double *approx_x, int N) {
  double error, max;
  
  max = 0.0;
  for (int i = 0; i < N; ++i) {
    error = fabs(real_x[i] - approx_x[i]);
    if (error > max)
      max = error;
  }

  return max;
}


// Determine if a matrix is symmetric
int is_symmetric(double *A, int N) {
  int bool = 1;

  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      if (A[i*N+j] != A[j*N+i])
	bool = 0;

  return bool;
}


// Solve Ax = b for x, using the Conjugate Gradient method.
// Terminates once the maximum number of steps or tolerance has been reached
double *solve_conjugate_gradient(double *A, double *b, int N, int max_steps, double tol) {
  double *x, *r, *p, *a, *z;
  double gamma, gamma_new, alpha, beta;

  x = malloc(N*sizeof(double));
  r = malloc(N*sizeof(double));
  p = malloc(N*sizeof(double));
  z = malloc(N*sizeof(double));

  malloc_test(x);
  malloc_test(r);
  malloc_test(p);
  malloc_test(z);

  // x = [0 ... 0]
  // r = b - A * x
  // p = r
  // gamma = r' * r
  gamma = 0.0;
  for (int i = 0; i < N; ++i) {
    x[i] = 0.0;
    r[i] = b[i];
    p[i] = r[i];
    gamma += r[i] * r[i];
  }

  for (int n = 0; n < max_steps; ++n) {
    // z = A * p
    for (int i = 0; i < N; ++i) {
      a = A + (i*N);
      z[i] = 0.0;
      for (int j = 0; j < N; ++j)
	z[i] += a[j] * p[j]; 
    }

    // alpha = gamma / (p' * z)
    alpha = 0.0;
    for (int i = 0; i < N; ++i)
      alpha += p[i] * z[i];
    alpha = gamma / alpha;
    
    // x = x + alpha * p
    // r = r - alpha * z
    // gamma_new = r' * r
    gamma_new = 0.0;
    for (int i = 0; i < N; ++i) {
      x[i] += alpha * p[i];
      r[i] -= alpha * z[i];
      gamma_new += r[i] * r[i];
    }

    if (sqrt(gamma_new) < tol)
      break;
    
    beta = gamma_new / gamma;

    // p = r + (gamma_new / gamma) * p;
    for (int i = 0; i < N; ++i)
      p[i] = r[i] + beta * p[i];
    
    // gamma = gamma_new
    gamma = gamma_new;
  }

  free(r);
  free(p);
  free(z);

  return x;
}


void solve_conjugate_gradient_par(process_data row, equation_data equation, int max_steps, double tol) {
  double *a, *p, *p_recv, *p_send, *p_tmp, *r, *z, *x;
  double alpha, alpha_tmp, beta, gamma, gamma_new, gamma_tmp;
  int rank_up, rank_down, rank_block, displ_send, count_send;
  MPI_Request send_req;
  
  x = &(equation.x_star[0]);

  p = malloc(row.count*sizeof(double));
  r = malloc(row.count*sizeof(double));
  z = malloc(row.count*sizeof(double));

  malloc_test(p);
  malloc_test(r);
  malloc_test(z);

  p_recv = malloc(row.count_max*sizeof(double));
  p_send = malloc(row.count_max*sizeof(double));
  malloc_test(p_recv);
  malloc_test(p_send);

  rank_up = row.ranks[(row.Np + row.coord - 1)%row.Np];
  rank_down = row.ranks[(row.coord + 1)%row.Np];

  // x = [0 ... 0] - initial guess
  // r = b
  // p = r
  // gamma = r' * r
  gamma_tmp = 0.0;
  for (int i = 0; i < row.count; ++i) {
    x[i] = 0.0;
    r[i] = equation.b[i];
    p[i] = r[i]; 
    p_recv[i] = p[i];
    z[i] = 0.0;
    gamma_tmp += r[i] * r[i];
  }
  MPI_Allreduce(&gamma_tmp, &gamma, 1, MPI_DOUBLE, MPI_SUM, row.comm);

  for (int n = 0; n < max_steps; ++n) {
    // z = A * p
    for (int m = 0; m < row.Np; ++m) {
      p_tmp = p_recv;
      p_recv = p_send;
      p_send = p_tmp;

      if (m < row.Np-1) 
	MPI_Isend(p_send, row.count_max, MPI_DOUBLE, rank_up, 222, row.comm, &send_req);

      rank_block = row.ranks[(row.coord + m)%row.Np];
      displ_send = row.displs[rank_block];
      count_send = row.counts[rank_block];

      for (int i = 0; i < row.count; ++i) {
	a = equation.A + (i*row.N + displ_send);
	for (int j = 0; j < count_send; ++j) 
	  z[i] += p_send[j]*a[j];
      }

      if (m < row.Np-1) {
	MPI_Recv(p_recv, row.count_max, MPI_DOUBLE, rank_down, 222, row.comm, MPI_STATUS_IGNORE);
	MPI_Wait(&send_req, MPI_STATUS_IGNORE);
      }
    }

    // alpha = gamma / (p' * z)
    alpha_tmp = 0.0;
    for (int i = 0; i < row.count; ++i)
      alpha_tmp += p[i] * z[i];

    MPI_Allreduce(&alpha_tmp, &alpha, 1, MPI_DOUBLE, MPI_SUM, row.comm);

    alpha = gamma / alpha;

    // x = x + alpha * p
    // r = r - alpha * z
    // gamma_new = r' * r
    gamma_tmp = 0.0;
    for (int i = 0; i < row.count; ++i) {
      x[i] += alpha * p[i];
      r[i] -= alpha * z[i];
      gamma_tmp += r[i] * r[i];
    }

    MPI_Allreduce(&gamma_tmp, &gamma_new, 1, MPI_DOUBLE, MPI_SUM, row.comm);

    if (sqrt(gamma_new) < tol)
      break;
    
    beta = gamma_new / gamma;

    // p = r + beta * p;
    for (int i = 0; i < row.count; ++i) {
      p[i] = r[i] + beta * p[i];
      p_recv[i] = p[i];
      z[i] = 0.0;
    }
    
    // gamma = gamma_new
    gamma = gamma_new;
  }

  free(p);
  free(r);
  free(z);
  free(p_send);
  free(p_recv);

  return;
}
