#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <argp.h>
#include <assert.h>
#include <stdbool.h>

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
 *  program info
 *-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/
/**
 * Author: Bc. Dominika Draesslerova
 * program for solving system of linear equations
 * available 4 methods:
 *      gradient descent method         11/2020
 *      conjugate gradients method      11/2020
 *      Jacobi method                   12/2020
 *      Gauss-Seibel method             12/2020
 * TODO:
 *      input treatment (is input - symetric, positive defitive, diagonal dominant)
 *      boundary conditions test
 * 
 * Usage: a.out [-bcgjs?V] [-k K] [-l L] [-o OUTFILE] [--jacobi_1]
            [--conjugate-gradient] [--gradient-descent] [--jacobi]
            [--number-of-steps=K] [--lambda=L] [--output=OUTFILE]
            [--gauss-seidel] [--help] [--usage] [--version] [FILENAMES]...
 *
 * Usage: a.out [OPTION...] [FILENAMES]...
Solve algebraic equations

  -b, --jacobi_1             Use Jacobi method using LU decomposition.
  -c, --conjugate-gradient   Use conjugate gradient method.
  -g, --gradient-descent     Use gradient descent method.
  -j, --jacobi               Use Jacobi method.
  -k, --number-of-steps=K    maximal number of iterative steps
  -l, --lambda=L
  -o, --output=OUTFILE       Output file path.
  -s, --gauss-seidel         Use Gauss-Seidel method.
  -?, --help                 Give this help list
      --usage                Give a short usage message
  -V, --version              Print program version

Mandatory or optional arguments to long options are also mandatory or optional
for any corresponding short options.

Report bugs to <draesdom@cvut.cz>.
*/

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
 *  globals and constants
 *-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

#define EPSILON 1e-6

size_t n;     /*  matrix/vector size     */
int k = 1000; /*  max steps             */

double **matrix;
double *vector;

typedef struct arguments
{
    enum method
    {
        METHOD_GD,
        METHOD_CG,
        METHOD_JM,
        METHOD_GS,
        METHOD_J1
    } method;
    char *outfile;
} Arguments;

Arguments arguments;

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
 *  parser setting
 *-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*  parsing setting */
const char *argp_program_version = "1.0";
const char *argp_program_bug_address = "<draesdom@cvut.cz>";
static char doc[] = "Solve algebraic equations";
static char args_doc[] = "[FILENAMES]...";
static struct argp_option options[] = {
    {"gradient-descent", 'g', 0, 0, "Use gradient descent method."},
    {"conjugate-gradient", 'c', 0, 0, "Use conjugate gradient method."},
    {"jacobi", 'j', 0, 0, "Use Jacobi method."},
    {"jacobi_1", 'b', 0, 0, "Use Jacobi method using LU decomposition."},
    {"gauss-seidel", 's', 0, 0, "Use Gauss-Seidel method."},
    {"output", 'o', "OUTFILE", 0, "Output file path."},
    {"number-of-steps", 'k', "K", 0, "maximal number of iterative steps"},
    {"lambda", 'l', "L", 0, ""},
    {0}};

/*  set method, format and other program options    */
static error_t parse_opt(int key, char *arg, struct argp_state *state)
{
    struct arguments *arguments = state->input;
    switch (key)
    {
    /*  method keys */
    case 'g':
        arguments->method = METHOD_GD;
        break;
    case 'c':
        arguments->method = METHOD_CG;
        break;
    case 'j':
        arguments->method = METHOD_JM;
        break;
    case 'b':
        arguments->method = METHOD_J1;
        break;
    case 's':
        arguments->method = METHOD_GS;
        break;
    /*  output files    */
    case 'o':
        arguments->outfile = arg;
        break;
    /*  other   */
    case 'k':
        k = atoi(arg);
        break;
    case ARGP_KEY_ARG:
        return 0;
    default:
        return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

static struct argp argp = {options, parse_opt, args_doc, doc, 0, 0, 0};
/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
 *  other
 *-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*  return string name of enum method   */
char *method_to_string(enum method a)
{
    switch (a)
    {
    case METHOD_GD:
        return "gradient descent method";
        break;
    case METHOD_CG:
        return "conjugate gradient method";
        break;
    case METHOD_JM:
        return "Jacobi method";
        break;
    case METHOD_J1:
        return "Jacobi method, with LU decomposition";
        break;
    case METHOD_GS:
        return "Gauss-Seibel method";
        break;
    default:
        return "unknown_method";
        break;
    }
}

/*  control print of matrix */
void print_m(double **a)
{

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            printf("%lf ", a[i][j]);
        printf("\n");
    }
}

/*  control print of vector */
void print_v(double *b)
{
    for (int j = 0; j < n; j++)
        printf("%1.15e\n", b[j]);
    printf("\n");
}

/*  free allocated memory */
void tidy_up(double *output)
{
    for (int i = 0; i < n; i++)
        free(matrix[i]);
    free(matrix);
    free(vector);
    free(output);
}

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
 *  readers/writers
 *-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*  read matrix */
int read_matrix(const char *filename)
{
    FILE *pf;
    pf = fopen(filename, "r");
    if (pf == NULL)
        return 0;

    printf("Reading 2d array format\n");
    fscanf(pf, " %ld", &n);
    /*  read like 2d array */
    matrix = calloc(n, sizeof(double *)); /*  column index */
    for (int i = 0; i < n; ++i)
    {
        matrix[i] = calloc(n, sizeof(double));
        for (int j = 0; j < n; ++j)
            fscanf(pf, "%lf", matrix[i] + j);
    }

    fclose(pf);
    return 1;
}

/*  read vector */
int read_vector(const char *filename)
{
    FILE *pf;
    pf = fopen(filename, "r");
    if (pf == NULL)
        return 0;

    for (int i = 0; i < n; ++i)
        fscanf(pf, " %lf\n", &vector[i]);

    fclose(pf);
    return 1;
}

/*  write output */
int write_output(double *a, const char *filename)
{
    FILE *pf;
    pf = fopen(filename, "w");
    if (pf == NULL)
        return 0;

    for (int i = 0; i < n; ++i)
        fprintf(pf, "%1.15e\n", a[i]);

    fclose(pf);
    return 1;
}

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
 *  calculations
 *-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*  matric * vector */
void multiply_m_v(double **a, double *v, double *res)
{
    for (int i = 0; i < n; i++)
    {
        res[i] = 0;
        for (int j = 0; j < n; j++)
            res[i] += (a[i][j] * v[j]);
    }
}

/*  vector * vector */
double multiply_v_v(double *a, double *b)
{
    double sum = 0;
    for (int i = 0; i < n; i++)
        sum += a[i] * b[i];
    return sum;
}

/*  vector * constant */
void multiply_v_c(double *a, double c, double *res)
{
    for (int i = 0; i < n; i++)
        res[i] = a[i] * c;
}

/*  vector - vector */
void subtract_v_v(double *a, double *b, double *res)
{
    for (int i = 0; i < n; i++)
        res[i] = a[i] - b[i];
}

/*  vector + vector */
void addition_v_v(double *a, double *b, double *res)
{
    for (int i = 0; i < n; i++)
        res[i] = a[i] + b[i];
}

/*  return ordinary vector size */
double vector_size(double *v)
{
    return sqrt(multiply_v_v(v, v));
}

/*  return vector size in generalized sense */
double gen_vector_size(double *v)
{
    double r[n];
    multiply_m_v(matrix, v, r);
    return sqrt(multiply_v_v(v, r));
}

/*  create copy of vector */
void copy_vector(double *s, double *r)
{
    for (int i = 0; i < n; i++)
        r[i] = s[i];
}

/* replace matrix with multiplication D^(-1)*(L+U) */
int get_DxR(double **res)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            if (matrix[i][i] == 0)
            {
                printf("tady %lf \n", matrix[i][i]);
                return -1;
            }
            if (i == j)
                continue;
            res[i][j] = -(matrix[i][j] / matrix[i][i]);
        }
        res[i][i] = 0;
    }
    return 1;
}

/* return multiplication inverse diagonal of matrix with vector b*/
int get_Db(double *res)
{
    for (int i = 0; i < n; i++)
    {
        if (matrix[i][i] == 0)
        {
            return -1;
        }
        res[i] = vector[i] / matrix[i][i];
    }
    return 1;
}

/*-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
 *  methods
 *-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=*/

/*  gradient descent method */
int gradient_descent_method(double *xk)
{

    /*  initialization vector x */
    double *Ax = (double *)calloc(n, sizeof(double));
    double *rk = (double *)calloc(n, sizeof(double));
    double *ar = (double *)calloc(n, sizeof(double));
    for (int i = 0; i < n; i++)
    {
        xk[i] = 0;
        rk[i] = 0;
        Ax[i] = 0;
        ar[i] = 0;
    }

    double ak = 0; /*  lenght of step */

    /*  rk = b - A*xk */
    multiply_m_v(matrix, xk, Ax);
    subtract_v_v(vector, Ax, rk);

    double rho = multiply_v_v(rk, rk);
    if (sqrt(rho) == 0)
        return 1;

    /*  lets do it! :)) */
    for (int i = 0; i < k; i++)
    {
        if ((i % 100) == 0 && i > 1)
        {
            printf("iteration %d, diff: %1.15e\n", i - 1, rho);
        }
        /*  ak = rk'*rk / rk'*A*rk */
        multiply_m_v(matrix, rk, Ax);
        ak = rho / multiply_v_v(rk, Ax);

        /*  xk_1 = xk + ak*rk */
        multiply_v_c(rk, ak, ar);
        addition_v_v(xk, ar, xk);

        /*  rk = b - A*xk */
        multiply_v_c(Ax, ak, Ax);
        subtract_v_v(rk, Ax, rk);

        rho = multiply_v_v(rk, rk);

        if (sqrt(rho) < EPSILON)
        {
            printf("stop at %d iteration\n", i);
            break;
        }
    }
    return 1;
}

/*  conjugate gradient method */
int conjugate_gradient_method(double *xk)
{
    /*  initialization vectors */
    double *sk = (double *)calloc(n, sizeof(double));
    double *pom1 = (double *)calloc(n, sizeof(double));
    double *rk = (double *)calloc(n, sizeof(double));

    /* rk = b - A*xk */
    multiply_m_v(matrix, xk, rk);
    subtract_v_v(vector, rk, rk);

    /* sk = rk */
    copy_vector(rk, sk);

    double ak = 0; /*  lenght of step */
    double bk = 0; /*  pom constant */
    double diff = 0;
    double rho = multiply_v_v(rk, rk);
    if (sqrt(rho) == 0)
        return 1;

    for (int i = 0; i < k; i++)
    {
        if ((i % 100) == 0 && i > 1)
        {
            printf("iteration %d, diff: %1.15e\n", i - 1, diff);
        }
        /*  ak = rk'*rk / sk'*A*sk */
        multiply_m_v(matrix, sk, pom1);
        ak = rho / multiply_v_v(sk, pom1);

        /*  rk_1 = rk - ak*A*sk */
        multiply_v_c(pom1, ak, pom1);
        subtract_v_v(rk, pom1, rk);

        /*  xk_1 = xk + ak*sk */
        multiply_v_c(sk, ak, pom1);
        addition_v_v(xk, pom1, xk);

        /*  time to end? */
        diff = gen_vector_size(rk);
        if (diff < EPSILON)
        {
            printf("stop at %d iteration\n", i);
            break;
        }

        /*  bk = rk_1' * rk_1 / rk' * rk */
        bk = rho;
        rho = multiply_v_v(rk, rk);
        bk = rho / bk;

        /*  sk_1 = rk_1 + bk*sk */
        multiply_v_c(sk, bk, sk);
        addition_v_v(rk, sk, sk);
    }

    free(sk);
    free(pom1);
    free(rk);

    return 1;
}

/* Jacobi method using LU decomposition*/
int jacobi_method_1(double *xk)
{
    /*  initialization vector x */
    double *Db = (double *)calloc(n, sizeof(double));
    double *rk = (double *)calloc(n, sizeof(double));
    double **DR = (double **)calloc(n, sizeof(double *));
    for (int i = 0; i < n; i++)
        DR[i] = (double *)calloc(n, sizeof(double));

    /* Db = D^(−1)*b */
    /* matrix = D^(−1)*(L + U ) */
    if ((get_Db(Db) != 1) || (get_DxR(DR) != 1))
    {
        printf("Solution not exist!\n");
        return -1;
    }

    double diff = 0;
    double lb = vector_size(vector);
    double old = 0;

    for (int i = 0; i < k; i++)
    {
        if ((i % 100) == 0 && i > 1)
        {
            printf("iteration %d, diff: %1.15e\n", i - 1, diff);
        }

        /*   xk_1 = Db - DR*xk */
        multiply_m_v(DR, xk, rk);
        addition_v_v(Db, rk, xk);

        /*  rk = b - A*xk */
        multiply_m_v(matrix, xk, rk);
        subtract_v_v(rk, vector, rk);

        diff = vector_size(rk) / lb;
        if (diff < EPSILON)
        {
            printf("stop at %d iteration\n", i);
            break;
        }
        old = diff;
    }
    return 1;
}

/*  Jacobi method   */
int jacobi_method(double *xk)
{

    double *rk = (double *)calloc(n, sizeof(double)); /*  residuum */
    double *y = (double *)calloc(n, sizeof(double));  /* help vector storage */

    double diff = 0;

    double lb = vector_size(vector); /*  vector size - for difference calculation    */

    /*  while diff is not smaller than epsilon or allowed steps are not overload */
    for (int ii = 0; ii < k; ii++)
    {
        /*  log info    */
        if ((ii % 100) == 0 && ii > 1)
            printf("iteration %d, diff: %1.15e\n", ii - 1, diff);

        /*  for each equation in matrix solve next aproximation */
        for (int i = 0; i < n; i++)
        {
            diff = vector[i];
            for (int j = 0; j < n; j++)
                diff -= matrix[i][j] * xk[j];
            diff /= matrix[i][i];
            y[i] += diff;
        }
        for (int i = 0; i < n; i++)
            xk[i] = y[i];

        /* rk = b - A*xk */
        multiply_m_v(matrix, xk, rk);
        subtract_v_v(vector, rk, rk);

        /*  time to end? */
        diff = vector_size(rk) / lb;
        if (diff <= EPSILON)
        {
            printf("stop at %d iteration\n", ii);
            break;
        }
    }
    free(rk);
    free(y);
    return 1;
}

/*  Gauss-Seibel method     */
int gauss_seibel_method(double *xk)
{

    double *rk = (double *)calloc(n, sizeof(double)); /*  residuum and help vector storage*/
    double diff = 0;

    double lb = vector_size(vector); /*  vector size - for difference calculation    */

    /*  while diff is not smaller than epsilon or allowed steps are not overload */
    for (int ii = 0; ii < k; ii++)
    {
        /*  log info    */
        if ((ii % 100) == 0 && ii > 1)
            printf("iteration %d, diff: %1.15e\n", ii - 1, diff);

        /*  for each equation in matrix solve next aproximation */
        for (int i = 0; i < n; i++)
        {
            diff = vector[i];
            for (int j = 0; j < n; j++)
                diff -= matrix[i][j] * xk[j];
            diff /= matrix[i][i];
            xk[i] += diff;
        }

        /* rk = b - A*xk */
        multiply_m_v(matrix, xk, rk);
        subtract_v_v(vector, rk, rk);

        /*  time to end?    */
        diff = vector_size(rk) / lb;
        if (diff <= EPSILON)
        {
            printf("stop at %d iteration\n", ii);
            break;
        }
    }
    free(rk);
    return 1;
}

int main(int argc, char *argv[])
{
    /*  default arguments */
    arguments.outfile = NULL; /*  default output file - no file */

    /*  input filenames */
    char const *matrix_input = argv[argc - 2];
    char const *vector_input = argv[argc - 1];

    /*  get arguments */
    argp_parse(&argp, argc, argv, 0, 0, &arguments);

    /*  print basic information about calculation */
    printf("Procces terminated after %d steps\n", k);
    printf("Method: %s.\n", method_to_string(arguments.method));
    printf("------------------------------------------\n");
    printf("READING:\n");
    printf("------------------------------------------\n");

    /*  read matrix */
    assert(read_matrix(matrix_input) == 1);
    printf("Matrix successfuly readed.\n");

    /* read vector */
    vector = (double *)calloc(n, sizeof(double));
    assert(read_vector(vector_input) == 1);
    printf("Vector successfuly readed.\n");
    // print_v(vector);

    /*  declare output vector */
    double *output = (double *)calloc(n, sizeof(double));

    /*  log inform */
    printf("------------------------------------------\n");
    printf("CALCULATION:\n");
    printf("------------------------------------------\n");

    int res = 0;

    /*  switch by method */
    switch (arguments.method)
    {
    case METHOD_GD:
        res = gradient_descent_method(output);
        break;
    case METHOD_CG:
        res = conjugate_gradient_method(output);
        break;
    case METHOD_JM:
        res = jacobi_method(output);
        break;
    case METHOD_J1:
        res = jacobi_method_1(output);
        break;
    case METHOD_GS:
        res = gauss_seibel_method(output);
        break;
    default:
        break;
    }

    if (res != 1)
    {
        printf("Calculation unsuccessful\n");
        return -1;
    }

    /*  log inform */
    printf("Calculation done.\n");

    /*  print result */
    if (arguments.outfile)
    {
        write_output(output, arguments.outfile);
        printf("Result at %s.\n", arguments.outfile);
    }
    else
    {
        printf("Result:\n");
        print_v(output);
    }

    /*  tidy up */
    tidy_up(output);

    return 0;
}
