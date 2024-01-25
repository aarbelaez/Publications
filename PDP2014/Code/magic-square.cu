
/*
 *  Adaptive search - GPU Version
 *
 *
 * Please visit the https://pauillac.inria.fr/~diaz/adaptive/manual/index.html for a complete version of the original Adaptive Search code
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//#include "ad_solver.h"


/*-----------*
 * Constants *
 *-----------*/

/*-------*
 * Types *
 *-------*/

/* for a k in 0..size-1 gives line, col, d1 (1/0), d2 (0/1) */

/*#if 1
#define CELL
#endif

#ifdef CELL

typedef struct
{
  short w1;
  short w2;
} XRef;


#define XSet(xr, line, col, diag1, diag2)   xr.w1 = (diag1 << 15) | line; xr.w2 = (diag2 << 15) | col
#define XGetL(xr)     (xr.w1 & 0x7f)
#define XGetC(xr)     (xr.w2 & 0x7f)
#define XIsOnD1(xr)   (xr.w1 < 0)
#define XIsOnD2(xr)   (xr.w2 < 0)

#elif 1 // BEST on x86_64

typedef unsigned int XRef;

#define XSet(xr, line, col, diag1, diag2)   xr = (diag1 << 31) | (col << 16) | (diag2 << 15) | line
#define XGetL(xr)     (xr & 0x7f)
#define XGetC(xr)     ((xr >> 16) & 0x7f)
#define XIsOnD1(xr)   ((int) xr < 0)
#define XIsOnD2(xr)   ((xr & 0x00008000) != 0)

#elif 0  // others...

typedef unsigned int XRef;

#define XSet(xr, line, col, diag1, diag2)   xr = (diag1 << 31) | (diag2 << 30) | (col << 15) | line
#define XGetL(xr)     (xr & 0x7f)
#define XGetC(xr)     ((xr >> 15) & 0x7f)
#define XIsOnD1(xr)   ((int) xr < 0)
#define XIsOnD2(xr)   ((xr & 0x40000000) != 0)

#elif 0*/
typedef struct
{
  unsigned int d1:1;
  unsigned int d2:1;
  unsigned int l:15;
  unsigned int c:15;
} XRef;


#define XSet(xr, line, col, diag1, diag2)   xr.l = line; xr.c = col; xr.d1 = diag1; xr.d2 = diag2
#define XGetL(xr)     xr.l
#define XGetC(xr)     xr.c
#define XIsOnD1(xr)   (xr.d1 != 0)
#define XIsOnD2(xr)   (xr.d2 != 0)

//#endif



/*------------------*
 * Global variables *
 *------------------*/

//int size;		/* copy of p_ad->size (square_length*square_length) */
//int *sol;		/* copy of p_ad->sol */

__shared__ int square_length;	/* side of the square */
__shared__ int square_length_m1;    /* square_length - 1 */
__shared__ int square_length_p1;    /* square_length + 1 */
__shared__ int avg;			/* sum to reach for each l/c/d */

//int *err_l, *err_l_abs;  /* errors on lines (relative + absolute) */
//int *err_c, *err_c_abs;	/* errors on columns */
//int err_d1, err_d1_abs; 	/* error on d1 (\) */
//int err_d2, err_d2_abs;	/* error on d2 (/) */

__shared__ int err_l[SIZE_S2];
__shared__ int err_l_abs[SIZE_S2];
__shared__ int err_c[SIZE_S2];
__shared__ int err_c_abs[SIZE_S2];
__shared__ int err_d1;
__shared__ int err_d1_abs;
__shared__ int err_d2;
__shared__ int err_d2_abs;
__shared__ XRef *xref;


__shared__ int sol_swap;
__shared__ int l_swap;
__shared__ int c_swap;

/*------------*
 * Prototypes *
 *------------*/


/*
 *  MODELING
 *
 *  sol[] = vector of values (by line),
 *             sol[0..square_length-1] contain the first line, 
 *             sol[square_length-2*square_length-1] contain the 2nd line, ...
 *             values are in 1..square_length*square_length
 *
 *  The constraints are: for each line, column, diagonal 1 (\) and 2 (/)
 *  the sum must be equal to avg = square_length * (square_length*square_length + 1) / 2;
 *
 *  err_l[i] = -avg + sum of line i
 *  err_c[j] = -avg + sum of column j
 *  err_d1   = -avg + sum of diagonal 1
 *  err_d2   = -avg + sum of diagonal 2
 *
 *                   square_length-1  square_length-1
 *  The total cost = Sum |err_l[i]| + Sum |err_c[i]| + |err_d1| + |err_d2|
 *                   i=0              j=0
 *
 *  The projection on a variable at i, j:
 *  // err_var[i][j] = | err_l[i] + err_c[j] + F1(i,j) + F2(i,j) |  SLOW version
 *  err_var[i][j] = | err_l[i] + err_c[j] + F1(i,j) + F2(i,j) |
 *  with F1(i,j) = err_d1 if i,j is on diagonal 1 (i.e. i=j) else = 0
 *  and  F2(i,j) = err_d2 if i,j is on diagonal 2 (i.e. j=square_length-1-i) else = 0
 */


/*
 *  SOLVE
 *
 *  Initializations needed for the resolution.
 */
__shared__ int gmem;
__device__ void * own_malloc(int size, int dtype) {
  //printf("Mem so far: %d KB\n", (gmem/1024));
  void *tmp = (void*)malloc(size * dtype);
  if(!tmp) {
    printf("Error allocating %d Bytes\n", (size *dtype) );
    printf("Mem used so far: %d\n", (gmem/1024));
    asm("trap;");
  }
  gmem+=(size*dtype);
  return tmp;
}


#define CHECK_MEM(var) {if(!var) {printf("Block: %d, Thread: %d, Error allocating, mem used so far: %d\n", blockIdx.x, threadIdx.x, gmem/1024); }}

__device__ void
Solve(int *winner)
{
  int i, j, k;
  XRef xr;

  /*
  sol = p_ad->sol;
  size = p_ad->size;
  
  square_length = p_ad->param;
  square_length_m1 = square_length - 1;
  square_length_p1 = square_length + 1;
  */
  
  if( isMaster ) {
      gmem=0;
      p_ad.size = SIZE;
      square_length=SIZE_S2;
      square_length_m1 = square_length - 1;
      square_length_p1 = square_length + 1;

      p_ad.data32[0] = square_length * (p_ad.size + 1) / 2;
      avg = p_ad.data32[0];

      //if (err_l == NULL)
      //{
	  /*err_l = (int *) malloc(square_length * sizeof(int));
	  err_c = (int *) malloc(square_length * sizeof(int));
	  err_l_abs = (int *) malloc(square_length * sizeof(int));
	  err_c_abs = (int *) malloc(square_length * sizeof(int));*/
      xref = (XRef *) own_malloc(p_ad.size, sizeof(XRef));
	  
      p_ad.sol = (int*) own_malloc(p_ad.size,  sizeof(int));
      p_ad.mark = (int*) own_malloc(p_ad.size, sizeof(int));
      p_ad.list_i = (int *) own_malloc(p_ad.size, sizeof(int));
      p_ad.list_j = (int *) own_malloc(p_ad.size, sizeof(int));
      
      p_ad.list_i2 = (int**) own_malloc( (THREADS+1), sizeof(int*));
      p_ad.list_j2 = (int**) own_malloc( (THREADS+1), sizeof(int*));

      CHECK_MEM(xref); CHECK_MEM(p_ad.sol); CHECK_MEM(p_ad.mark); 
      CHECK_MEM(p_ad.list_i); CHECK_MEM(p_ad.list_j); CHECK_MEM(p_ad.list_i2);
      CHECK_MEM(p_ad.list_j2);
      if(!p_ad.list_i2 || !p_ad.list_j2 || !xref || !p_ad.sol || !p_ad.mark || !p_ad.list_i || !p_ad.list_j) {
	printf("Error allocating memory \n");
	asm("trap;");
      }
      if( blockIdx.x == 0 ) {
	printf("SIZE: %d, THREADS: %d\n",p_ad.size, THREADS);
	printf("square_length: %d\n",square_length);
	printf("size: %d, square_length: %d, square_length_m1 %d avg: %d\n", p_ad.size, square_length, square_length_m1,avg);
      }
      for(i=0;i<=THREADS;i++) {
	int tmp;
	if(i==THREADS) tmp = THREADS;
	else tmp = Div_Round_Up(SIZE, THREADS);
	p_ad.list_i2[i]= (int*) own_malloc(tmp, sizeof(int));
	p_ad.list_j2[i]= (int*) own_malloc(tmp, sizeof(int));
	CHECK_MEM(p_ad.list_i2[i]);
	CHECK_MEM(p_ad.list_j2[i]);
      }

      printf("Block[%d], Mem: %d KB\n", blockIdx.x, (gmem/1024));
      //printf("Block[%d], pointer: %p\n", blockIdx.x, p_ad.list_j2);
      for(k = 0; k < p_ad.size; k++)
      {
	  i = k / square_length;
	  j = k % square_length;

	  XSet(xr, i, j, (i == j), (i + j == square_length_m1));

	  xref[k] = xr;
      }

  }
  __syncthreads();
  Ad_Solve(winner);
}


__device__ void
Free_Mem(void) {
  free( xref );
  free( p_ad.sol );
  free( p_ad.mark );
  free( p_ad.list_i );
  free( p_ad.list_j );
  for(int i=0;i<=THREADS;i++) {
    free(p_ad.list_i2[i]);
    free(p_ad.list_j2[i]);
  }
  free( p_ad.list_i2 );
  free( p_ad.list_j2 );
}

/*
 *  COST_OF_SOLUTION
 *
 *  Returns the total cost of the current solution.
 *  Also computes errors on constraints for subsequent calls to
 *  Cost_On_Variable, Cost_If_Swap and Executed_Swap.
 */

__device__ int
Cost_Of_Solution(int should_be_recorded)
{
  int k;
  int neg_avg = -avg;
  unsigned int r;

  err_d1 = err_d2 = neg_avg;

  /*memset(err_l, 0, sizeof(int) * square_length);
    memset(err_c, 0, sizeof(int) * square_length);*/
  for(int i=0;i<square_length;i++) {
    err_l[i]=0;
    err_c[i]=0;
  }
  
  k = 0;
  do
    {
      XRef xr = xref[k];
      
      err_l[XGetL(xr)] += p_ad.sol[k];
      err_c[XGetC(xr)] += p_ad.sol[k];

      //printf("k: %d, err_l: %d, err_c: %d-- GetL: %d, GetC: %d -- sol: %d\n", k, err_l[XGetL(xr)], err_c[XGetC(xr) ], (XGetL(xr)), (XGetC(xr)),  p_ad.sol[k]);

    }
  while(++k < p_ad.size);
  int k1 = 0, k2 = 0;
  do
    {
      k2 += square_length_m1;
      err_d1 += p_ad.sol[k1];
      err_d2 += p_ad.sol[k2];
      
      k1 += square_length_p1;
    }
  while(k1 < p_ad.size);
  
  err_d1_abs = abs(err_d1);
  err_d2_abs = abs(err_d2);
  
  r = err_d1_abs + err_d2_abs;
  //printf("r: %u \nerr d1 %d -- err d2 %d\n", r, err_d1_abs, err_d2_abs);
  //printf("k1: %d k2: %d --- err_d1: %d err_d2: %d\n", k1, k2, err_d1, err_d2);

  k = 0;
  do
    {
      //printf("err_l: %d, err_c: %d\n", err_l[k], err_c[k]);
      err_l[k] -= avg; err_l_abs[k] = abs(err_l[k]); r += err_l_abs[k];
      err_c[k] -= avg; err_c_abs[k] = abs(err_c[k]); r += err_c_abs[k];
      //printf("k[%d] r: %d\n",k ,r);
    }
  while(++k < square_length);
  return r;
}



/*
 *  COST_ON_VARIABLE
 *
 *  Evaluates the error on a variable.
 */

//__device__ int Cost_On_Variable(int k) { return k+1; }
__device__ int
Cost_On_Variable(int k)
{
  //XRef xr = xref[k];
  int r;
  int line, col, diag1, diag2;
  
  int i, j;
  
  i = k / square_length;
  j = k % square_length;
  
  line = i;
  col  = j;
  diag1 = (i == j);
  diag2 = (i + j == square_length_m1); 
  
  /*
#ifndef SLOW

  r = err_l_abs[XGetL(xr)] + err_c_abs[XGetC(xr)] + 
    (XIsOnD1(xr) ? err_d1_abs : 0) + 
    (XIsOnD2(xr) ? err_d2_abs : 0);
  
#else  // less efficient use it with -f 5 -p 10 -l (ad.size/4)+1

  r = err_l[XGetL(xr)] + err_c[XGetC(xr)] + 
    (XIsOnD1(xr) ? err_d1 : 0) + 
    (XIsOnD2(xr) ? err_d2 : 0);
    
  r = abs(r);
#endif
  */
  
   r = err_l_abs[ line ] + err_c_abs[ col ] +
    (diag1 != 0 ? err_d1_abs : 0) +
    (diag2 != 0 ? err_d2_abs : 0);

  r = abs(r);

  
  return r;
}


/*
 *  COST_IF_SWAP
 *
 *  Evaluates the new total cost for a swap.
 */

#define AdjustL(r, diff, k)   r = r - err_l_abs[k] + abs(err_l[k] + diff)
#define AdjustC(r, diff, k)   r = r - err_c_abs[k] + abs(err_c[k] + diff)
#define AdjustD1(r, diff)     r = r - err_d1_abs   + abs(err_d1   + diff)
#define AdjustD2(r, diff)     r = r - err_d2_abs   + abs(err_d2   + diff)

__device__ int
Cost_If_Swap(int current_cost, int k1, int k2)
{
  /*
  XRef xr1 = xref[k1];
  XRef xr2 = xref[k2];
  int l1 = XGetL(xr1);
  int c1 = XGetC(xr1);
  int l2 = XGetL(xr2);
  int c2 = XGetC(xr2);
  */
  int l1, l2, c1, c2;
  l1 = k1 / square_length;
  l2 = l_swap;
  //l2 = k2 / square_length;
  c1 = k1 % square_length;
  //c2 = k2 % square_length;
  c2 = c_swap;
  int d11, d12, d21, d22;
  d11 = (l1 == c1); d12 = ( l1 + c1 == square_length_m1);
  d21 = (l2 == c2); d22 = ( l2 + c2 == square_length_m1);
  
   
  int diff1, diff2, r;
  
  r = current_cost;

  //diff1 = p_ad.sol[k2] - p_ad.sol[k1];
  diff1 = sol_swap - p_ad.sol[k1];
  diff2 = -diff1;

  if (l1 != l2)			/* not on the same line */
    {
      AdjustL(r, diff1, l1);
      AdjustL(r, diff2, l2);
    }

  if (c1 != c2)			/* not on the same column */
    {
      AdjustC(r, diff1, c1);
      AdjustC(r, diff2, c2);
    }

  /*
  if (XIsOnD1(xr1))	       // only one of both is on diagonal 1 
    {
      if (!XIsOnD1(xr2))
	AdjustD1(r, diff1);
    }
  else if (XIsOnD1(xr2))
    {
      AdjustD1(r, diff2);
    }
  if (XIsOnD2(xr1))		// only one of both is on diagonal 2 
    {
      if (!XIsOnD2(xr2))
	AdjustD2(r, diff1);
    }
  else if (XIsOnD2(xr2))
    {
      AdjustD2(r, diff2);
    }
  */
  if( d11 != 0 ) {
    if( !(d21 != 0) )
      AdjustD1(r, diff1); 
  }
  else if ( d21 != 0 ) {
    AdjustD1(r, diff2);
  }
  if( d12 != 0 ) {
    if ( !(d22 != 0) )
      AdjustD2(r, diff1);
  }
  else if( d22 != 0 ) {
    AdjustD2(r, diff2);
  }

  return r;
}




/*
 *  EXECUTED_SWAP
 *
 *  Records a swap.
 */

__device__ void
Executed_Swap(int k1, int k2)
{
  if( isMaster ) {

  XRef xr1 = xref[k1];
  XRef xr2 = xref[k2];
  int l1 = XGetL(xr1);
  int c1 = XGetC(xr1);
  int l2 = XGetL(xr2);
  int c2 = XGetC(xr2);
  int diff1, diff2;
  
  diff1 = p_ad.sol[k1] - p_ad.sol[k2]; /* swap already executed */
  diff2 = -diff1;

  err_l[l1] += diff1; err_l_abs[l1] = abs(err_l[l1]);
  err_l[l2] += diff2; err_l_abs[l2] = abs(err_l[l2]);
  
  err_c[c1] += diff1; err_c_abs[c1] = abs(err_c[c1]);
  err_c[c2] += diff2; err_c_abs[c2] = abs(err_c[c2]);
  
  if (XIsOnD1(xr1))
    {
      err_d1 += diff1;
      err_d1_abs = abs(err_d1);
    }

  if (XIsOnD1(xr2))
    {
      err_d1 += diff2;
      err_d1_abs = abs(err_d1);
    }

  if (XIsOnD2(xr1))
    {
      err_d2 += diff1;
      err_d2_abs = abs(err_d2);
    }


  if (XIsOnD2(xr2))
    {
      err_d2 += diff2;
      err_d2_abs = abs(err_d2);
    }



#if 0
  printf("----- after swapping %d and %d", k1, k2);
  int i;
  for(i = 0; i < size; i++)
    {
      if (i % square_length == 0)
	printf("\n");
      printf(" %d", sol[i]);
      
    }
  printf("\n");
  printf("err_lin:");
  for(i = 0; i < square_length; i++)
    printf(" %d", err_l[i]);
  printf("\n");

  printf("err_col:");
  for(i = 0; i < square_length; i++)
    printf(" %d", err_c[i]);
  printf("\n");

  printf("err_d1: %d   err_d2:%d\n", err_d1, err_d2);
  printf("----------------------------------------------\n");
#endif

  }
}




__device__ int param_needed = 1;		/* overwrite var of main.c */


/*
 *  INIT_PARAMETERS
 *
 *  Initialization function.
 */

/*void
Init_Parameters(int seed)
{
  //int square_length = p_ad->param;

  //p_ad->size = square_length * square_length;

  if( isMaster ) {
      p_ad.size = SIZE;
      square_length = SIZE_S2;
      p_ad.nb_var_to_reset = -1;
      
      int avg = square_length * (p_ad.size + 1) / 2;
      printf("sum of each line/col/diag = %d\n", avg);
      
      p_ad.prob_select_loc_min = 50;
      p_ad.freeze_loc_min = 1;
      p_ad.freeze_swap = 0;
      p_ad.reset_limit = 1;
      //p_ad.reset_percent = 5;
      p_ad.restart_limit = 1000000000;
      p_ad.restart_max = 0;
      p_ad.first_best = 0;

      p_ad.base_value = 1;


      p_ad.data32[0] = avg;


      p_ad.base_value = 1;
      p_ad.break_nl = square_length;
				
      if (p_ad.prob_select_loc_min == -1)
	  p_ad.prob_select_loc_min = 6;

      if (p_ad.freeze_loc_min == -1)
	  p_ad.freeze_loc_min = 1;

      if (p_ad.freeze_swap == -1)
	  p_ad.freeze_swap = 0;

      if (p_ad.reset_limit == -1)
	  p_ad.reset_limit = square_length * 1.2;

      if (p_ad.reset_percent == -1)
	  p_ad.reset_percent = 25;

      if (p_ad.restart_limit == -1)
	  p_ad.restart_limit = 10000000;

      if (p_ad.restart_max == -1)
	  p_ad.restart_max = 0;
      
      
      curand_init(p_ad.seed,0, 0, &global_cudaRand);
      int i;
      for(i=0;i<=blockIdx.x;i++) {
	  p_ad.seed=(int)(curand_uniform(&global_cudaRand) * 100000.0);
      }
      curand_init(p_ad.seed, 0, 0, &cudaRand);
  }
  __syncthreads();
}*/



/*
 *  CHECK_SOLUTION
 *
 *  Checks if the solution is valid.
 */

__device__ __host__ int
Check_Solution(int *sol, int size)
{
  //int square_length = p_ad->param;
  //int *sol = p_ad->sol;
  
  int square_length = SIZE_S2;
  int avg = square_length * (size + 1) / 2;
  int i, j;
  int sum_d1 = 0, sum_d2 = 0;
  

  for(i = 0; i < square_length; i++)
    {
      sum_d1 += sol[i * (square_length + 1)];
      sum_d2 += sol[(i + 1) * (square_length - 1)];
      int sum_l = 0, sum_c = 0;

      for(j = 0; j < square_length; j++)
        {
          sum_l += sol[i * square_length + j];
          sum_c += sol[j * square_length + i];
        }

      if (sum_l != avg)
	{
	  printf("ERROR line %d, sum: %d should be %d\n", i, sum_l, avg);
	  return 0;
	}

      if (sum_c != avg)
	{
	  printf("ERROR column %d, sum: %d should be %d\n", i, sum_c, avg);
	  return 0;
	}
    }


  if (sum_d1 != avg)
    {
      printf("ERROR column 1 (\\), sum: %d should be %d\n", sum_d1, avg);
      return 0;
    }

  if (sum_d2 != avg)
    {
      printf("ERROR column 2 (/), sum: %d should be %d\n", sum_d2, avg);
      return 0;
    }

  printf("SOLUTION OK\n");
  return 1;
}

void Display_Solution(int *sol, int size) {
    printf("Ad_Display\n");
  int k = 0, n;
    
    for(int i=0;i<size;i++) {
	for(int j=0;j<size;j++) {
	    printf("%d ",sol[k]);
	    k++;
	}
	printf("\n");
    }

}
