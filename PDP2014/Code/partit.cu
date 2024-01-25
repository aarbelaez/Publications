/*
 *  Adaptive search - GPU Version
 *
 *
 * Please visit the https://pauillac.inria.fr/~diaz/adaptive/manual/index.html for a complete version of the original Adaptive Search code
 */

#include <stdio.h>
#include <stdlib.h>

//#include "ad_solver.h"



/*-----------*
 * Constants *
 *-----------*/

/*-------*
 * Types *
 *-------*/

/*------------------*
 * Global variables *
 *------------------*/

/*
static int size;		// copy of p_ad->size 
static int *sol;		// copy of p_ad->sol

static int size2;		// size / 2 

static int coeff;
static int sum_mid_x, cur_mid_x;
static long long sum_mid_x2, cur_mid_x2;
*/

__shared__ int size2;
__shared__ int coeff;
__shared__ int sum_mid_x;
__shared__ int cur_mid_x;
__shared__ long long sum_mid_x2;
__shared__ long long cur_mid_x2;

/*------------*
 * Prototypes *
 *------------*/


/*
 *  MODELING
 *
 *  NB: partit 32 and 48 are often very difficult, 
 *  we then use a restart limit (e.g. at 100 iters do a restart)
 *  with theses restarts it works well.
 */



/*
 *  SOLVE
 *
 *  Initializations needed for the resolution.
 */

__device__ void
Solve(int *winner)
{
  /*sol = p_ad->sol;
  size = p_ad->size;

  size2 = size / 2;*/

  size2 = p_ad.size / 2;


  if(isMaster) {
#ifdef GLOBALMEM
    p_ad.mark = (int*) malloc(SIZE * sizeof(int));
#endif
#ifdef PAR
    p_ad.list_ij2 = (int**) malloc( (THREADS+1) * sizeof(int*) );
    for(int i=0;i <= THREADS; i++) {
      int tmp = Div_Round_Up(SIZE, THREADS);
      if( i == THREADS ) tmp = THREADS;
      p_ad.list_ij2[i] = (int*) malloc( tmp * sizeof(int) ); 
      if(!p_ad.list_ij2[i]) {
#if __CUDA_ARCH__ >= 200
	printf("Error allocating memory!\n");
#endif
	asm("trap;");
      }
    }
#endif
  }
  __syncthreads();
  
  /*sum_mid_x = p_ad.data32[0];
  coeff = p_ad.data32[1];
  sum_mid_x2 = p_ad.data64[0];*/

  Ad_Solve(winner);
}




/*
 *  COST_OF_SOLUTION
 *
 *  Returns the total cost of the current solution.
 *  Also computes errors on constraints for subsequent calls to
 *  Cost_On_Variable, Cost_If_Swap and Executed_Swap.
 */

//for some reason abs fails with some big numbers???
__device__ int ABS(int x) {
  return (x<0)?-x:x;
}

__device__ int
Cost_Of_Solution(int should_be_recorded)
{
  int i;
  int r;
  int x;

  //cur_mid_x = cur_mid_x2 = 0;
  cur_mid_x = 0;
  cur_mid_x2 = 0;
  for(i = 0; i < size2; i++)
    {
      x = p_ad.sol[i];
      cur_mid_x += x;
      cur_mid_x2 += x * x;
    }
  r = coeff * ABS(sum_mid_x - cur_mid_x) + ABS(sum_mid_x2 - cur_mid_x2);

  /*int tmp =  abs(sum_mid_x - cur_mid_x);
  printf("abs1: %d\n", tmp);
  tmp = abs(sum_mid_x2 - cur_mid_x2);
  printf("abs2: %d\n", tmp);
  printf("R: %d\n",r);*/
  return r;
}


/*
 *  NEXT_I and NEXT_J
 *
 *  Return the next pair i/j to try (for the exhaustive search).
 */

__device__ int Next_I(int i)
{
  i++;
  return i < size2 ? i : p_ad.size;
}

__device__ int Next_J(int i, int j)
{
  return (j < 0) ? size2 : j + 1;
}




/*
 *  COST_IF_SWAP
 *
 *  Evaluates the new total cost for a swap.
 */

__device__ int
Cost_If_Swap(int current_cost, int i1, int i2)
{
  int xi1, xi12, xi2, xi22, cm_x, cm_x2, r;

#if 0				/* useless with customized Next_I and Next_J */
  if (i1 >= size2 || i2 < size2)
    return (unsigned) -1 >> 1;
#endif

  xi1 = p_ad.sol[i1];
  xi2 = p_ad.sol[i2];

  xi12 = xi1 * xi1;
  xi22 = xi2 * xi2;

  cm_x = cur_mid_x - xi1 + xi2;
  cm_x2 = cur_mid_x2 - xi12 + xi22;
  r = coeff * ABS(sum_mid_x - cm_x) + ABS(sum_mid_x2 - cm_x2);

  return r;
}




/*
 *  EXECUTED_SWAP
 *
 *  Records a swap.
 */

__device__ void
Executed_Swap(int i1, int i2)
{
  int xi1, xi12, xi2, xi22;

  if( isMaster ) {
    xi1 = p_ad.sol[i2];		/* swap already executed */
    xi2 = p_ad.sol[i1];
    
    xi12 = xi1 * xi1;
    xi22 = xi2 * xi2;
    
    cur_mid_x = cur_mid_x - xi1 + xi2;
    cur_mid_x2 = cur_mid_x2 - xi12 + xi22;
  }
}




//int param_needed = 1;		/* overwrite var of main.c */

/*
 *  INIT_PARAMETERS
 *
 *  Initialization function.
 */

/*void
Init_Parameters(AdData *p_ad)
{
  int size = p_ad->param;

  p_ad->size = size;

  if (size < 8 || size % 4 != 0)
    {
      printf("no solution with size = %d\n", size);
      exit(1);
    }

//  The sums are as follows:
//  sum_x = size * (size + 1) / 2
//  sum_x2 = size * (size + 1) * (2 * size + 1) / 6
//
//  We are interested in theses sums / 2 thus:
//

  int sum_mid_x = (size * (size + 1)) / 4;
  long long sum_mid_x2 = ((long long) sum_mid_x * (2 * size + 1)) / 3LL;
  int coeff = sum_mid_x2 / sum_mid_x;

  printf("mid sum x = %d,  mid sum x^2 = %lld, coeff: %d\n",
	 sum_mid_x, sum_mid_x2, coeff);

  p_ad->data32[0] = sum_mid_x;
  p_ad->data32[1] = coeff;
  p_ad->data64[0] = sum_mid_x2;


  p_ad->base_value = 1;
  p_ad->break_nl = size / 2;

  // defaults 

  if (p_ad->prob_select_loc_min == -1)
    p_ad->prob_select_loc_min = 80;

  if (p_ad->freeze_loc_min == -1)
    p_ad->freeze_loc_min = 1;

  if (p_ad->freeze_swap == -1)
    p_ad->freeze_swap = 0;

  if (p_ad->reset_limit == -1)
    p_ad->reset_limit = 1;

  if (p_ad->reset_percent == -1)
    p_ad->reset_percent = 1;

  if (p_ad->restart_limit == -1)
    p_ad->restart_limit = 100; // (size < 100) ? 10 : (size < 1000) ? 150 : size / 10;

  if (p_ad->restart_max == -1)
    p_ad->restart_max = 100000;
}
*/



/*
 *  CHECK_SOLUTION
 *
 *  Checks if the solution is valid.
 */

__device__ __host__ int
Check_Solution(int *sol, int size) 
{
  int i;
  //int size = p_ad->size;
  int size2 = size / 2;
  int sum_a = 0, sum_b = 0;
  long long sum_a2 = 0, sum_b2 = 0;

  for(i = 0; i < size2; i++)
    {
      sum_a += sol[i];
      sum_a2 += sol[i] * sol[i];
    }

  for(; i < size; i++)
    {
      sum_b += sol[i];
      sum_b2 += sol[i] * sol[i];
    }

  if (sum_a != sum_b)
    {
      printf("ERROR sum a: %d != sum b: %d\n", sum_a, sum_b);
      return 0;
    }

  if (sum_a2 != sum_b2)
    {
      printf("ERROR sum a^2: %lld != sum b: %lld\n", sum_a2, sum_b2);
      return 0;
    }
  
  printf("Solution OK!\n");
  return 1;
}


/*
 *  RESET
 *
 * Performs a reset (returns the new cost or -1 if unknown)
 */
__device__ int
Reset(int n)
{
  if( isMaster ) {
    int i, j, x;
    int size = p_ad.size;
    int *sol = p_ad.sol;

    while(n--)
      {
	i = Random(size);
	j = Random(size);

	p_ad.nb_swap++;

	x = p_ad.sol[i];
	p_ad.sol[i] = p_ad.sol[j];
	p_ad.sol[j] = x;

#if UNMARK_AT_RESET == 1
	Ad_Un_Mark(i);
	Ad_Un_Mark(j);
#endif
      }
  }
  //__syncthreads();

  return -1;
}

