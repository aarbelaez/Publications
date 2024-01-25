
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


#if 0
#define SLOW
#endif

#if 1
#define NO_TRIVIAL		/* define it to reduce de number of trivial sols */
#endif


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
*/

//static int *nb_occ;		/* nb occurrences (to compute total cost) 0 is unused */

__shared__ int nb_occ[SIZE];



/*------------*
 * Prototypes *
 *------------*/


/*
 *  MODELING
 */

__device__ int
Is_Trivial_Solution(void)
{
  return (p_ad.sol[0] == 0 || p_ad.sol[0] == p_ad.size - 1 || p_ad.sol[p_ad.size - 1] == 0 || p_ad.sol[p_ad.size - 1] == p_ad.size - 1);
}


/*
 *  SOLVE
 *
 *  Initializations needed for the resolution.
 */

__device__ void
Solve(int *winner)
{
  /*
  sol = p_ad->sol;
  size = p_ad->size;

  if (nb_occ == NULL)
    {
      nb_occ = (int *) malloc(size * sizeof(int));

      if (nb_occ == NULL)
	{
	  printf("%s:%d malloc failed\n", __FILE__, __LINE__);
	  exit(1);
	}
    }
  */
  Ad_Solve(winner);
}


/*
 *  COST
 *
 *  Computes a cost associated to the array of occurrences.
 */

__device__ int
Cost(void)

{

#ifndef PAR
  //#ifndef SLOW
  //#ifdef SLOW
  //asm("trap;");
  int i = p_ad.size;

  nb_occ[0] = 0;                /* 0 is unused, use it as a sentinel */

  while(nb_occ[--i])
    ;

  return i;

#else  // less efficient (use it with -p 5 -f 4 -l 2 -P 80)


  /*
#ifndef PAR

    int r = 0, i;
    
    for(i = 1; i < p_ad.size; i++)
      if (nb_occ[i] == 0) {
	r += i;
      }
    return r;
#else
  */
  int tid = threadIdx.x;


  /*  Buffer_Vals[0] = 0;
  __syncthreads();
  if(tid < SIZE)
    Buffer_Vals[tid] = (nb_occ[tid]==0)?tid:0;
    
  //for(unsigned int s=1; s < p_ad.size; s++) 
  //for(int s=p_ad.size/2; s > 0; s >>= 1) {
  //for(int s=Div_Round_Up(p_ad.size,2); s > 1; s=Div_Round_Up(s,2) ) {

  for(int s=THREADS/2; s>0; s>>=1) {
    //if(isMaster) printf("SSS: %d\n",s);
    __syncthreads();
    if( tid < s && tid < p_ad.size && tid + s < p_ad.size) {
      //printf("SSS: %d - [%d,%d] -> (%d, %d)\n", s, tid, tid+s, Buffer_Vals[tid], Buffer_Vals[tid+s]);
      Buffer_Vals[tid]+=Buffer_Vals[tid+s];
    }

  }
  __syncthreads();
  //if(p_ad.nb_iter > 5 && isMaster) printf("RR: %d\n",Buffer_Vals[0]);
  */
  
  //if( Buffer_Vals[0] < (p_ad.size - 50) ) {
  if( Buffer_Vals[0] < p_ad.size) {
    Buffer_Vals[0] = 0;
    __syncthreads();
    if(tid < SIZE)
      Buffer_Vals[tid] = (nb_occ[tid]==0?tid:-1);
    
    for(int s=THREADS/2; s>0; s>>=1) {
      __syncthreads();
      if( tid < s && tid < p_ad.size && tid + s < p_ad.size ) {
	//printf("TID[%d]: %d -- TID2[%d]: %d --(%d, %d)\n", tid, nb_occ[ Buffer_Vals[tid]], (tid+s), nb_occ[ Buffer_Vals[tid+s] ], tid, (tid+s) );
	if(Buffer_Vals[tid] < Buffer_Vals[tid+s])
	  Buffer_Vals[tid] = Buffer_Vals[tid+s];
      }
    }
    __syncthreads();
    
    /*if( isMaster ) {
      int i = p_ad.size;
      nb_occ[0] = 0;
      while(nb_occ[--i]);
      if(Buffer_Vals[0] != i) {
      printf("Error, i[%d] %d, buffer[%d]: %d\n", i, nb_occ[i], Buffer_Vals[0], nb_occ[Buffer_Vals[0]]);
      asm("trap;");
      }
      //else {printf("OKKKKKK\n");}
      //printf("Cost: %d\n", Buffer_Vals[0]);
      }
      __syncthreads();*/
  }
  else {
    if( isMaster ) {
      Buffer_Vals[0] = p_ad.size;
      nb_occ[0] = 0;
      while( nb_occ[ --Buffer_Vals[0] ] );
    }
    __syncthreads();
  }
  return Buffer_Vals[0];
  //#endif

#endif
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
  int i;

  //memset(nb_occ, 0, size * sizeof(int));
  for(i=0; i<p_ad.size;i++) nb_occ[i]=0;
  
  for(i = 0; i < p_ad.size - 1; i++)
    nb_occ[abs(p_ad.sol[i] - p_ad.sol[i + 1])]++;

#ifdef NO_TRIVIAL
  if (Is_Trivial_Solution())
    return p_ad.size;
#endif

  return Cost_Seq();
}


__device__ int
Cost_Seq(void)

{
#ifndef SLOW
  int i = p_ad.size;

  nb_occ[0] = 0;                /* 0 is unused, use it as a sentinel */

  while(nb_occ[--i])
    ;

  return i;

#else  // less efficient (use it with -p 5 -f 4 -l 2 -P 80)

  int r = 0, i;
  
  for(i = 1; i < p_ad.size; i++)
    if (nb_occ[i] == 0)
      r += i;
  return r;

#endif
}


/*
 *  COST_IF_SWAP
 *
 *  Evaluates the new total cost for a swap.
 */

__device__ int
Cost_If_Swap(int current_cost, int i1, int i2)
{
  int s1, s2;
  int rem1, rem2, rem3, rem4;
  int add1, add2, add3, add4;

#ifdef NO_TRIVIAL
  if ((i1 == 0 && (p_ad.sol[i2] == 0 || p_ad.sol[i2] == p_ad.size - 1)) ||
      (i2 == 0 && (p_ad.sol[i1] == 0 || p_ad.sol[i1] == p_ad.size - 1)))
    return p_ad.size;
#endif

				/* we know i1 < i2 due to ad.exhaustive */
				/* else uncomment this */
#if 0
  if (i1 > i2)
    {
      i = i1;
      i1 = i2;
      i2 = i;
    }
#endif

  if( isMaster ) {

    s1 = p_ad.sol[i1];
    s2 = p_ad.sol[i2];
    
    if (i1 > 0)
      {
	rem1 = abs(p_ad.sol[i1 - 1] - s1); nb_occ[rem1]--; 
	add1 = abs(p_ad.sol[i1 - 1] - s2); nb_occ[add1]++; 
      }
    else
      rem1 = add1 = 0;
    
    
    if (i1 < i2 - 1)		/* i1 and i2 are not consecutive */
      {
	rem2 = abs(s1 - p_ad.sol[i1 + 1]); nb_occ[rem2]--; 
	add2 = abs(s2 - p_ad.sol[i1 + 1]); nb_occ[add2]++; 
	
	rem3 = abs(p_ad.sol[i2 - 1] - s2); nb_occ[rem3]--; 
	add3 = abs(p_ad.sol[i2 - 1] - s1); nb_occ[add3]++; 
      }
    else
      rem2 = add2 = rem3 = add3 = 0;

    if (i2 < p_ad.size - 1)
      {
	rem4 = abs(s2 - p_ad.sol[i2 + 1]); nb_occ[rem4]--;
	add4 = abs(s1 - p_ad.sol[i2 + 1]); nb_occ[add4]++;
      }
    else
      rem4 = add4 = 0;

  }

  int r = Cost();
  
  __syncthreads();
  /* undo */
  if( isMaster ) {
    nb_occ[rem1]++; nb_occ[rem2]++; nb_occ[rem3]++; nb_occ[rem4]++; 
    nb_occ[add1]--; nb_occ[add2]--; nb_occ[add3]--; nb_occ[add4]--;
  }

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
  int s1, s2;
  int rem1, rem2, rem3, rem4;
  int add1, add2, add3, add4;

				/* we know i1 < i2 due to ad.exhaustive */
				/* else uncomment this */
#if 0
  if (i1 > i2)
    {
      int i = i1;
      i1 = i2;
      i2 = i;
    }
#endif

  if( isMaster ) {

    s1 = p_ad.sol[i2];			/* swap already executed */
    s2 = p_ad.sol[i1];
    
    if (i1 > 0)
      {
	rem1 = abs(p_ad.sol[i1 - 1] - s1); nb_occ[rem1]--; 
	add1 = abs(p_ad.sol[i1 - 1] - s2); nb_occ[add1]++; 
      }
    
    
    if (i1 < i2 - 1)              /* i1 and i2 are not consecutive */
      {
	rem2 = abs(s1 - p_ad.sol[i1 + 1]); nb_occ[rem2]--; 
	add2 = abs(s2 - p_ad.sol[i1 + 1]); nb_occ[add2]++; 
	
	rem3 = abs(p_ad.sol[i2 - 1] - s2); nb_occ[rem3]--; 
	add3 = abs(p_ad.sol[i2 - 1] - s1); nb_occ[add3]++; 
      }

    if (i2 < p_ad.size - 1)
      {
	rem4 = abs(s2 - p_ad.sol[i2 + 1]); nb_occ[rem4]--;
	add4 = abs(s1 - p_ad.sol[i2 + 1]); nb_occ[add4]++;
      }
  }
}


/*
 *  RESET
 *
 * Performs a reset (returns the new cost or -1 if unknown)
 */

//#ifdef NO_TRIVIAL  // not defining Reset() is slower but produces a bit less trivial sols
__device__ int
Reset(int n)
{
  if( isMaster ) {

  int dist_min = p_ad.size - 3;	// size - 1 also works pretty well 
  int i, j;
      
  for(i = 1; i < p_ad.size; i++)
    {
      if (abs(p_ad.sol[i - 1] - p_ad.sol[i]) >= dist_min)
	{
	  j = Random(p_ad.size);
	  Ad_Swap(i, j);
	}
    }

  }

  return -1;

}
//#endif


//__device__ int param_needed = 1;		// overwrite var of main.c 

/*char *user_stat_name = "trivial";
  int (*user_stat_fct)(AdData *p_ad) = Trivial_Statistics;*/

/*
 *  INIT_PARAMETERS
 *
 *  Initialization function.
 */

/*void
Init_Parameters(AdData *p_ad)
{
  p_ad->size = p_ad->param;

#ifndef SLOW
  p_ad->first_best = 1;
#endif

				// defaults 
  if (p_ad->prob_select_loc_min == -1)
    p_ad->prob_select_loc_min = 66;

  if (p_ad->freeze_loc_min == -1)
    p_ad->freeze_loc_min = 1;

  if (p_ad->freeze_swap == -1)
    p_ad->freeze_swap = 0;

  if (p_ad->reset_limit == -1)
    p_ad->reset_limit = 1;

  if (p_ad->reset_percent == -1)
    p_ad->reset_percent = 25;

  if (p_ad->restart_limit == -1)
    p_ad->restart_limit = 10000000;

  if (p_ad->restart_max == -1)
    p_ad->restart_max = 0;
    }*/


/*
 *  CHECK_SOLUTION
 *
 *  Checks if the solution is valid.
 */

__device__ __host__ int
Check_Solution(int *sol, int size)
{
  int r = 1;
  int i;

  int sol_ok=1;

  int *nb_occ1 = NULL;
  if (nb_occ1 == NULL)
    {
      nb_occ1 = (int *) malloc(size * sizeof(int));
      if (nb_occ1 == NULL)
	{
	  printf("%s:%d malloc failed\n", __FILE__, __LINE__);
	  //exit(1);
	}
    }

  memset(nb_occ1, 0, size * sizeof(int));

  for(i = 0; i < size - 1; i++) {
    if(sol[i]<0) {
      printf("Negative value in the solution\n");
      sol_ok=0;
      break;
    }
    nb_occ1[abs(sol[i] - sol[i + 1])]++;
  }

  for(i = 1; i < size; i++)
    if (nb_occ1[i] > 1)
      {
	printf("ERROR distance %d appears %d times\n", i, nb_occ1[i]);
	r = 0;
	sol_ok=0;
      }
  free( nb_occ1 );
  if( sol_ok )
    printf("Solution OK\n");

  return r;
}
