/*
 *  Adaptive search - GPU Version
 *
 *
 * Please visit the https://pauillac.inria.fr/~diaz/adaptive/manual/index.html for a complete version of the original Adaptive Search code
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include<curand_kernel.h>
#include <time.h>

#include "mrand.cu"


#define BIG ((unsigned int) -1 >> 1) 

#define BASE_MARK    ((unsigned) p_ad.nb_swap)
//#define Mark(i, k)   p_ad.mark[i] = BASE_MARK + (k)
#define Mark(i,k) {}
#define UnMark(i)    p_ad.mark[i] = 0
#define Marked(i)    (BASE_MARK + 1 <= p_ad.mark[i])

#define USE_PROB_SELECT_LOC_MIN ((unsigned) p_ad.prob_select_loc_min <= 100)

#define Div_Round_Up(x, y)   (((x) + (y) - 1) / (y))

#ifndef SIZE
#define SIZE 10
#endif

#ifndef THREADS
#define THREADS 2
#endif

//#define Arrayaccess(a, i, j) ((a)[(i) * SIZE + (j)])
#define Arrayaccess(i,j) ( (i) * SIZE + (j) )

typedef struct
{
  int i, j;
}Pair;

#ifndef GLOBALMEM //using global memory to store some variables
#define GLOBALMEM
#endif

typedef struct {
    int max_i;
    int min_j;
    int new_cost;
    int best_cost;
     //int *mark;
#ifndef GLOBALMEM
    int mark[SIZE];
#else
    int *mark;
#endif
    int nb_var_marked;
    int size;
    int seed;

    //int list_i[SIZE];
    int *list_i;
    int list_i_nb;

    //int list_j[SIZE];
    int *list_j;
    int list_j_nb;

    //int list_i2_nb[THREADS];
    //int list_j2_nb[THREADS];
    /*int *list_j2[THREADS];
    int *list_i2[THREADS];*/
    //int **list_j2;
    //int **list_i2;
#ifndef PAR
    int list_ij_nb;
    Pair list_ij[ SIZE ];
#else
    int list_ij2_nb[THREADS];
    int **list_ij2;
#endif
    int sol[SIZE];
    //int *sol;
    int total_cost;
    
    int nb_iter;
    int nb_iter_tot;
    int restart_limit;
    int nb_restart;
    int nb_swap;
    int nb_swap_tot;
    int swap;
    int nb_same_var;
    int nb_local_min;
    int nb_local_min_tot;
    int freeze_loc_min;
    int reset_limit;
    int nb_var_to_reset;
    int freeze_swap;
    int restart_max;
    int restart;
    int prob_select_loc_min;
    int first_best;
    int nb_reset;
    int nb_reset_tot;
    int reset_percent;
    
    int base_value;

    int data32[4];
    long long data64[2];          /* some 64 bits  */

} AdData;

__shared__ AdData p_ad;


__device__ void Select_Var_Min_Conflict_Parallel(void); 

__device__ void Select_Var_High_Cost(void);
__device__ void Select_Var_Min_Conflict(void); 


__device__ void Select_Var_High_Cost_Par(void);
__device__ void Select_Var_Min_Conflict_Par(void); 


__device__ void Select_Vars_To_Swap(void);
__device__ void Select_Vars_To_Swap_Par(void);

__device__ void Do_Reset(int);
__device__ int Reset(int);

__device__ int Cost(void);
__device__ int Cost_Seq(void);
__device__ int Cost_Of_Solution(int);
__device__ int Cost_On_Variable(int);
__device__ void Executed_Swap(int, int);
__device__ int Cost_If_Swap(int, int, int);

__device__ void Ad_Swap(int, int);

__device__ int Ad_Solve(int *);
__device__ void print_stat(void);

__shared__ int Buffer_Vals[THREADS];

#define isMaster  (threadIdx.x == 0)
//#define isMaster (tid == 0)


//#include "cap_tmp.cu"
//#include "magic-square.cu"
//#include "all-interval.cu"
#include "partit.cu"


__shared__ int do_reset;

__shared__ int use_local_min;
__shared__ int stop_var_sel;
__shared__ int list_len_j;
__shared__ int list_len_i;
__shared__ int buffer_tmp;

__device__ int Ad_Solve(int *winner) {

    int nb_in_plateau, best_of_best;
    
    if( isMaster ) {
      p_ad.nb_iter_tot = 0;
      p_ad.nb_swap_tot = 0;
      p_ad.nb_reset_tot = 0;
      p_ad.nb_restart = 0;
      
      p_ad.nb_iter = 0;
      p_ad.nb_swap = 0;
      p_ad.swap = 0;
      p_ad.nb_same_var = 0;
      p_ad.nb_local_min = 0;
      list_len_i = 0;
    }

restart:
    __syncthreads();
    if( isMaster ) {
      p_ad.nb_iter_tot+=p_ad.nb_iter;
      p_ad.nb_swap_tot+=p_ad.nb_swap;
      p_ad.nb_reset_tot+=p_ad.nb_reset;
      p_ad.nb_local_min_tot+=p_ad.nb_local_min;

      p_ad.nb_restart++;

      p_ad.nb_iter = 0;
      p_ad.nb_swap = 0;
      p_ad.swap = 0;
      p_ad.nb_same_var = 0;
      p_ad.nb_local_min = 0;
      list_len_i = 0;
    }
    //printf("x: %d\n",threadIdx.x);
    
#if __CUDA_ARCH__ >=200
    if( isMaster && blockIdx.x == 0 && p_ad.nb_restart==1) {
      printf("freeze local min: %d\nfreeze_swap: %d\n", p_ad.freeze_loc_min, p_ad.freeze_swap);
      printf("reset limit: %d\n", p_ad.reset_limit);
      printf("reset perc: %d\n", p_ad.reset_percent);
    }
#endif

    if( isMaster ) {
      int i;
      Random_Permut(p_ad.sol, p_ad.size, p_ad.base_value);
      //for(i=0;i<p_ad.size;i++) p_ad.sol[i]=i+1;

      //the number partition problem does not require a tabu list
      //for(i=0;i<p_ad.size;i++) p_ad.mark[i]=0;
      //memset(p_ad.mark, 0, p_ad.size * sizeof(unsigned));
      
      nb_in_plateau = 0;
      best_of_best = BIG;
      
      //p_ad.best_cost = p_ad.total_cost = Cost_Of_Solution(1);
    }
    
    __syncthreads();
    
    if( isMaster ) {
      p_ad.min_j = 0;
      int tmp_cost = Cost_Of_Solution(1);
      p_ad.best_cost = p_ad.total_cost = tmp_cost;
      //printf("Initial Cost[%d]: %d\n", blockIdx.x, p_ad.best_cost);
    }

    __syncthreads();
    while( (p_ad.total_cost!=0) && winner[0]==-1 && (p_ad.nb_iter < 100 && p_ad.nb_restart<1000000)) {
      //if( isMaster && p_ad.nb_iter % 100 == 0 )  {
      /*if( isMaster && blockIdx.x == 1) {
	printf("Iter[%d]: %d -- (%d, %d) -- swaps: %d, mark vars: %d -- list_len: %d, buffer_tmp[%d]: %d, local min: %d, nb_plateau: %d\n", p_ad.nb_iter, p_ad.total_cost, p_ad.max_i, p_ad.min_j, p_ad.nb_swap, p_ad.nb_var_marked, list_len_i, buffer_tmp, p_ad.list_i2_nb[buffer_tmp], p_ad.nb_local_min, nb_in_plateau);
	//printf("Marked max_i: %d -- min_j: %d\n", p_ad.mark[p_ad.max_i], p_ad.mark[p_ad.min_j]);
	}*/

      /*if( isMaster ) {
	printf("Iter[%d, %d]: %d (%d, %d), nb_var_marked: %d, nb_swap: %d, winner[0]: %d, winner[1]: %d\n", blockIdx.x, (p_ad.nb_iter+p_ad.nb_iter_tot), p_ad.total_cost, p_ad.max_i, p_ad.min_j, p_ad.nb_var_marked, p_ad.nb_swap, winner[0], winner[1]);
	}*/
      __syncthreads();
      
      if( isMaster ) {
	//printf("iter: %d \n",p_ad.nb_iter);
	if(p_ad.best_cost < best_of_best)
	  best_of_best = p_ad.best_cost;
	
	p_ad.nb_iter++;
      }
      
      //uncomment to use restarts or to stop the algorithm after a given number of iterations
      __syncthreads();
      if(p_ad.nb_iter >= p_ad.restart_limit) {
	if(p_ad.restart < p_ad.restart_max) 
	  goto restart;
	break;
      }
      
      __syncthreads();
      
#ifdef PAR
	Select_Vars_To_Swap_Par();
#else
	Select_Vars_To_Swap();
#endif
	__syncthreads();
	//if(isMaster) {printf("ITER: %d, var_marked: %d\n", p_ad.nb_iter, p_ad.nb_var_marked); }
	
	if(p_ad.min_j == -1)
	  continue;
        
	if( isMaster ) {
	  /*printf("total_cost: %d -- new_cost: %d\n",
	    p_ad.total_cost, p_ad.new_cost);*/
	    if(p_ad.total_cost != p_ad.new_cost) {
	      nb_in_plateau = 0;
	    }

	    if(p_ad.new_cost < p_ad.best_cost)
	      p_ad.best_cost = p_ad.new_cost;

	    nb_in_plateau++;
	}

	__syncthreads();

	//if(threadIdx.x == 0) {
	if(p_ad.max_i == p_ad.min_j) {
	  __syncthreads();
	  if( isMaster ) {
	    do_reset=0;
	    p_ad.nb_local_min++;
	    //Mark variable
	    Mark(p_ad.max_i, p_ad.freeze_loc_min);
	    if(p_ad.nb_var_marked + 1 >= p_ad.reset_limit) {
	      do_reset=1;
	      //Do_Reset(p_ad.nb_var_to_reset);
	    }
	  }
	  __syncthreads();
	  if(do_reset) {
	    Do_Reset(p_ad.nb_var_to_reset);
	  }
	  // __syncthreads();
	}
	else {
	    //__syncthreads();
	    if( isMaster ) {
		Mark(p_ad.max_i, p_ad.freeze_swap);
		Mark(p_ad.min_j, p_ad.freeze_swap);
		Ad_Swap(p_ad.max_i, p_ad.min_j);
		//printf("Marking variables: %d, %d\n",p_ad.mark[p_ad.max_i], p_ad.mark[p_ad.min_j]);
	    }
	    __syncthreads();

	    Executed_Swap(p_ad.max_i, p_ad.min_j);
	    
	    __syncthreads();

	    if( isMaster ) {
		p_ad.total_cost = p_ad.new_cost;
	    }

	    //__syncthreads();
	}
	//}
	__syncthreads();
    }

#if __CUDA_ARCH__ >= 200
    if( threadIdx.x == 0 && 0) {
      //printf("final cost: %d \n",p_ad.total_cost);
      print_stat();
      printf("WIN: %d\n", winner[0]);
    }
#endif

    if( winner[0] < 0 && isMaster /*&& p_ad.total_cost == 0*/) {
	int tmp = atomicExch(&winner[0], blockIdx.x);
	//in case that many threads finish nearly at the same time, only the winner will have access to this var
	if(tmp == -1) {
	  //Check_Solution(p_ad.sol, p_ad.size);
	    winner[1] = blockIdx.x;
	    winner[2] = p_ad.total_cost;
	    winner[3] = p_ad.nb_iter + p_ad.nb_iter_tot;
	    winner[4] = p_ad.nb_swap + p_ad.nb_swap_tot;
	    winner[5] = p_ad.nb_reset + p_ad.nb_reset_tot;
	    winner[6] = p_ad.nb_restart;
	    //printf("WINNER: %d, block.id: %d (%d), cost: %d, iter: %d, swaps: %d, resets: %d\n", winner[1], blockIdx.x, tmp,  p_ad.total_cost, p_ad.nb_iter, p_ad.nb_swap, p_ad.nb_reset);
	    printf("winner--[0]: %d\n", winner[0]);
	    //print_stat();
	    //__threadfence();
	    //asm("trap;");
	}
    }

    __syncthreads();

    return p_ad.total_cost;
}


#ifdef PAR
__shared__ int Buffer_Cost[THREADS];
__shared__ int Buffer_Vars[THREADS];


__device__ void 
Select_Vars_To_Swap_Par(void) {
  int s2 = size2*size2;
  int tid = threadIdx.x;
  int thread_size = Div_Round_Up(s2, THREADS);
  int init = tid * thread_size;
  int end  = init + thread_size;

  Buffer_Cost[ tid ] = p_ad.total_cost;
  Buffer_Vars[ tid ] = -1;

  p_ad.list_ij2_nb[tid ] = 0;

  int tmp;
  int i, j;
  
  //printf("TID[%d] init: %d, end: %d\n", tid, init, end);
  __syncthreads();

  for(int index=init; (index<end && index<s2); index++) {
    //1d to 2d mapping
    i = index / size2;
    j = index % size2 + size2;
    
    tmp = Cost_If_Swap(p_ad.total_cost, i, j);
    //printf("thread[%d], index: %d, i: %d, j: %d --> %d\n", tid, index, i, j);

    if( Buffer_Cost[tid] >= tmp ) {
      if( Buffer_Cost[tid] > tmp ) {
	Buffer_Cost[tid] = tmp;
	p_ad.list_ij2_nb[ tid ] = 0;
      }
      /*
	p_ad.list_ij2[ tid ][ p_ad.list_ij2_nb[tid] ].i = i;
	p_ad.list_ij2[ tid ][ p_ad.list_ij2_nb[tid] ].j = j;
	p_ad.list_ij2_nb[tid]++;
      */

      /*p_ad.list_ij2[ tid ][ p_ad.list_ij2_nb[tid]++ ] = index;
	if(p_ad.list_ij2_nb[tid] >= p_ad.size/THREADS) 
	p_ad.list_ij2_nb[tid] = 0;*/
      
      p_ad.list_ij2[ tid ][ p_ad.list_ij2_nb[tid] ] = index;
      p_ad.list_ij2_nb[tid] = (p_ad.list_ij2_nb[tid] + 1) % (p_ad.size / THREADS);
    }
  }

  if( p_ad.list_ij2[ tid ] == 0 )
    Buffer_Vars[ tid ] = -2;

  __syncthreads();
  //printf("tid[%d] init: %d, end: %d\n", tid, init, end);
  //__syncthreads();
  //asm("trap;");
  if( isMaster ) {
    int list_len = 0;
    int compute_vars = 1;
    p_ad.new_cost = p_ad.total_cost;
    for(int i=0;i<THREADS;i++) {
      if( p_ad.new_cost  >= Buffer_Cost[i]  && Buffer_Vars[i] != -2 ) {
	if( p_ad.new_cost > Buffer_Cost[i] ) {
	  p_ad.new_cost = Buffer_Cost[i];
	  list_len = 0;
	}
	p_ad.list_ij2[ THREADS ][ list_len++ ] = i;
      }
    }
    
    if( p_ad.new_cost >= p_ad.total_cost  ) {
      if( list_len == 0 || ( USE_PROB_SELECT_LOC_MIN && Random(100) <(unsigned) p_ad.prob_select_loc_min) ) {
	// 1 -> p_ad.nb_var_marked 
	p_ad.max_i = p_ad.min_j = 1;
	compute_vars = 0;
      }
      //else if( !USE_PROB_SELECT_LOC_MIN && (tmp = Random
    }

    if( compute_vars ) {
      int tmp2 = Random(list_len);
      int buffer_tmp = p_ad.list_ij2[THREADS][tmp2];
      
      tmp = Random( p_ad.list_ij2_nb[buffer_tmp] );
      /*p_ad.max_i = p_ad_list_ij2[ tmp ].i;
	p_ad.min_j = p_ad.list_ij2[ tmp ].j;*/
      int index2= p_ad.list_ij2[buffer_tmp][ tmp ];
      p_ad.max_i = index2 / size2;
      p_ad.min_j = index2 % size2 + size2;
    }
  }
}

#else

__device__ void
Select_Vars_To_Swap(void) {
  int i, j;
  int x;
  

  if( isMaster ) {
    
    p_ad.list_ij_nb = 0;
    p_ad.new_cost = BIG;
    p_ad.nb_var_marked = 0;
    stop_var_sel=0;
    
    for(i=0; i<size2; i++) {
      if(Marked(i)) {
	p_ad.nb_var_marked++;
	continue;
      }
      for(j=size2; j< p_ad.size; j++) {
      //for(j=i+1; j<p_ad.size; j++) {
	if(Marked(j)) continue;
	x = Cost_If_Swap(p_ad.total_cost, i, j);
	//printf("XX[%d,%d]: %d\n", i,j,x);
	if( x<= p_ad.new_cost ) {
	  if( x < p_ad.new_cost ) {
	    p_ad.new_cost = x;
	    p_ad.list_ij_nb = 0;
	    p_ad.list_ij[ p_ad.list_ij_nb ].i = i;
	    p_ad.list_ij[ p_ad.list_ij_nb ].j = j;
	    
	    p_ad.list_ij_nb = (p_ad.list_ij_nb + 1) % p_ad.size;
	    /*if( x < p_ad.total_cost) {
	      stop_var_sel=1;
	      goto a;
	      }*/
	  }
	}
      }
    }
  a:
    __syncthreads();
    if(stop_var_sel) {
      p_ad.max_i = p_ad.list_ij[0].i;
      p_ad.min_j = p_ad.list_ij[0].j;
      return;
    }
    int compute_vars = 1;
    
    if( p_ad.new_cost >= p_ad.total_cost ) {
      if( p_ad.list_ij_nb == 0 ||
	  (USE_PROB_SELECT_LOC_MIN && Random(100) < (unsigned) p_ad.prob_select_loc_min) ) {
	p_ad.max_i = p_ad.min_j = i;
	compute_vars = 0;
      }
      else if( !USE_PROB_SELECT_LOC_MIN && (x = Random(p_ad.list_ij_nb + p_ad.size)) < p_ad.size ) {
	p_ad.max_i = p_ad.min_j = x;
	compute_vars = 0;
      }
    }

    if( compute_vars ) {
      x = Random(p_ad.list_ij_nb);
      p_ad.max_i = p_ad.list_ij[x].i;
      p_ad.min_j = p_ad.list_ij[x].j;
    }

  }
  
}

#endif

__device__ void Ad_Swap(int i, int j) {
    int x;
    p_ad.nb_swap++;
    x = p_ad.sol[i];
    p_ad.sol[i] = p_ad.sol[j];
    p_ad.sol[j] = x;
}

//Only works for the deafult reset function.... other reset functions might require some modificaitons
//Be aware that in this case I am not unmarking varibles to avoid accessing global varaibles... One could just increase nb_swap+=[size of tabu list] to unmark those variables
__device__ void Do_Reset(int n) {
    //int cost = Reset(n);
    int cost;

    if( isMaster ) {
      // printf("Do reset\n");
	cost=Reset(n);
	p_ad.nb_reset++;
    }

    __syncthreads();

    if( isMaster ) {
      cost = Cost_Of_Solution(1);
      //printf("Reset...\n");
      p_ad.total_cost = cost;
      //for(int i=0;i<p_ad.size;i++) UnMark(i);
    }
    __syncthreads();
    //p_ad.total_cost = (cost < 0) ? Cost_Of_Solution(1) : cost;
}

__shared__ int iReset, jReset, xReset;
__shared__ int sizeReset;

__device__ int Reset1(int n) {
    int i, j, x;
    int size = p_ad.size;
 
    //sizeReset = p_ad.size;

    while(n--) {
        i = Random(size);
        j = Random(size);
        
        p_ad.nb_swap++;
        
        x = p_ad.sol[i];
        p_ad.sol[i] = p_ad.sol[j];
        p_ad.sol[j] = x;
    }

    return -1;
}


__shared__ curandState global_cudaRand;


__device__ void init_param(void) {

    if( isMaster ) {
    p_ad.size = SIZE;
    p_ad.nb_var_to_reset = -1;

    
    p_ad.restart_max = 0;


    //p_ad.seed = 12311;

    p_ad.prob_select_loc_min = 50;
    p_ad.freeze_loc_min = 1;
    p_ad.freeze_swap = 0;
    p_ad.reset_limit = 1;
    //p_ad.reset_percent = 5;
    p_ad.restart_limit = 1000000000;
    p_ad.restart_max = 0;
    p_ad.first_best = 0;

    
    if( p_ad.size < 8 || p_ad.size % 4 != 0 ) {
      printf("No solution with size = %d\n'",p_ad.size);
      asm("trap;");
    }

    sum_mid_x = (p_ad.size * (p_ad.size + 1)) / 4;
    sum_mid_x2 = ((long long) sum_mid_x * (2 * p_ad.size + 1)) / 3LL;
    coeff = sum_mid_x2 / sum_mid_x;
    
#if __CUDA_ARCH__ >= 200
    if(blockIdx.x == 0) {
      printf("mid sum x = %d,  mid sum x^2 = %lld, coeff: %d\n",
	     sum_mid_x, sum_mid_x2, coeff);
    }
#endif

    p_ad.prob_select_loc_min = 80;
    p_ad.freeze_loc_min = 1;
    p_ad.freeze_swap = 0;
    //    p_ad.reset_limit = SIZE_S2 * 1.2;
    p_ad.reset_percent = 1;
    p_ad.restart_limit = 100;
    p_ad.restart_max = 1000000;

    /*p_ad.prob_select_loc_min = 80;
    p_ad.freeze_loc_min = 4;
    p_ad.freeze_swap = 0;
    p_ad.reset_limit = 2;
    p_ad.reset_percent = 5;*/

    p_ad.base_value = 1;

    p_ad.nb_var_to_reset = Div_Round_Up(p_ad.size * p_ad.reset_percent, 100);

    if (p_ad.nb_var_to_reset < 2) {
	  p_ad.nb_var_to_reset = 2;
	  //printf("increasing nb var to reset since too small, now = %d\n", p_ad.nb_var_to_reset);
    }

    curand_init(p_ad.seed,0, 0, &global_cudaRand);
    int i;
    for(i=0;i<=blockIdx.x;i++) {
      p_ad.seed=(int)(curand_uniform(&global_cudaRand) * 100000.0);
    }
    //printf("Block[%d] -- seed: %d\n",blockIdx.x,p_ad.seed);
    curand_init(p_ad.seed, 0, 0, &cudaRand);

    }
}

__device__ void print_stat(void) {
#if __CUDA_ARCH__ >= 200
  if( isMaster ) {
    printf("block[%d] ->NB_iter: %d, local min: %d, swaps: %d, resets: %d, restarts: %d, cost: %d\n", blockIdx.x,
	   p_ad.nb_iter+p_ad.nb_iter_tot, p_ad.nb_local_min, p_ad.nb_swap+p_ad.nb_swap_tot, p_ad.nb_reset+p_ad.nb_reset_tot, p_ad.restart, p_ad.total_cost);
    //printf("Winner: %d\n",blockIdx.x);
  }
#endif
}

__device__ void sol2device( int *sol ) {
    
  if( isMaster ) {
    //printf("sol2device\n");
    for(int i=0;i<p_ad.size;i++) {
      sol[i]=p_ad.sol[i];
      //printf("SolD[%d]: %d -- %d\n",i,p_ad.sol[i], sol[i]);
    }
  }
  __syncthreads();
}

__global__ void main2(int seed, int *sol_device, int *winner) {

    p_ad.seed = seed;
    init_param();
    Solve(winner);

    if( isMaster && p_ad.total_cost == 0 && winner[1] == blockIdx.x) {
	print_stat();
	sol2device(sol_device);
	//printf("Winner: %d\n", winner[1] );
    }

    //printf("ENDING %d\n",threadIdx.x);

    __syncthreads();

}


#ifndef NBLOCK
#define NBLOCK 1
#endif

void sol2host(int *sol_device) {
    //int sol_host[SIZE];
    int *sol_host = (int*) malloc( SIZE * sizeof(int) );
    cudaMemcpy(sol_host, sol_device, SIZE * sizeof(int), cudaMemcpyDeviceToHost); 
    /*for(int i=0;i<SIZE;i++) {
	printf("Sol[%d]: %d\n",i, sol_host[i]);
    }*/
    Check_Solution(sol_host, SIZE);
    //Display_Solution(sol_host, SIZE);
}


inline void check_cuda_errors(const char *filename, const int line_number)
{

  cudaThreadSynchronize();
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    printf("CUDA error at %s:%i: %s\n", filename, line_number, cudaGetErrorString(error));
    exit(-1);
  }

}


int main() {
#ifdef PAR
  printf("Parallel Neighbor Exploration On\n");
#else
  printf("Parallel Neighbor Exploration Off\n");
#endif
  time_t start, end;
  double length;
  time(&start);

  int *sol, *winner;
  
  //int w[2]={-1,-1};
  //w[0..1] reserved for winner information
  //Winner info
  //w[2] cost, w[3] nb_iters, w[4]  nb_swaps, w[5] nb_resets, w[6] nb_restarts
  int w[7] = {-1, -1, -1, -1, -1, -1, -1};
  
  size_t limit = 0;
  /*cudaDeviceGetLimi(&limit, cudaLimitStackSize);
    printf("cudaLimitStackSize: %u\n", (unsigned)limit); 
  */
  //    cudaThreadSetLimit(cudaLimitMallocHeapSize, 512);
  //cudaDeviceSetLimit(cudaLimitMallocHeapSize, 128*1024*1024);
  cudaDeviceSetLimit(cudaLimitMallocHeapSize, 1024*1024*1024);
  cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024*1024*100);
  cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
  printf("cudaLimitHeapSize: %u\n", (unsigned)limit);
  cudaMalloc( (void**)&sol, SIZE * sizeof(int) );
  cudaMalloc( (void**)&winner, 7*sizeof(int) );
  
  
  cudaMemcpy( winner, w, 7*sizeof(int), cudaMemcpyHostToDevice );
  

  int *sol_host = (int*) malloc( SIZE * sizeof(int) );
  for(int i=0;i<SIZE;i++) sol_host[i]=-1;
  
  cudaMemcpy( sol, sol_host, SIZE * sizeof(int), cudaMemcpyHostToDevice);
  int seed=time(NULL);
  //int seed = 1372314800;
  //int seed=1367568510;
  //int seed=368594130;
  //int seed=1372404090;
  printf("Seed: %d -- Threads: %d\n",seed, THREADS);
  printf("Dimensions, blocks: %d, threads: %d\n", NBLOCK, THREADS);
  main2<<<NBLOCK, THREADS>>>(seed, sol, winner);
  
  cudaDeviceSynchronize();
  check_cuda_errors(__FILE__, __LINE__);
  //checkCUDAError("TEST");
  cudaMemcpy(sol_host, sol, SIZE * sizeof(int), cudaMemcpyDeviceToHost); 
  cudaMemcpy(w, winner, 7*sizeof(int), cudaMemcpyDeviceToHost);
  
  cudaFree(sol);
  cudaFree(winner);
  
  time(&end);
  length = difftime(end, start);
  printf("time: %.0f secs -- WINNER[%d], cost: %d, iter: %d, swaps: %d, resets: %d, restarts: %d\n", length, w[1], w[2], w[3], w[4], w[5], w[6]);

  //  printf("Time: %.0f secs\n", length);
  //cudaDeviceReset();
  
  if( w[2] == 0 ) {
    Check_Solution(sol_host, SIZE);
    /*for(int i=0; i < SIZE/2; i++) 
      printf("%d ", sol_host[i]);
    printf("\n");
    for(int i=SIZE/2;i<SIZE;i++) 
      printf("%d ", sol_host[i]);
      printf("\n");*/
    //Display_Solution(sol_host, SIZE);
  }
  else {
    printf("NO SOLUTION FOUND!\n");
  }
  
  printf("Ending execution\n");
  return 1;
}
