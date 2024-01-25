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

#include "mrand.cu"


#define BIG ((unsigned int) -1 >> 1)

#define BASE_MARK    ((unsigned) p_ad.nb_swap)
#define Mark(i, k)   p_ad.mark[i] = BASE_MARK + (k)
#define UnMark(i)    p_ad.mark[i] = 0
#define Marked(i)    (BASE_MARK + 1 <= p_ad.mark[i])

#define USE_PROB_SELECT_LOC_MIN ((unsigned) p_ad.prob_select_loc_min <= 100)

#define Div_Round_Up(x, y)   (((x) + (y) - 1) / (y))

#ifndef SIZE
#define SIZE 10
#endif

typedef struct {
    int max_i;
    int min_j;
    int new_cost;
    int best_cost;
    int mark[SIZE];
    int nb_var_marked;
    int size;
    int seed;

    int list_i[SIZE];
    int list_i_nb;

    int list_j[SIZE];
    int list_j_nb;
    
    int sol[SIZE];
    int total_cost;
    
    int nb_iter;
    int restart_limit;
    int nb_restart;
    int nb_swap;
    int swap;
    int nb_same_var;
    int nb_local_min;
    int freeze_loc_min;
    int reset_limit;
    int nb_var_to_reset;
    int freeze_swap;
    int restart_max;
    int restart;
    int prob_select_loc_min;
    int first_best;
    int nb_reset;
    int reset_percent;
    
    int base_value;

} AdData;

__shared__ AdData p_ad;


__device__ void Select_Var_Min_Conflict_Parallel(void); 

__device__ void Select_Var_High_Cost(void);
__device__ void Select_Var_Min_Conflict(void); 
__device__ void Do_Reset(int);
__device__ int Reset(int);

__device__ int Cost(int *);
__device__ int Cost_Of_Solution(int);
__device__ int Cost_On_Variable(int);
__device__ void Executed_Swap(int, int);
__device__ int Cost_If_Swap(int, int, int);

__device__ void Ad_Swap(int, int);

__device__ int Ad_Solve(int *);

#define isMaster  (threadIdx.x == 0)
//#define isMaster (tid == 0)

#define getId tid = threadIdx.y*blockDim.y + threadIdx.x

#include "cap_tmp.cu"


__shared__ int do_reset;



__device__ int Ad_Solve(int *winner) {

    int nb_in_plateau, best_of_best;

    if( isMaster ) {
	p_ad.nb_restart=-1;
	p_ad.nb_iter = 0;
	p_ad.nb_swap = 0;
	p_ad.swap = 0;
	p_ad.nb_same_var = 0;
	p_ad.nb_restart = 0;
	p_ad.nb_local_min = 0;
    }
    //printf("x: %d\n",threadIdx.x);
restart:
    
    if( isMaster ) {
	Random_Permut(p_ad.sol, p_ad.size, p_ad.base_value);
	int i;
	for(i=0;i<p_ad.size;i++) p_ad.mark[i]=0;
	//memset(p_ad.mark, 0, p_ad.size * sizeof(unsigned));
    
	nb_in_plateau = 0;
	p_ad.nb_restart++;
	best_of_best = BIG;
 
    //p_ad.best_cost = p_ad.total_cost = Cost_Of_Solution(1);
    }
    
    __syncthreads();
    int tmp_cost = Cost_Of_Solution(1);
    
    if( isMaster ) {
	p_ad.best_cost = p_ad.total_cost = tmp_cost;
    }

    __syncthreads();

    //if(threadIdx.x == 0) { printf("HHHPPP: %d\n",blockIdx.x); }

    while( (p_ad.total_cost>0) && winner[0]==-1 /*&& p_ad.nb_iter < 1000*/ ) {
	/*if( isMaster && 
	    p_ad.nb_iter >= 194 && p_ad.nb_iter <= 196
	  ) { printf("Iter[%d]: %d\n",p_ad.nb_iter, p_ad.total_cost); }*/
    //while (p_ad.total_cost) {
        //printf("iter, cost: %d\n",p_ad.total_cost);
	__syncthreads();

	if( isMaster ) {
	    //printf("iter: %d \n",p_ad.nb_iter);
	    if(p_ad.best_cost < best_of_best)
		best_of_best = p_ad.best_cost;

	    p_ad.nb_iter++;
	}

	//active to use restarts or to stop the algorithm after a given number of iterations
	/*
        if(p_ad.nb_iter >= p_ad.restart_limit) {
            if(p_ad.restart < p_ad.restart_max) 
                goto restart;
            break;
        }
        */
	__syncthreads();
        //if(1) {
            Select_Var_High_Cost();
	__syncthreads();
	/*if(isMaster) printf("iter: %d - MAX_I: %d\n",p_ad.nb_iter,p_ad.max_i);
	__syncthreads();*/
#ifndef PAR
         Select_Var_Min_Conflict();
#else
	 Select_Var_Min_Conflict_Parallel();
#endif

	__syncthreads();
	/*if( isMaster )
	    printf("iter[%d] max_i: %d, min_j: %d\n",p_ad.nb_iter,p_ad.max_i,p_ad.min_j);*/

        //}
        //exhaustive search
        //else {}
        
	if( isMaster ) {
	    if(p_ad.total_cost != p_ad.new_cost) {
		nb_in_plateau = 0;
	    }
        
	    if(p_ad.new_cost < p_ad.best_cost) 
		p_ad.best_cost = p_ad.new_cost;
        
	    nb_in_plateau++;
	}

	//__syncthreads();
        
	if(p_ad.min_j == -1)
            continue;
        
	__syncthreads();

	//if(threadIdx.x == 0) {
	if(p_ad.max_i == p_ad.min_j) {
	    //__syncthreads();
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
	    //__syncthreads();
	}
	else {
	    //__syncthreads();
	    if( isMaster ) {
		Mark(p_ad.max_i, p_ad.freeze_swap);
		Mark(p_ad.min_j, p_ad.freeze_swap);
		Ad_Swap(p_ad.max_i, p_ad.min_j);
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

    /*if( threadIdx.x == 0) {
	//printf("final cost: %d -- %d\n",p_ad.total_cost, Cost_Of_Solution(1));
      printf("final cost: %d \n",p_ad.total_cost);
    }*/

    //if(threadIdx.x == 0) { printf("HP : %d -- iter: %d\n",blockIdx.x, p_ad.nb_iter); }

    if( winner[0] < 0 && isMaster /*&& p_ad.total_cost == 0*/) {
	int tmp = atomicExch(&winner[0], blockIdx.x);
	//only want thread get access to the shared variable at time (the first one is the winner)
	if(tmp == -1) {
	    winner[1]=blockIdx.x;
	    printf("WINNER: %d, block.id: %d (%d), cost: %d, iter: %d\n", winner[1], blockIdx.x, tmp,  p_ad.total_cost, p_ad.nb_iter);
	    /*for(int j=0;j<p_ad.size;j++)
		printf("%d ",p_ad.sol[j]);
	    printf("\n");*/
	}
    }
    __syncthreads();
    return p_ad.total_cost;
}




/*
 *  SELECT_VAR_HIGH_COST
 *
 *  Computes err_swap and selects the maximum of err_var in max_i.
 *  Also computes the number of marked variables.
 */
__device__ void
Select_Var_High_Cost(void)
{
    int i;
    int x, max;
    
    if( isMaster ) {
	p_ad.list_i_nb = 0;
	max = 0;
	p_ad.nb_var_marked = 0;
    
	for(i = 0; i < p_ad.size; i++)
	{
	    if (Marked(i))
	    {
		p_ad.nb_var_marked++;
		continue;
	    }
        
	    x = Cost_On_Variable(i);
        
	    if (x >= max)
	    {
		if (x > max)
		{
		    max = x;
		    p_ad.list_i_nb = 0;
		}
		p_ad.list_i[p_ad.list_i_nb++] = i;
	    }
	}

	p_ad.nb_same_var += p_ad.list_i_nb;
	x = Random(p_ad.list_i_nb);
	p_ad.max_i = p_ad.list_i[x];

    }
    //__syncthreads();
}




/*
 *  SELECT_VAR_MIN_CONFLICT
 *
 *  Computes swap and selects the minimum of swap in min_j.
 */

__shared__ int use_local_min;
__shared__ int stop_var_sel;

__device__ void 
Select_Var_Min_Conflict_Parallel(void) {
    int j;
    int x;
    
    __syncthreads();
    if( isMaster ) 
	stop_var_sel = 0;
    __syncthreads();


    while(!stop_var_sel) {
	__syncthreads();
	//if(threadIdx.x < p_ad.size) 
	Cost_If_Swap_Parallel( threadIdx.x, p_ad.max_i );
	__syncthreads();

	if( isMaster ) {
	    stop_var_sel = 1; 
	    p_ad.list_j_nb = 0;
	    p_ad.new_cost = p_ad.total_cost;
	    //printf("==============");
	    for(j = 0; j < p_ad.size; j++) {

		if (USE_PROB_SELECT_LOC_MIN && j == p_ad.max_i)
		    continue;

		x = err_swap[j];

		if (x <= p_ad.new_cost)
		{
		    if (x < p_ad.new_cost)
		    {
			p_ad.list_j_nb = 0;
			p_ad.new_cost = x;
			/*
			if (p_ad.first_best)
			{
			    p_ad.min_j = p_ad.list_j[p_ad.list_j_nb++] = j;
			    return;
			}
			*/
		    }
            
		    p_ad.list_j[p_ad.list_j_nb++] = j;
		}
	    }

	    if (USE_PROB_SELECT_LOC_MIN && isMaster)
	    {
		if (p_ad.new_cost >= p_ad.total_cost && 
		      (Random(100) < (unsigned) p_ad.prob_select_loc_min ||
		      (p_ad.list_i_nb <= 1 && p_ad.list_j_nb <= 1)))
		{
		      p_ad.min_j = p_ad.max_i;
			//return;
		      stop_var_sel=2;
		      //break;
		}
		else 
		if (p_ad.list_j_nb == 0)
		{
		    if( isMaster ) {
			p_ad.nb_iter++;
			x = Random(p_ad.list_i_nb);
			p_ad.max_i = p_ad.list_i[x];
		    }
		    //goto a;
		    stop_var_sel=0;
		    //break;
		}
	    }
	}
	__syncthreads();
    }
    __syncthreads();


    if( isMaster && stop_var_sel != 2) {
	x = Random(p_ad.list_j_nb);
	p_ad.min_j = p_ad.list_j[x];
    }

}

__device__ void
Select_Var_Min_Conflict(void)
{
    int j;
    int x;

//a:
//    int stop=0;

    if( isMaster )
      stop_var_sel = 0;
    __syncthreads();

    while(!stop_var_sel) {
	__syncthreads();

	if( isMaster ) {
	    stop_var_sel=1;
	    p_ad.list_j_nb = 0;
	    p_ad.new_cost = p_ad.total_cost;
	}
 
	__syncthreads();
	//printf("+++++++++++\n");
	for(j = 0; j < p_ad.size; j++)
	{
	    //__syncthreads();
	    if (USE_PROB_SELECT_LOC_MIN && j == p_ad.max_i)
		continue;

	    __syncthreads();
	    x = Cost_If_Swap(p_ad.total_cost, j, p_ad.max_i);
	    __syncthreads();

	    if( isMaster ) {
		if (x <= p_ad.new_cost)
		{
		    if (x < p_ad.new_cost)
		    {
			p_ad.list_j_nb = 0;
			p_ad.new_cost = x;
		    /*
		    if (p_ad.first_best)
		    {
			p_ad.min_j = p_ad.list_j[p_ad.list_j_nb++] = j;
			return;
		    }
		    */
		    }
            
		    p_ad.list_j[p_ad.list_j_nb++] = j;
		}
	    }
	    //__syncthreads();
	}

	__syncthreads();
	if( isMaster ) {

	    if (USE_PROB_SELECT_LOC_MIN)
	    {
		//addedt the left part of the condition in order to have the same exact trajectory as the sequential algorithm
		/*if(p_ad.new_cost >= p_ad.total_cost && threadIdx.x == 0) {    
		    use_local_min = (Random(100) < (unsigned) p_ad.prob_select_loc_min );
		}
		__syncthreads();*/
		if (p_ad.new_cost >= p_ad.total_cost && 
		    (Random(100) < (unsigned) p_ad.prob_select_loc_min ||
		    (p_ad.list_i_nb <= 1 && p_ad.list_j_nb <= 1)))
		/*if ( p_ad.new_cost >= p_ad.total_cost && 
		    (use_local_min || (p_ad.list_i_nb <= 1 && p_ad.list_j_nb <= 1)) )*/
		{
		    p_ad.min_j = p_ad.max_i;
		    //return;
		    stop_var_sel=2;
		    //break;
		}
		else 
		if (p_ad.list_j_nb == 0)
		{
		    if( isMaster ) {
		
			p_ad.nb_iter++;
			x = Random(p_ad.list_i_nb);
			p_ad.max_i = p_ad.list_i[x];
		    }
		    //goto a;
		    stop_var_sel=0;
		    //break;
		}
	    }
	}
	__syncthreads();

    }

    __syncthreads();
    if( isMaster && stop_var_sel != 2) {
	x = Random(p_ad.list_j_nb);
	p_ad.min_j = p_ad.list_j[x];
    }
    //__syncthreads();
}


__device__ void Ad_Swap(int i, int j) {
    int x;
    p_ad.nb_swap++;
    x = p_ad.sol[i];
    p_ad.sol[i] = p_ad.sol[j];
    p_ad.sol[j] = x;
}

//Only works for the deafult reset function.... other reset functions might require some modificaitons
__device__ void Do_Reset(int n) {
    //int cost = Reset(n);
    int cost;

    if( isMaster ) {
	cost=Reset(n);
	p_ad.nb_reset++;
    }

    __syncthreads();
    cost = Cost_Of_Solution(1);
    __syncthreads();

    if( isMaster ) 
	p_ad.total_cost = cost;
    __syncthreads();
    //p_ad.total_cost = (cost < 0) ? Cost_Of_Solution(1) : cost;
}

__shared__ int iReset, jReset, xReset;
__shared__ int sizeReset;

__device__ int Reset(int n) {
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

    p_ad.base_value = 1;

    p_ad.reset_percent = 5;
    p_ad.nb_var_to_reset = Div_Round_Up(p_ad.size * p_ad.reset_percent, 100);

    if (p_ad.nb_var_to_reset < 2) {
	  p_ad.nb_var_to_reset = 2;
	  //printf("increasing nb var to reset since too small, now = %d\n", p_ad.nb_var_to_reset);
    }


    //srand( time(NULL) );

    curand_init(p_ad.seed,0, 0, &global_cudaRand);
    int i;
    for(i=0;i<=blockIdx.x;i++) {
    //for(i=0;i<=79+blockIdx.x;i++) {
	p_ad.seed=(int)(curand_uniform(&global_cudaRand) * 100000.0);
	//printf("i: %d\n",p_ad.seed);
    }
    //printf("Block[%d] -- seed: %d\n",blockIdx.x,p_ad.seed);
    curand_init(p_ad.seed, 0, 0, &cudaRand);

    }
}

__device__ void print_stat(void) {

    if( isMaster ) {
	printf("nb_iter: %d, local min: %d, swaps: %d, resets: %d, cost: %d\n",
	      p_ad.nb_iter, p_ad.nb_local_min, p_ad.nb_swap, p_ad.nb_reset, p_ad.total_cost);
	//printf("Winner: %d\n",blockIdx.x);
    }
}

__device__ void sol2device( int *sol ) {
    
    if( isMaster ) {
	printf("sol2device\n");
	for(int i=0;i<p_ad.size;i++) {
	    sol[i]=p_ad.sol[i];
	    //printf("SolD[%d]: %d -- %d\n",i,p_ad.sol[i], sol[i]);
	}
    }
    __syncthreads();
}

__global__ void main2(int seed, int *sol_device, int *winner) {

//    AdData *data = (AdData*) malloc(sizeof(AdData));
    //AdData data;
    //int *winner = (int*) malloc(sizeof(int));
    p_ad.seed = seed;
    //printf("Initial seed: %d\n",seed);
    init_param();
    Solve(winner);

    //if( *winner == blockIdx.x && threadIdx.x == 0)  {
    if( isMaster && p_ad.total_cost == 0 && winner[1] == blockIdx.x) {
	print_stat();
	sol2device(sol_device);
	//printf("Winner: %d\n", winner[1] );
    }
    __syncthreads();
    /*if(threadIdx.x == 0) 	
	printf("Ending: %d\n", (blockIdx.x) );*/

    //Random_Permut(data.sol, data.size, data.base_value);
    //Check_Solution();    
    //Display_Solution();
    //asm("trap;");
    //return 1;
    //return Ad_Solve(data);
}


#ifndef NBLOCK
#define NBLOCK 1
#endif

void sol2host(int *sol_device) {
    int sol_host[SIZE];
    cudaMemcpy(sol_host, sol_device, SIZE * sizeof(int), cudaMemcpyDeviceToHost); 
    /*for(int i=0;i<SIZE;i++) {
	printf("Sol[%d]: %d\n",i, sol_host[i]);
    }*/
    Check_Solution(sol_host, SIZE);
    Display_Solution(sol_host, SIZE);
}

/*void checkCUDAError(const char* msg) {
cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}*/

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



#ifdef DEBUG

#if 0
__global__ void kernel(int seed) {
    init_param();


    if( isMaster ) {
	size2 = (p_ad.size - 1) / 2;
    
	size_sq = p_ad.size * p_ad.size;

	for(int i=0;i<p_ad.size;i++) p_ad.mark[i]=0;
	
	//curand_init(seed,0, 0, &global_cudaRand);
	Random_Permut(p_ad.sol, p_ad.size, p_ad.base_value);

	p_ad.nb_iter = 0;
    }

    __syncthreads();

    int tmp_cost = Cost_Of_Solution(1);

    __syncthreads();

    if( isMaster ) 
	p_ad.best_cost = p_ad.total_cost = tmp_cost;

    __syncthreads();

    //printf("HHHH\n");
    while( p_ad.total_cost > 0  && p_ad.nb_iter < 10) {
	if( isMaster )
	    printf("iter: %d\n",p_ad.nb_iter);
    //for(int i=0;i<40000;i++) {
	__syncthreads();

        Select_Var_High_Cost();
	/*if(p_ad.nb_iter >= 27380) {
	      int a=p_ad.nb_iter+1;
	}*/
        //Select_Var_Min_Conflict();
	__syncthreads();
	Select_Var_Min_Conflict_Parallel();

	__syncthreads();

	/*if(p_ad.min_j == -1)
	    continue;
	*/
	if(p_ad.min_j != -1 ) {
	do_reset=0;

	if( isMaster ) {
	    if(p_ad.max_i == p_ad.min_j) {
		Mark(p_ad.max_i, p_ad.freeze_loc_min);
		if(p_ad.nb_var_marked + 1 >= p_ad.reset_limit)
		    do_reset=1;
	    }
	    else {
	    //printf("R: %d\n",r);
		Mark(p_ad.max_i, p_ad.freeze_loc_min);
		Mark(p_ad.min_j, p_ad.freeze_loc_min);
		Ad_Swap(p_ad.max_i, p_ad.min_j);
	    }
	}

	__syncthreads();
	
	if(do_reset) Do_Reset(p_ad.nb_var_to_reset);
	else Executed_Swap(p_ad.max_i, p_ad.min_j);
	
	__syncthreads();

	if( isMaster ) {
	    if(!do_reset)
		p_ad.total_cost = p_ad.new_cost;
	    p_ad.nb_iter++;
	}
	__syncthreads();
	}
    }

    __syncthreads();

    if( isMaster )
      print_stat();
}
#endif

__global__ void kernel(int seed) {
    init_param();


    for(int i=0;i<100;i++) 
      if( isMaster ) {
	size2 = (p_ad.size - 1) / 2;
    
	size_sq = p_ad.size * p_ad.size;

	for(int i=0;i<p_ad.size;i++) p_ad.mark[i]=0;
	
	//curand_init(seed,0, 0, &global_cudaRand);
	Random_Permut(p_ad.sol, p_ad.size, p_ad.base_value);

	p_ad.nb_iter = 0;

	int tmp = Cost_If_Swap(0, 0, 10);
	Cost_If_Swap_Parallel(0, 10);
	printf("tmp: %d -- tmp2: %d\n", tmp, err_swap[0]);
      }
}

int main() {
    kernel<<<NBLOCK, 64>>>(1234);
    cudaDeviceSynchronize();
    check_cuda_errors(__FILE__, __LINE__);

}

#else

int main() {
    int *sol, *winner;
    int w[2]={-1,-1};
    cudaMalloc( (void**)&sol, SIZE * sizeof(int) );
    cudaMalloc( (void**)&winner, 2*sizeof(int) );


    cudaMemcpy( winner, w, 2*sizeof(int), cudaMemcpyHostToDevice );

    int *sol_host = (int*) malloc( SIZE * sizeof(int) );
    for(int i=0;i<SIZE;i++) sol_host[i]=-1;

    cudaMemcpy( sol, sol_host, SIZE * sizeof(int), cudaMemcpyHostToDevice);
    //int seed=time(NULL);
    int seed=1367568510;
    //int seed=1368594130;
    printf("Seed: %d\n",seed);
    main2<<<NBLOCK, 128>>>(seed, sol, winner);
    //main2<<<NBLOCK, 64>>>(1234, sol, winner);
    //main2<<<112, 64>>>(1234, sol, winner);
    //main2<<<NBLOCK, 32>>>(1234);
    //sol2host(sol);

    //int sol_host[SIZE];
    //int *sol_host = (int*) malloc( SIZE * sizeof(int) );
    cudaDeviceSynchronize();
    check_cuda_errors(__FILE__, __LINE__);
    //checkCUDAError("TEST");
    cudaMemcpy(sol_host, sol, SIZE * sizeof(int), cudaMemcpyDeviceToHost); 
    /*for(int i=0;i<SIZE;i++) {
	printf("Sol[%d]: %d\n",i, sol_host[i]);
    }*/

    cudaFree(sol);
    cudaFree(winner);
    //cudaDeviceReset();

    Check_Solution(sol_host, SIZE);
    Display_Solution(sol_host, SIZE);

    printf("Ending execution\n");
    return 1;
}

#endif
