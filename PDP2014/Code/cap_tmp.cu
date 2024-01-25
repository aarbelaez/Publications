/*
 *  Adaptive search - GPU Version
 *
 *
 * Please visit the https://pauillac.inria.fr/~diaz/adaptive/manual/index.html for a complete version of the original Adaptive Search code
 */

/*
static int size2;
static int size_sq;
static int *nb_occ;
static int *first; 
static int *err;
*/

__shared__ int size2;
__shared__ int size_sq;
__shared__ int nb_occ[SIZE*2];
__shared__ int first[SIZE*2];
__shared__ int err[SIZE];

//__shared__ int sol2[SIZE*2][SIZE*2];
//__shared__ int sol2[SIZE][SIZE];
__shared__ int nb_occ2[SIZE][SIZE*2];
__shared__ int err_swap[SIZE];
__shared__ int cost_swap[SIZE];


#define ERROR  (size_sq - (dist * dist))

#define ErrOn(k)   { err[k] += ERROR; err[k - dist] += ERROR; }


__device__ void Solve(int *winner) {
    if( isMaster ) {
	size2 = (p_ad.size - 1) / 2;
    
	size_sq = p_ad.size * p_ad.size;
    }
    __syncthreads();
/*    if (nb_occ == NULL)
    {
        nb_occ = (int *) malloc(p_ad.size * 2 * sizeof(int));
        first = (int *) malloc(p_ad.size * 2 * sizeof(int));
        err = (int *) malloc(p_ad.size * sizeof(int));
    }
*/    
    Ad_Solve(winner);
}

/*__device__ int 
Cost_If_Swap_Parallel(int l1, int l2) {
    int i, j;

    int size = p_ad.size;
    int cost = 0;
    int dist;

    int v1, v2;

    for(i=0;i<size;i++) {
	sol2[threadIdx.x][i] = p_ad.sol[i];
    }

    int tmp = sol2[threadIdx.x][l1];
    sol2[threadIdx.x][l1] = sol2[threadIdx.x][l2];
    sol2[threadIdx.x][l2] = tmp;

    for(dist=1; dist<=size2; dist++) {
	for(i=dist; i<size; i++) {
	    v1 = sol2[threadIdx.x][i] - sol2[threadIdx.x][i-dist];
	    for(j=i+1; j<size; j++) {
		v2 = sol2[threadIdx.x][j] - sol2[threadIdx.x][j-dist];
		if( v1 == v2 ) cost+=ERROR;
	    }
	}
    }

    cost_swap[threadIdx.x] = cost;
}*/


__device__
void Cost_If_Swap_Parallel(int l1, int l2) {
    int dist = 1;
    int diff, diff_translated;
    int nb;
    int r = 0;
    int v1, v2;
    int tmp, i;

    int W = 32;
    int H = 4;
    int idx = threadIdx.x % W;
    int idy = threadIdx.x / W;

    int s = Div_Round_Up( (p_ad.size*2), H );

    
    //printf("T[%d] (%d, %d)\n",threadIdx.x, idx, idy);
		    /*printf("Errorrrr\n");
		    asm("trap;");*/

/*__syncthreads();
	if(idx < p_ad.size) {
	    if(idx==0) printf("YYYY: %d\n",idx);
	}
__syncthreads();*/
    do {
	
	/*if(threadIdx.x < p_ad.size) 
	    for(i=0; i< p_ad.size*2; i++) 
		nb_occ2[threadIdx.x][i]=1;*/
	//if(dist == 2) asm("trap;");
	__syncthreads();

	tmp=idy*s;

	/*if( idx == 9 && dist == 1 ) {
	    printf("%d, %d-- tmp[%d] %d - %d -- max: %d\n", 
		  p_ad.nb_iter, idy, threadIdx.x, tmp,(tmp+s), (p_ad.size*2));
	}*/

 	//if( isMaster ) printf("%d +++++++++++\n",dist); __syncthreads();
	if(idx < p_ad.size) {
	    //if(idx==0) printf("IDX: %d - %d --- %d\n",idx, dist, p_ad.nb_iter);
	    for( int j=tmp; (j<tmp+s && j<p_ad.size*2); j++)  {
		nb_occ2[ idx ][ j ] = 0;
	    }
	}

	__syncthreads();

	/*if(threadIdx.x < p_ad.size) {
	    for(int j=0;j<p_ad.size*2;j++) {
		nb_occ2[ threadIdx.x ][j] = 0;
		if(nb_occ2[ threadIdx.x ][j] != 0 ) {
		    printf("%d, %d, Errorrrr %d --- %d = %d\n", 
		    p_ad.nb_iter, dist,
		    threadIdx.x, j, nb_occ2[ threadIdx.x ][j]);
		    tt=0;
		    //asm("trap;");
		}
	    }
	}
	__syncthreads();*/
	i=dist;
	if(threadIdx.x < p_ad.size) {
	    //printf("%d: ",i);
	    do {
		v1 = ( i == l1 ? p_ad.sol[l2] : 
			(i == l2 ? p_ad.sol[l1] : p_ad.sol[i]) );

		tmp = i - dist;
		v2 = ( tmp == l1 ? p_ad.sol[l2] : 
			( tmp == l2 ? p_ad.sol[l1] : p_ad.sol[tmp]) );

		diff = v2 - v1;
		diff_translated = diff + p_ad.size;
		nb = ++nb_occ2[threadIdx.x][diff_translated];
		//nb_occ2[threadIdx.x][diff_translated]++;

		//printf("%d ",diff_translated);
		if( nb > 1 ) r+=ERROR;

	    } while(++i < p_ad.size);
	}
	    /*printf("%d -- ", dist);
	    for(i=0;i<p_ad.size*2;i++) printf("%d ",nb_occ2[threadIdx.x][i]);
	    printf("\n");*/
	__syncthreads();
    } while(++dist <= size2);
    //printf("===== size2: %d\n", size2);
    __syncthreads();
    if(threadIdx.x < p_ad.size) {
	//for(i=0;i<p_ad.size*2;i++) printf("%d ",nb_occ2[threadIdx.x][i]);
	//printf("\n");
	err_swap[threadIdx.x] = r;
	//printf("%d ----%d\n", threadIdx.x,err_swap[threadIdx.x]);
    }
    __syncthreads();

}

__device__ 
//inline 
int 
Cost(int *err)
{

    int dist = 1;
    int i, first_i;
    int diff, diff_translated;
    int nb;
    int r = 0;

    //__syncthreads();

    //if(threadIdx.x != 0) printf("threadIdx.x: %d, SIZE: %d\n",threadIdx.x, p_ad.size);
    if(err) {
	/*if(threadIdx.x == 0) 
	    for(i=0;i<p_ad.size;i++) err[i]=0;*/
	if(threadIdx.x < p_ad.size) 
	    err[threadIdx.x] = 0;
    }
    
    
    /*if (err) 
        memset(err, 0, p_ad.size * sizeof(int));
    */

    //__syncthreads();

    do
    {
	/*if(threadIdx.x==0)
	    memset(nb_occ, 0,  p_ad.size * (2 * sizeof(int)));
        __syncthreads();	*/
	
	if(threadIdx.x < p_ad.size*2) 
	    nb_occ[threadIdx.x]=0;
	
        __syncthreads();
	if( isMaster ) {
	    //memset(nb_occ, 0,  p_ad.size * (2 * sizeof(int)));
	    /*for(i=0;i<p_ad.size*2;i++) 
		if(nb_occ[i]!=0) {
		    nb_occ[i]=0;
		    //printf("HHHPPP: %d --- dist: %d\n",i,dist);
		}
	    */
	
        i = dist;
	//printf("%d: ",i);
        do
        {
            diff = p_ad.sol[i - dist] - p_ad.sol[i];
            diff_translated = diff + p_ad.size;
            nb = ++nb_occ[diff_translated];
            //printf("%d ",diff_translated);
            if (err) 
            {
                if (nb == 1) 
                    first[diff_translated] = i;
                else
                {
                    if (nb == 2)
                    {
                        first_i = first[diff_translated];
                        ErrOn(first_i);
                    }
                    
                    ErrOn(i);
                }
            }
            
            if (nb > 1)
                r += ERROR;

        }
        while(++i < p_ad.size);
	}

        __syncthreads();
    }
    while(++dist <= size2);
    
    __syncthreads();
    return r;
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
    return Cost( (should_be_recorded) ? err : NULL );
}


/*
 *  COST_ON_VARIABLE
 *
 *  Evaluates the error on a variable.
 */
__device__ int
Cost_On_Variable(int i)
{
    return err[i];
}


/*
 *  EXECUTED_SWAP
 *
 *  Records a swap.
 */

__device__ void
Executed_Swap(int i1, int i2)
{
    Cost(err);
}


/*
 *  COST_IF_SWAP
 *
 *  Evaluates the new total cost for a swap.
 */

__device__ int
Cost_If_Swap(int current_cost, int i1, int i2)
{

    int x = p_ad.sol[i1];
    int r;

    if( isMaster ) {
	x = p_ad.sol[i1];
	p_ad.sol[i1] = p_ad.sol[i2];
	p_ad.sol[i2] = x;
    }

    __syncthreads();
    r = Cost(NULL);

    if( isMaster ) {
	p_ad.sol[i2] = p_ad.sol[i1];
	p_ad.sol[i1] = x;
    }
    __syncthreads();
    return r;  
}


/*
 *  DISPLAY_SOLUTION
 *
 *  Displays a solution.
 */
//__device__ 
void
Display_Solution(int *sol, int size)
{
    printf("Display Solution\n");
    //int size = p_ad.size;
    //int *sol = p_ad.sol;
    int i, j;
    
    int len = 4 + ((size * 4) - (1 + 2 * size)) / 2;
    char buff[len + 1];
    
    sprintf(buff, "%*s", len, "");
    printf("%s", buff);
    for(i = 0; i < size; i++)
        printf("+-");
    printf("+\n%s", buff);
    for(i = 0; i < size; i++)
    {
        for(j = 0; j < size; j++)
        {
            if (sol[i] - 1 == j)
                printf("|*");
            else
                printf("| ");
        }
        printf("|\n%s", buff);
    }
    for(i = 0; i < size; i++)
        printf("+-");
    printf("+\n");
    
    printf("sol:");
    for(i = 0; i < size; i++)
        printf("%4d", sol[i]);
    printf("\n");
    printf("----");
    for(i = 0; i < size; i++)
        printf("----");
    printf("\n");
    
    
    for(i = 1; i < size; i++)
    {
        printf("%3d:", i);
        for(j = 1; j <= i; j++)
            printf("  ");
        
        for(j = i; j < size; j++)
            printf("%4d", sol[j - i] - sol[j]);
        
        printf("\n");
    }
}


/*
 *  CHECK_SOLUTION
 *
 *  Checks if the solution is valid.
 */

int
Check_Solution(int *sol, int size)
{
    printf("Check Solution\n");
    for(int i=0;i<size;i++) {
	printf("%d - ",sol[i]);
    }
    printf("\n");
    //int size = p_ad.size;
    //int *sol = p_ad.sol;
    int i, j, d;
    int r = 1;

    int *nb_occ=NULL;
    
    if (nb_occ == NULL)
    {
        nb_occ = (int *) malloc(size * 2 * sizeof(int));
        if (nb_occ == NULL)
        {
            printf("%s:%d malloc failed\n", __FILE__, __LINE__);
            exit(1);
        }
    }

    for(i = 1; i < size; i++)
    {
        memset(nb_occ, 0, size * 2 * sizeof(int));
        for(j = i; j < size; j++)
        {
            d = sol[j - i] - sol[j];
            nb_occ[d + size]++;
        }
        
        for(d = 1; d < 2 * size; d++)
        {
            int nr = nb_occ[d];
            if (nr > 1)
            {
                int dist = d - size;
                printf("ERROR at row %d: distance %d appears %d times\n", i, dist, nr);
                r = 0;
            }
        }
    }
    
    return r;
}


