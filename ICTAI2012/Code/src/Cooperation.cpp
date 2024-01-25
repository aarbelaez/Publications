
#include <mpi.h>
#include "Cooperation.h"
#include <assert.h>
#include <omp.h>

#include "mt19937p.h"

namespace ubcsat {

Cooperation::Cooperation(int _nThreads, int _aGroup, int _rank) {
	nThreads=_nThreads;
	End = false;
	winnerThread=-1;
	globalSeed=-1;
    
    aGroup=_aGroup;
    gNumber=_rank/aGroup;
    idGroup=_rank%aGroup;
    rNumber=_rank;
    
    if(nThreads%aGroup != 0 && _rank==0) {
        std::cout<<"Error with number of groups and total number of threads"<<std::endl;
        MPI_Abort(MPI_COMM_WORLD, 911);
    }

	//InitSeed(globalSeed);
	//sgenrand(globalSeed, &mt);
	//sharing
	initVars = new BOOL[aGroup];
	aBestVarValue = new UINT32*[aGroup];
	iBestNumFalse = new UINT32[aGroup];
	isLocalMinimum = new BOOL[aGroup];
	bestScore = new SINT32*[aGroup];
	FlipCount = new UINT32*[aGroup];
	MaxFlips = new UINT32[aGroup];
	BestImp = new BOOL*[aGroup];
    
    LastAdaptStep = new UINT32[aGroup];
    LastAdaptNumFalse = new UINT32[aGroup];
    
    NumFalseClauses = new int[aGroup];
    ClauseFailure = new int*[aGroup];
    FalseList = new UINT32*[aGroup];
    FalseListPos = new UINT32*[aGroup];
    aVarInfo = new UINT32*[aGroup];
    
	for(int i=0;i<aGroup;i++) { 
		BestImp[i]=new BOOL[aGroup];
		for(int j=0;j<aGroup;j++) 
			BestImp[i][j]=false;
	}


	for(int i=0;i<aGroup;i++) {
        LastAdaptNumFalse[i]=0;
        LastAdaptStep[i]=0;
		initVars[i]=false;
		MaxFlips[i]=0;
	}
    
    sInfo=SLEEP;
    vSend=true;

    
    if(rNumber==0) {
        /*
        algIds=new int[nThreads];*/
        algIds=new CoreInfo[nThreads];
        qValueVector=new double[aGroup];
    }
    qValue=static_cast<double>(iNumClauses);
    alpha=0.5;
}

Cooperation::~Cooperation(void) {
	/*omp_destroy_lock(&read);
	for(int i=0;i<nThreads;i++)
		omp_destroy_lock(&write[i]);*/
}

void Cooperation::InitInfo(int threadID) {
    ackrecv=0;
    

#ifdef SHARE_INFO
    bSend = new int[iNumVars+1];
    bRecv = new int*[aGroup];

    for(int i=0;i<aGroup;i++) {
        bRecv[i]=new int[iNumVars+1];
        //bSend[i]=(int*) malloc( (Solver[i].iNumVars+1)* sizeof(int) );
        //bRecv[i]=(int*) malloc( (Solver[i].iNumVars+1)* sizeof(int) );
        
        aBestVarValue[i]=new UINT32[iNumVars+1];
        /*bestScore[i]=new SINT32[iNumVars+1];

        ClauseFailure[i]=new int[iNumClauses];
        FalseList[i]=new UINT32[iNumClauses];
        FalseListPos[i]=new UINT32[iNumClauses];

        aVarInfo[i]=new UINT32[iNumVars+1];
        
        
        for(int j=0;j<iNumClauses;j++)
            ClauseFailure[i][j]=0;
        */
    }
#endif
}

//#define Array3(bTmp) UINT32 bTmp[Solver[threadID].iNumVars+1]; for(int i=0;i<Solver[threadID].iNumVars+1;i++) {bTmp[i]=aBestVarValue[threadID][i];}
//#define Array(bTmp) for(int i=0;i<Solver[threadID].iNumVars+1;i++) {bTmp[i]=aBestVarValue[threadID][i];}
//#define Array2(bTmp) for(int i=0;i<Solver[threadID].iNumVars+1;i++) {aBestVarValue[threadID][i]=bTmp[i];}

int compare(const void *a, const void *b) {
    CoreInfo *ma = (CoreInfo*) a;
    CoreInfo *mb = (CoreInfo*) b;
    return (mb)->qValue - (ma)->qValue;
    //return ( (CoreInfo*)a->iNumFalse - (CoreInfo*)b->iNumFalse  );
}
    
void Cooperation::GetObjFun(void) {
    int pi;
    int signalSend=1;
    algIds[0].setInfo(qValue, 0);

    int trest=static_cast<int>(static_cast<double>(nThreads)*0.2);

    for(pi=1;pi<nThreads; pi++) {
        MPI_Send(&signalSend, 1, MPI_INT, pi, GET_VAL_MSG, MPI_COMM_WORLD);
    }
    for(pi=1;pi<nThreads;pi++) {
        //MPI_Recv(&iBestNumFalse[pi], 1, MPI_INT, pi, VAL_MSG, MPI_COMM_WORLD, &status);
        //algIds[pi].setInfo(iBestNumFalse[pi], pi); 
        MPI_Recv(&qValueVector[pi], 1, MPI_DOUBLE, pi, VAL_MSG, MPI_COMM_WORLD, &status);
        algIds[pi].setInfo(qValueVector[pi], pi);
        //std::cout<<"receiving from: "<<pi<<std::endl;
    }
    
    //define the list of algortihms that will be restarted.. and communicate them
    //naive one which restart all algorithms..
    //for(pi=0;pi<nThreads;pi++) restAlgs[pi]=1;
    
    
    
    /*
    int selthread;
    for(;trest>0;trest--) {
        selthread = RandomInt(nThreads);
        if(restAlgs[selthread] == 1) 
            trest++;
        else { restAlgs[selthread]=1;
            std::cout<<selthread<<" ";
        }
    }
    std::cout<<std::endl;
    */

    /*std::cout<<"BSorted: "<<std::endl;
    for(int i=0;i<nThreads;i++) {
        std::cout<<algIds[i].idCore<<" ("<<algIds[i].iNumFalse<<") ";
    }
    std::cout<<std::endl;*/

    qsort( algIds, nThreads, sizeof(CoreInfo), compare);

    std::cout<<"Sorted: "<<std::endl;
    for(int i=0;i<nThreads;i++) {
        std::cout<<algIds[i].idCore<<" ("<<algIds[i].qValue<<") ";
    }
    std::cout<<std::endl;
    /*std::cout<<"Restart: "<<std::endl;
    for(;trest>0;trest--) {
        //restAlg[algIds[trest-1].idCore]=1;
        algIds[trest-1].restart=1;
        std::cout<<(algIds[trest-1].idCore)<<" ("<<algIds[trest-1].qValue<<") ";
    }
    std::cout<<std::endl;    */

    for(pi=0;pi<nThreads;pi++) {
        //std::cout<<"Send restart signal to: "<<algIds[pi].idCore<<std::endl;
        //MPI_Send(&restAlgs[pi], 1, MPI_INT, pi, OPT_MSG, MPI_COMM_WORLD);
        if(algIds[pi].idCore != 0)
            MPI_Send(&algIds[pi].restart, 1, MPI_INT, algIds[pi].idCore, OPT_MSG, MPI_COMM_WORLD);
        else {
            if(algIds[pi].restart==1) bRestart=TRUE;
        }
    }
}

void Cooperation::RestartConfirmation(void) {

    MPI_Iprobe(0, GET_VAL_MSG, MPI_COMM_WORLD, &tFlag, &status);
    if(tFlag) {
        //std::cout<<"Core: "<<rNumber<<" iNumFalse: "<<iNumFalse<<std::endl;
        MPI_Recv(&rOK, 1, MPI_INT, 0, GET_VAL_MSG, MPI_COMM_WORLD, &status);
        //MPI_Send(&iNumFalse, 1, MPI_INT, 0, VAL_MSG, MPI_COMM_WORLD);
        MPI_Send(&qValue, 1, MPI_DOUBLE, 0, VAL_MSG, MPI_COMM_WORLD);
    }
    MPI_Iprobe(0, OPT_MSG, MPI_COMM_WORLD, &tFlag, &status);
    if(tFlag) {
        MPI_Recv(&rOK, 1, MPI_INT, 0, OPT_MSG, MPI_COMM_WORLD, &status);
        if(rOK==1) bRestart=TRUE;
        //std::cout<<"Core: "<<rNumber<<" Action: "<<rOK<<std::endl;
    }
    
}
#define STMP (iNumVars+1)
//#define STMP 10

void Cooperation::ReceiveCheck(int threadID) {
#ifdef SHARE_INFO
    MPI_Status statusS;
    MPI_Request requestS;
    do {
        MPI_Iprobe(MPI_ANY_SOURCE, RECV_MSG, MPI_COMM_WORLD, &tFlag, &status);
        if(tFlag) {
            MPI_Recv(&rOK, 1, MPI_INT, status.MPI_SOURCE, RECV_MSG, MPI_COMM_WORLD,&statusS);
            ackrecv--;
#ifdef DEBUG
            if(ackrecv<0) {
                std::cout<<"Error ackrecv < 0 ("<<ackrecv<<")"<<std::endl;
                MPI_Abort(MPI_COMM_WORLD, 911);
            }
#endif
        }
    } while(tFlag);
    
    //ack recv
    if(ackrecv > 0 || !vSend) return;
    if(sInfo==SWAIT) { sInfo=SEND; SenderMachine(threadID); vSend=false;}
    else if(sInfo==SEND) sInfo=SLEEP;
#endif
}
void Cooperation::SenderMachine(int threadID) {
#ifdef SHARE_INFO

    if(sInfo!=SEND) {
        std::cout<<"Unexpected state..."<<std::endl;
        MPI_Abort(MPI_COMM_WORLD, 911);
    }

    //std::cout<<"Here ("<<threadID<<"):" <<iBestNumFalse[threadID]<<std::endl;
    int pi;
    bSend[0]=iBestNumFalse[threadID];
    for(pi=1;pi<iNumVars+1;pi++) {
        bSend[pi]=aBestVarValue[threadID][pi];
    }

    int tDest;
    for(pi=0;pi<aGroup;pi++) {
        if(pi!=threadID) {
            tDest=gNumber*aGroup+pi;
#ifdef DEBUG
            std::cout<<"Send -> TO: "<<pi<<"("<<(tDest)<<") FROM: "<<threadID<<" size: "<<STMP<<std::endl;
            std::cout<<"Val: "<<bSend[0]<<std::endl;
#endif
            MPI_Isend(bSend, STMP, MPI_INT, tDest, SEND_MSG, MPI_COMM_WORLD, &requestS);
            //MPI_Isend(aBestVarValue[threadID], STMP, MPI_INT, pi, SEND_MSG, MPI_COMM_WORLD, &request);
        }
    }
    ackrecv=aGroup-1;
#endif
}


void Cooperation::ReceiveMachine(int threadID) {
#ifdef SHARE_INFO
    //sharing information
    do {
        //int tmp = 
        MPI_Iprobe( MPI_ANY_SOURCE, SEND_MSG, MPI_COMM_WORLD, &tFlag, &statusR );
        /*for(int i=0;i<nThreads;i++) {
            MPI_Iprobe( i, SEND_MSG, MPI_COMM_WORLD, &tFlag, &statusR );
            if(tFlag) break;
        }*/

        if(tFlag) {
            int sidGroup=statusR.MPI_SOURCE%aGroup;
#ifdef DEBUG
            std::cout<<"Before recv: "<<sidGroup<<" ("<<statusR.MPI_SOURCE<<") ME: "<<threadID<<std::endl;
            int count;
            MPI_Get_count(&statusR, MPI_INT, &count);
            std::cout<<"Size: "<<count<<" Expected: "<<STMP<<std::endl;
            if(sidGroup == threadID) {
                std::cout<<"Error MSG Send == Recv ("<<threadID<<")"<<std::endl;
                //exit(0);
                MPI_Abort(MPI_COMM_WORLD, 911);
            }
#endif
            MPI_Recv(aBestVarValue[sidGroup], STMP, MPI_INT, statusR.MPI_SOURCE, SEND_MSG, MPI_COMM_WORLD, &status);
            iBestNumFalse[sidGroup]=aBestVarValue[sidGroup][0];
#ifdef DEBUG
            std::cout<<"Val: "<<iBestNumFalse[sidGroup]<<std::endl;
#endif DEBUG
            //cleaning the first bit
            aBestVarValue[sidGroup][0]=0;
            BestImp[threadID][sidGroup]=true;
            initVars[sidGroup]=true;
            //MPI_Recv(bRecv[statusR.MPI_SOURCE], STMP, MPI_INT, statusR.MPI_SOURCE, SEND_MSG, MPI_COMM_WORLD, &statusR2);
            MPI_Isend(&rOK, 1, MPI_INT, statusR.MPI_SOURCE, RECV_MSG, MPI_COMM_WORLD, &request);
        }
    }while(tFlag);
#endif
    //Ending information
    if(threadID == 0 && gNumber==0) {
        MPI_Iprobe( MPI_ANY_SOURCE, WINNER_MSG, MPI_COMM_WORLD, &tFlag, &status );
        if(tFlag) {
            MPI_Recv(&cOK, 1, MPI_INT, status.MPI_SOURCE, WINNER_MSG, MPI_COMM_WORLD, &status);
            End=true;
            winnerThread=status.MPI_SOURCE;
            std::cout<<"winner thread: "<<winnerThread<<std::endl;
        }
    }
    else {
        MPI_Iprobe(MPI_ANY_SOURCE, WINNER_MSG, MPI_COMM_WORLD, &tFlag, &status);
        if(tFlag) {
            MPI_Recv(&cOK, 1, MPI_INT, status.MPI_SOURCE, WINNER_MSG, MPI_COMM_WORLD, &status);
            End=true;
        }
    }
}
//thread 0 reports to all others that a winner has been found
void Cooperation::StopThreads(void) {
    for(int i=1;i<nThreads;i++) {
        MPI_Isend(&cOK, 1, MPI_INT, i, WINNER_MSG, MPI_COMM_WORLD, &request);
    }
}

void Cooperation::AckWinner(int _winner) {
    for(int i=1;i<nThreads;i++) {
        MPI_Send(&_winner, 1, MPI_INT, i, WINNER_ACK, MPI_COMM_WORLD);
    }
}

/*void Ubcsat::InitFalseClauseList() {
    UINT32 j;
    
    iNumFalseList = 0;
    
    for (j=0;j<iNumClauses;j++) {
        if (aNumTrueLit[j]==0) {
            aFalseList[iNumFalseList] = j;
            aFalseListPos[j] = iNumFalseList++;      
        }
    }
}*/

void Cooperation::iFalseClauses(int threadID) {
    
    UINT32 j,i,va;

    NumFalseClauses[threadID]=0;
    
    for(i=0;i<iNumVars;i++)
        aVarInfo[threadID][i]=0;
    
    for(j=0;j<iNumClauses;j++) {
        if (aNumTrueLit[j]==0) {
            ///std::cout<<j<<" ";
            /*FalseList[threadID][NumFalseClauses[threadID]]=j;
            FalseListPos[threadID][j] = NumFalseClauses[threadID]++;*/
            
            ////aClauseLen[j]         length of clause[j]
            ////aClauseLits[j][k]     literal [k] of clause[j]
            //std::cout<<"Clause: "<<j<<std::endl;
            for(i=0;i<aClauseLen[j];i++) {
                va=GetVarFromLit( pClauseLits[j][i] );
                //std::cout<<Solver[threadID].pClauseLits[j][i]<<" "<<va<<std::endl;
                if(aVarInfo[threadID][va] == 0) {
                    aVarInfo[threadID][va]=1;
                    int ones=0, zeros=0;
                    for(int vi=0;vi<nThreads;vi++) {
                        if(!initVars[vi]) continue;
                        if(aBestVarValue[vi][va]==1) ones++;
                        else zeros++;
                    }
                    double probOne=FloatToProb(static_cast<double>(ones)/static_cast<double>(ones+zeros));
                    if(RandomProb(probOne)) aVarValue[i]=1;
                    else aVarValue[i]=0;
                }
            }
            //std::cout<<std::endl;
                //aVarInfo[aClauseLits[j][i]]=1;
            ClauseFailure[threadID][j]++;
            NumFalseClauses[threadID]++;
        }
    }

}

void Cooperation::RestartClause(int threadID) {
    UINT32 iInvTheta = 6;
    FLOAT fDelta = 0.1f;
    FLOAT fProb;
    
    if (iStep-(LastAdaptStep[threadID]) > iNumClauses / iInvTheta) {
        //if (iStep-(coop->LastAdaptStep[idThread]) > 100) {
        LastAdaptStep[threadID]=iStep;
        LastAdaptNumFalse[threadID] = iNumFalse;
        bRestart=TRUE;
    } else if (iNumFalse < LastAdaptNumFalse[threadID]) {
        /* if improvement, increase smoothing probability */
        LastAdaptStep[threadID] = iStep;
        LastAdaptNumFalse[threadID] = iNumFalse;
    }
}
/*
void Cooperation::RestartBest(int threadID) {
		std::cout<<"RestartBest: "<<threadID<<std::endl;
        omp_set_lock(&read);
        for(int i=0;i<nThreads;i++) {
                omp_set_lock(&write[i]);
        }

        int numVars=Solver[threadID].iNumVars;
		
		for(int i=1;i<=numVars;i++) {
			if(!initVars[i]) continue;
			int ones=0,zeros=0;
			for(int j=0;j<nThreads;j++) {
				if(bestScore[j][i]>0) {
					if(iBestNumFalse[i]==1) ones++;
					else zeros++;
				}
			}
			if(ones>zeros) Solver[threadID].aVarValue[i]=1;
			else if(zeros>ones) Solver[threadID].aVarValue[i]=0;
			else Solver[threadID].aVarValue[i]=Solver[threadID].RandomInt(2);
		}

        for(int i=0;i<nThreads;i++) {
                omp_unset_lock(&write[i]);
        }
        omp_unset_lock(&read);

}
*/


#ifdef RESTART_PRANK

//Resting using Averaging best known solutions
void Cooperation::RestartBest(int threadID) {
		//std::cout<<"RestartBest: "<<threadID<<std::endl;
        //omp_set_lock(&read);
		int orderFit[nThreads];
        for(int i=0;i<nThreads;i++) {
				BestImp[threadID][i]=false;
				orderFit[i]=i;
        }

        int numVars=iNumVars;
		
		//sorting and assigning weights
		//applying a simple bubble sort algorithm
		int n=nThreads-1;
		for(int i=0;i<n;i++) {
			for(int j=n;j>i;j--) {
				if(  iBestNumFalse[orderFit[j-1]] > iBestNumFalse[orderFit[j]] ) {
					std::swap(orderFit[j-1], orderFit[j]);
				}
			}
		}
		
		//Best alg=nThreads, 2nd best=nThreads-1, 3rd best=nThreads-2, ..., worse=1
		int W[nThreads];
		for(int i=0;i<nThreads;i++) {
			W[orderFit[i]]=nThreads-i;
		}
		

		if(initVars[threadID]) {
			for(int i=1;i<=numVars;i++) {
				//double prob=Solver[threadID].FloatToProb(0.3);
				//if(Solver[threadID].RandomProb(prob)) { continue; }
				int ones=0,zeros=0;
				//voting with ranks values
				for(int j=0;j<nThreads;j++) {
					if(!initVars[j]) continue;
					if(aBestVarValue[j][i]==1) ones+=W[j];
					else zeros+=W[j];
				}
				//if(ones>zeros) Solver[threadID].aVarValue[i]=1;
				//else if(zeros>ones) Solver[threadID].aVarValue[i]=0;
				//else Solver[threadID].aVarValue[i]=Solver[threadID].RandomInt(2);
				double probOne=FloatToProb(static_cast<double>(ones)/static_cast<double>(ones+zeros));
				if(RandomProb(probOne)) aVarValue[i]=1;
				else aVarValue[i]=0;
			}
		}

}

#endif

#ifdef RESTART_PNormW
//Resting using voting
void Cooperation::RestartBest(int threadID) {
#ifdef DEBUG
    std::cout<<"RestartBest: "<<threadID<<" Group: "<<gNumber<<" idGroup: "<<idGroup<<std::endl;
#endif
    /*for(int i=0;i<nThreads;i++) {
        std::cout<<iBestNumFalse[i]<<" - ";
    }
    std::cout<<std::endl;*/
        ///omp_set_lock(&read);
        for(int i=0;i<aGroup;i++) {
				BestImp[threadID][i]=false;
        }

        int numVars=iNumVars;
		double W[nThreads];

		double numClauses=static_cast<double>(iNumClauses);
		for(int i=0;i<aGroup;i++) {
			W[i]=(numClauses - static_cast<double>(iBestNumFalse[i]))/numClauses;
			//W[i]=W[i]*W[i];
		}

		if(initVars[threadID]) {
			for(int i=1;i<=numVars;i++) {
				//double prob=Solver[threadID].FloatToProb(0.3);
				//if(Solver[threadID].RandomProb(prob)) { continue; }
				double ones=0,zeros=0;
				//voting weigthing using fitness function
				for(int j=0;j<aGroup;j++) {
					if(!initVars[j]) continue;
					if(aBestVarValue[j][i]==1) ones+=W[j];
					else zeros+=W[j];
				}
				//if(ones>zeros) Solver[threadID].aVarValue[i]=1;
				//else if(zeros>ones) Solver[threadID].aVarValue[i]=0;
				//else Solver[threadID].aVarValue[i]=Solver[threadID].RandomInt(2);
				double probOne=FloatToProb(static_cast<double>(ones)/static_cast<double>(ones+zeros));
				if(RandomProb(probOne)) aVarValue[i]=1;
				else aVarValue[i]=0;
			}
		}

        //omp_unset_lock(&read);
}

#endif

#ifdef RESTART_RANK

//Resting using Averaging best known solutions
void Cooperation::RestartBest(int threadID) {
		//std::cout<<"RestartBest: "<<threadID<<std::endl;
        //omp_set_lock(&read);
		int orderFit[nThreads];
        for(int i=0;i<nThreads;i++) {
				BestImp[threadID][i]=false;
				orderFit[i]=i;
        }

        int numVars=iNumVars;
		
		//sorting and assigning weights
		//applying a simple bubble sort algorithm
		int n=nThreads-1;
		for(int i=0;i<n;i++) {
			for(int j=n;j>i;j--) {
				if(  iBestNumFalse[orderFit[j-1]] > iBestNumFalse[orderFit[j]] ) {
					std::swap(orderFit[j-1], orderFit[j]);
				}
			}
		}
		
		int W[nThreads];
		for(int i=0;i<nThreads;i++) {
			W[orderFit[i]]=nThreads-i;
		}
		

		if(initVars[threadID]) {
		for(int i=1;i<=numVars;i++) {
			//double prob=Solver[threadID].FloatToProb(0.3);
			//if(Solver[threadID].RandomProb(prob)) { continue; }
			int ones=0,zeros=0;
			//voting with ranks values
			for(int j=0;j<nThreads;j++) {
				if(!initVars[j]) continue;
				if(aBestVarValue[j][i]==1) ones+=W[j];
				else zeros+=W[j];
			}
			if(ones>zeros) aVarValue[i]=1;
			else if(zeros>ones) aVarValue[i]=0;
			else aVarValue[i]=RandomInt(2);
		}
		}
}

#endif

#ifdef RESTART_BEST
//Restarting using always best Fitness value
void Cooperation::RestartBest(int threadID) {
        omp_set_lock(&read);
        for(int i=0;i<nThreads;i++) {
                omp_set_lock(&write[i]);
				BestImp[threadID][i]=false;
        }
        int numVars=iNumVars;
		
		UINT32 *VarBest=NULL;
		UINT32 BestFitness=10000000;
		for(int i=0;i<nThreads;i++) {
			if(!initVars[i]) continue;
			if(BestFitness > iBestNumFalse[i]) {
				BestFitness=iBestNumFalse[i];
				VarBest=aBestVarValue[i];
			}
		}
		if(initVars[threadID]) {
			for(int i=1;i<=numVars;i++) {
				aVarValue[i]=VarBest[i];
			}
		}
        for(int i=0;i<nThreads;i++) {
			omp_unset_lock(&write[i]);
        }
        omp_unset_lock(&read);
}
#endif

#ifdef RESTART_PROB
//Fix variables that agree, take others at random biased towards the majority
void Cooperation::RestartBest(int threadID) {
		//std::cout<<"RestartBest: "<<threadID<<std::endl;
        //omp_set_lock(&read);
		int orderFit[nThreads];
        for(int i=0;i<nThreads;i++) {
                omp_set_lock(&write[i]);
				BestImp[threadID][i]=false;
				orderFit[i]=i;
        }

        int numVars=iNumVars;

		if(initVars[threadID]) {
			for(int i=1;i<=numVars;i++) {
				//double prob=Solver[threadID].FloatToProb(0.3);
				//if(Solver[threadID].RandomProb(prob)) { continue; }
				int ones=0,zeros=0;
				//voting with ranks values
				for(int j=0;j<nThreads;j++) {
					if(!initVars[j]) continue;
					if(aBestVarValue[j][i]==1) ones++;
					else zeros++;
				}
				//if all variables have the same value then the probability for selecting that value would be 1.
				double probOne=FloatToProb(static_cast<double>(ones)/static_cast<double>(ones+zeros));
				if(RandomProb(probOne)) aVarValue[i]=1;
				else aVarValue[i]=0;
			}
		}
        for(int i=0;i<nThreads;i++) {
			omp_unset_lock(&write[i]);
        }
        //omp_unset_lock(&read);
}

#endif

#ifdef RESTART_Norm
//Resting using voting
void Cooperation::RestartBest(int threadID) {
		//std::cout<<"RestartBest: "<<threadID<<std::endl;
        //omp_set_lock(&read);
        for(int i=0;i<nThreads;i++) {
                omp_set_lock(&write[i]);
				BestImp[threadID][i]=false;
        }

        int numVars=iNumVars;
		double W[nThreads];

		double numClauses=static_cast<double>(iNumClauses);
		for(int i=0;i<nThreads;i++) {
			W[i]=(numClauses - static_cast<double>(iBestNumFalse[i]))/numClauses;
			//W[i]=W[i]*W[i];
		}

		if(initVars[threadID]) {
			for(int i=1;i<=numVars;i++) {
				//double prob=Solver[threadID].FloatToProb(0.3);
				//if(Solver[threadID].RandomProb(prob)) { continue; }
				double ones=0,zeros=0;
				//voting weigthing using fitness function
				for(int j=0;j<nThreads;j++) {
					if(!initVars[j]) continue;
					if(aBestVarValue[j][i]==1) ones+=W[j];
					else zeros+=W[j];
				}
				if(ones>zeros) aVarValue[i]=1;
				else if(zeros>ones) aVarValue[i]=0;
				else aVarValue[i]=RandomInt(2);
			}
		}
        for(int i=0;i<nThreads;i++) {
			omp_unset_lock(&write[i]);
        }
        //omp_unset_lock(&read);
}
#endif

#ifdef RESTART_AGREE

//Fix variables that agree, take others at random
void Cooperation::RestartBest(int threadID) {
		//std::cout<<"RestartBest: "<<threadID<<std::endl;
        //omp_set_lock(&read);
		int orderFit[nThreads];
        for(int i=0;i<nThreads;i++) {
                omp_set_lock(&write[i]);
				BestImp[threadID][i]=false;
				orderFit[i]=i;
        }

        int numVars=iNumVars;
		

		if(initVars[threadID]) {
			for(int i=1;i<=numVars;i++) {
				//double prob=Solver[threadID].FloatToProb(0.3);
				//if(Solver[threadID].RandomProb(prob)) { continue; }
				int ones=0,zeros=0;
				//voting with ranks values
				for(int j=0;j<nThreads;j++) {
					if(!initVars[j]) continue;
					if(aBestVarValue[j][i]==1) ones++;
					else zeros++;
				}
				if(ones==nThreads) aVarValue[i]=1;
				else if(zeros==nThreads) aVarValue[i]=0;
				else aVarValue[i]=RandomInt(2);
			}
		}
        for(int i=0;i<nThreads;i++) {
			omp_unset_lock(&write[i]);
        }
        //omp_unset_lock(&read);
}

#endif

#ifdef RESTART_MAJORITY

//Fix variables that agree, take others at random
void Cooperation::RestartBest(int threadID) {
		//std::cout<<"RestartBest: "<<threadID<<std::endl;
        //omp_set_lock(&read);
		int orderFit[nThreads];
        for(int i=0;i<nThreads;i++) {
                omp_set_lock(&write[i]);
				BestImp[threadID][i]=false;
				orderFit[i]=i;
        }

        int numVars=iNumVars;
		

		if(initVars[threadID]) {
			for(int i=1;i<=numVars;i++) {
				//double prob=Solver[threadID].FloatToProb(0.3);
				//if(Solver[threadID].RandomProb(prob)) { continue; }
				int ones=0,zeros=0;
				for(int j=0;j<nThreads;j++) {
					if(!initVars[j]) continue;
					if(aBestVarValue[j][i]==1) ones++;
					else zeros++;
				}
				if(ones>zeros) aVarValue[i]=1;
				else if(zeros>ones) aVarValue[i]=0;
				else aVarValue[i]=RandomInt(2);
			}
		}
        for(int i=0;i<nThreads;i++) {
			omp_unset_lock(&write[i]);
        }
        //omp_unset_lock(&read);
}

#endif

#ifdef RESTART_AVERAGE
//Resting using Averaging best known solutions
void Cooperation::RestartBest(int threadID) {
		//std::cout<<"RestartBest: "<<threadID<<std::endl;
        //omp_set_lock(&read);
        for(int i=0;i<nThreads;i++) {
                omp_set_lock(&write[i]);
				BestImp[threadID][i]=false;
        }

        int numVars=iNumVars;
		
		UINT32 *VarBest=NULL;
		UINT32 BestFitness=10000000;
		/*for(int i=0;i<nThreads;i++) {
			if(!initVars[i]) continue;
			if(BestFitness > iBestNumFalse[i]) {
				BestFitness=iBestNumFalse[i];
				VarBest=aBestVarValue[i];
			}
		}*/
		if(initVars[threadID]) {
		for(int i=1;i<=numVars;i++) {
			double prob=FloatToProb(0.3);
			if(RandomProb(prob)) { continue; }
			int ones=0,zeros=0;
			//average but only taking into account important values
			/*for(int j=0;j<nThreads;j++) {
				if(!initVars[j]) continue;
				if(bestScore[j][i]>0) {
					if(aBestVarValue[j][i]==1) ones++;
					else zeros++;
				}
			}*/
			//only average
			for(int j=0;j<nThreads;j++) {
				if(!initVars[j]) continue;
				if(aBestVarValue[j][i]==1) ones++;
				else zeros++;
			}
			if(ones>zeros) aVarValue[i]=1;
			else if(zeros>ones) aVarValue[i]=0;
			else aVarValue[i]=RandomInt(2);
		}
		}
        for(int i=0;i<nThreads;i++) {
			omp_unset_lock(&write[i]);
        }
        //omp_unset_lock(&read);
}

#endif

void Cooperation::InitSeed(int _seed) {
	globalSeed=_seed;
	sgenrand(globalSeed, &mt);
}

void Cooperation::UpdateStep(bool _send) {
    //std::cout<<"UpdateStep"<<std::endl;
	//omp_set_lock(&read);

    int threadID=idGroup;
#ifdef SHARE_INFO
    if(_send) vSend=true;
	if(!initVars[threadID]) {
        //pVector[threadID]=new double[numVars+1];
        
		iBestNumFalse[threadID] = iNumFalse;
		int numVars=iNumVars;
		//aBestVarValue[threadID]=new UINT32[numVars+1];
		//bestScore[threadID]=new SINT32[numVars+1];
		//FlipCount[threadID]=new UINT32[numVars+1];
		for(int i=0;i<=numVars;i++) {
            //pVector[threadID][i]=0.5;
			aBestVarValue[threadID][i]=aVarValue[i];
			/*if(Solver[threadID].aVarScore)
				bestScore[threadID][i]=Solver[threadID].aVarScore[i];
			else if(Solver[threadID].aMakeCount && Solver[threadID].aBreakCount)
				bestScore[threadID][i]=Solver[threadID].aMakeCount-Solver[threadID].aBreakCount;*/
			//else assert(false);
		}
		//isLocalMinimum[threadID]=Solver[threadID].IsLocalMinimum(false);
		//std::cout<<"UpdateStep: "<<isLocalMinimum[threadID]<<std::endl;
		initVars[threadID]=true;
	}
	if ( iBestNumFalse[threadID] > iNumFalse ) {
        sInfo=SWAIT;
        /*if(sInfo==SLEEP) {
            sInfo=SEND;
            SenderMachine(threadID);
        }
        else if(sInfo==SEND) sInfo=SWAIT;*/
        //std::cout<<Solver[threadID].iNumFalse<<" -- ";
        //SenderMachine(threadID, Solver[threadID].iStep>10000?true:false);
		//std::cout<<"Updating... "<<Solver[threadID].iNumFalse<<std::endl;
		//broadcasting the improvement to all available threads
		for(int i=0;i<aGroup;i++) BestImp[i][threadID]=true;
		int numVars=iNumVars;
		iBestNumFalse[threadID] = iNumFalse;
		for(int i=0;i<=numVars;i++) {
			aBestVarValue[threadID][i]=aVarValue[i];
			//FlipCount[threadID][i]=Solver[threadID].aFlipCounts[i];
			//if( MaxFlips[threadID] < FlipCount[threadID][i] ) MaxFlips[threadID]=FlipCount[threadID][i];
			/*if(Solver[threadID].aVarScore)
				bestScore[threadID][i]=Solver[threadID].aVarScore[i];
			else if(Solver[threadID].aMakeCount && Solver[threadID].aBreakCount)
				bestScore[threadID][i]=Solver[threadID].aMakeCount-Solver[threadID].aBreakCount;*/
			//else assert(false);
		}
		//isLocalMinimum[threadID]=Solver[threadID].IsLocalMinimum(false);
		//std::cout<<"UpdateStep: "<<isLocalMinimum[threadID]<<std::endl;
	}
#endif
    ReceiveMachine(threadID);
    ReceiveCheck(threadID);

	//omp_unset_lock(&read);
}

void parsingNumber(int n,int m) {
	int tvars=m;
	while(true) {
		tvars--;
		if(n<2) {
			std::cout<<n<<" ";
			break;
		}
		
		int tmp=n%2;
		std::cout<<tmp<<" ";
		n=n/2;
	}
	while(tvars>0) {
		tvars--;
		std::cout<<"0 ";
	}
	std::cout<<std::endl;
}


void Cooperation::PrintBest(int threadID) {
	if(!initVars[threadID]) return;
	int numVars=iNumVars;
	std::cout<<"Cooperation best["<<threadID<<"]"<<std::endl;
	for(int i=1;i<=numVars;i++) 
		std::cout<<aBestVarValue[threadID][i];
	std::cout<<std::endl;
        for(int i=0;i<nThreads;i++) {
                std::cout<<"c Min clauses Core["<<i<<"]: "<<iBestNumFalse[i]<<std::endl;
        }

}

void Cooperation::PrintSol(void) {
	if(winnerThread==-1) return;
	if(!initVars[winnerThread]) return;
	int numVars=iNumVars;
	std::cout<<"s SATISFIABLE"<<std::endl;
	std::cout<<"v ";
	for(int i=1;i<=numVars;i++) {
		std::cout<<(aBestVarValue[winnerThread][i]==0?"-":"")<<i<<" ";
	}
	//std::cout<<std::endl<<"endSol"<<std::endl;
	std::cout<<"0"<<std::endl;
        for(int i=0;i<nThreads;i++) {
                std::cout<<"c Min false clauses Core["<<i<<"]: "<<iBestNumFalse[i]<<std::endl;
        }

}

int Cooperation::HammingDistance(int id1, int id2) {
	if(!initVars[id1] || !initVars[id2]) return -1;
	int numVars=iNumVars;
	int hDistance=0;
	for(int i=1;i<=numVars;i++) {
		if(aBestVarValue[id1][i] != aBestVarValue[id2][i]) hDistance++;
	}
	return hDistance;
}

//this method isn't efficient, but here we don't care about efficiency.
void Cooperation::addHammingDistance(void) {
	int hdTotal=0;
	int total=0;
	for(int i=0;i<nThreads;i++) {
		for(int j=i+1;j<nThreads;j++) {
			hdTotal+=HammingDistance(j,i);
			total++;
		}
	}
	HammingDis.push_back(static_cast<double>(hdTotal)/static_cast<double>(total));
}

void Cooperation::printHammingDistanceStats(void) {
	int max=-1, min=10000000, tValues=0, maxId1=-1, maxId2=-1;
	double mean=0;
	for(int i=0;i<nThreads;i++) {
		for(int j=i+1;j<nThreads;j++) {
			int tmp=HammingDistance(i,j);
			if(tmp==-1) continue;
			if(max < tmp) { max=tmp; maxId1=i; maxId2=j;}
			if(min > tmp) { min=tmp; }
			mean+=(double)tmp;
			tValues++;
		}
	}
	std::cout<<"Max: "<<max<<" "<<maxId1<<","<<maxId2<<std::endl
			 <<"Mean: "<<(mean/(double)tValues)<<std::endl;
}
    
}

/*void Cooperation::setEnd(void) {
#pragma omp critical (accessToEnd)
	{
		End=true;
	}
}

bool Cooperation::isSolution(void) {
	return End;
}*/
