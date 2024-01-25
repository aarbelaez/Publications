/*

      ##  ##  #####    #####   $$$$$   $$$$   $$$$$$    
      ##  ##  ##  ##  ##      $$      $$  $$    $$      
      ##  ##  #####   ##       $$$$   $$$$$$    $$      
      ##  ##  ##  ##  ##          $$  $$  $$    $$      
       ####   #####    #####  $$$$$   $$  $$    $$      
  ======================================================
  SLS SAT Solver from The University of British Columbia
  ======================================================
  ...Developed by Dave Tompkins (davet [@] cs.ubc.ca)...
  ------------------------------------------------------
  .......consult legal.txt for legal information........
  ......consult revisions.txt for revision history......
  ------------------------------------------------------
  ... project website: http://www.satlib.org/ubcsat ....
  ------------------------------------------------------
  .....e-mail ubcsat-help [@] cs.ubc.ca for support.....
  ------------------------------------------------------

*/

#include "mpi.h"

#include "ubcsat.h"
#include "Cooperation.h"

#include <omp.h>


#ifdef __cplusplus 
namespace ubcsat {
#endif

    int idThread;
//int ubcsatmain(int argc, char *argv[], Cooperation *coop) {
  int Solve(int argc, char *argv[], Cooperation *coop) {
  InitSeed();

  SetupUBCSAT();

  AddAlgorithms();
  AddParameters();
  AddReports();
  AddDataTriggers();
  AddReportTriggers();

  AddLocal();
  
  ParseAllParameters(argc,argv);

  ActivateAlgorithmTriggers();
  ActivateReportTriggers();

  RandomSeed(iSeed);

  RunProcedures(PostParameters);

  RunProcedures(ReadInInstance);

  RunProcedures(PostRead);

  RunProcedures(CreateData);
  RunProcedures(CreateStateInfo);

      coop->InitInfo(idThread);

  iRun = 0;
  iNumSolutionsFound = 0;
  bTerminateAllRuns = 0;

  RunProcedures(PreStart);

  StartTotalClock();
      
  MPI_Barrier( MPI_COMM_WORLD );

      /*if(coop->idGroup<2) {
          std::cout<<"idGroup: "<<coop->idGroup<<" gNumber:"<<coop->gNumber<<" fp: "<<( ( 1.0/static_cast<FLOAT>(coop->nThreads/coop->aGroup) ) / static_cast<FLOAT>(  (coop->idGroup+1) ) )<<std::endl;
          FLOAT fp=ProbToFloat(iPs);
          fp*= ( ( 1.0/static_cast<FLOAT>(coop->nThreads/coop->aGroup) ) / static_cast<FLOAT>(  (coop->idGroup+1) ) ) ;
          iPs=FloatToProb(fp);
      }*/
      
  double cTime=MPI_Wtime();

      double timeInterval=10;
      double timeIncrease=(cTime-coop->tStart)+timeInterval;
      
  //MPI_Abort(MPI_COMM_WORLD, 111);
      
  while ((!coop->End) &&  (iRun < iNumRuns) && (! bTerminateAllRuns)) {

    iRun++;

    iStep = 0;
    bSolutionFound = 0;
    bTerminateRun = 0;
    bRestart = 1;

    RunProcedures(PreRun);

    StartRunClock();
    
    while ((!coop->End) && (iStep < iCutoff) && (! bSolutionFound) && (! bTerminateRun)) {

      iStep++;
      iFlipCandidate = 0;

      RunProcedures(PreStep);
      RunProcedures(CheckRestart);
        
        /*if(bRestart && iStep!=1) {
            //std::cout<<"iStep: "<<iStep<<std::endl;
            //valid restart =>  at least one of the available cores (but not me) improved its own solution
            
            bRestart=FALSE;
            for(int i=0;i<coop->aGroup;i++)
                if(coop->BestImp[coop->idGroup][i] && i != coop->idGroup) bRestart=true;
            
        }*/

      if (bRestart) {
        RunProcedures(PreInit);
          
#ifdef SHARE_INFO
          if(iStep == 1) {
              RunProcedures(InitData);
          }
          else {
              //MPI_Abort(MPI_COMM_WORLD, 111);
              //if(idThread==omp_get_thread_num()) coop->addHammingDistance();
              coop->RestartBest(coop->idGroup);
              //coop->iFalseClauses(idThread);
              iNumRestarts++;
          }
#else
          //std::cout<<"Restarting..."<<std::endl;
          RunProcedures(InitData);
          coop->qValue=static_cast<double>(iNumClauses);
#endif
        RunProcedures(InitStateInfo);
        RunProcedures(PostInit);
        bRestart = 0;
      } else {
        RunProcedures(ChooseCandidate);
        RunProcedures(PreFlip);
        RunProcedures(FlipCandidate);
        RunProcedures(UpdateStateInfo);
        RunProcedures(PostFlip);
      }
      
      RunProcedures(PostStep);

        coop->UpdateStep((iStep%10000==0) );

      RunProcedures(StepCalculations);

      RunProcedures(CheckTerminate);
        
    

        cTime=MPI_Wtime();
        if(cTime-coop->tStart >= coop->tLimit) {
            bTerminateAllRuns=TRUE;
            bTerminateRun=TRUE;
        }
        
        coop->qValue=(1.0-coop->alpha)*coop->qValue+coop->alpha*static_cast<double>(iNumFalse);
        if(coop->rNumber==0) {
            if(cTime-coop->tStart >= timeIncrease) {
                std::cout<<"Time: "<<cTime-coop->tStart<<std::endl;
                coop->GetObjFun();
                timeIncrease+=timeInterval;
            }
        }
        else {
            if(iStep%100==0)
                coop->RestartConfirmation();
        }
    }

    StopRunClock();

    RunProcedures(RunCalculations);
    
    RunProcedures(PostRun);

    if (bSolutionFound) {
        if(!coop->End) {
			coop->End=true;
		}
      iNumSolutionsFound++;
      if (iNumSolutionsFound == iFind) {
        bTerminateAllRuns = 1;
      }
    }
  }

  StopTotalClock();

  RunProcedures(FinalCalculations);

  //RunProcedures(FinalReports);

  //CleanExit();
      //comunicando al 0
      if(idThread != 0) {
          int cOK=1;
          MPI_Isend(&cOK, 1, MPI_INT, 0, WINNER_MSG, MPI_COMM_WORLD, &coop->request);
      }
      else {
          if(coop->winnerThread==-1 && iNumFalse == 0)
              coop->winnerThread=idThread;
          coop->StopThreads();
      }
      //std::cout<<"Barrier reached"<<std::endl;
      MPI_Barrier( MPI_COMM_WORLD );
      
      if(idThread==0) {
          coop->AckWinner(coop->winnerThread);
          //std::cout<<"Winner thread: "<<coop->winnerThread<<std::endl;
      }
      else {
          MPI_Recv(&coop->winnerThread, 1, MPI_INT, 0, WINNER_ACK, MPI_COMM_WORLD, &coop->status);
      }
      
      StopTotalClock();
      if(coop->winnerThread == idThread) {
          std::cout<<"Winner thread: "<<idThread<<std::endl;
          //std::cout<<"printing thread: "<<omp_get_thread_num()<<std::endl;
          //RunProceduresALEJOclass(FinalCalculations,this);
          ///RunProceduresALEJOclass(FinalReports,this);
          //std::cout<<"ending thread: "<<omp_get_thread_num()<<std::endl;
          CleanExit();
      }
      else {
          FreeRAM();
      }
      //int bsize=(iNumVars+1)*coop->nThreads;
      //MPI_Buffer_detach(coop->buffer, &bsize);
      
      //std::cout<<"BEST found: "<<coop->iBestNumFalse[idThread]<<std::endl;
      //coop->PrintBest(idThread);
      MPI_Barrier( MPI_COMM_WORLD );
      std::cout<<"Solve END: "<<idThread<<" -- "<<coop->iBestNumFalse[coop->idGroup]<<" iStep: "<<iStep<<" iCutoff " <<iCutoff<<" END: "<<coop->End<<" bSolutionFound: "<< bSolutionFound<<" iNumFalse: "<<iNumFalse<<" sInfo: "<<coop->sInfo<<std::endl;

  return(0);
  
}

char *myargv[100];
int myargc = 0;

/*
int sparrowmain(int argc, char *argv[]) {

  if (argc != 3) {
    printf("ERROR Competition build requires 2 (and only 2) parameters: filename.cnf and seed\n");
    exit(0);
  }

  myargv[myargc++] = argv[0];

  myargv[myargc++] = "-i";
  myargv[myargc++] = argv[1];

  myargv[myargc++] = "-seed";
  myargv[myargc++] = argv[2];

  myargv[myargc++] = "-q";

  myargv[myargc++] = "-r";
  myargv[myargc++] = "satcomp";

  myargv[myargc++] = "-cutoff";
  myargv[myargc++] = "max";

  myargv[myargc++] = "-alg";
  myargv[myargc++] = "sparrow";

  myargv[myargc++] = "-v";
  myargv[myargc++] = "sat11";

  return(ubcsatmain(myargc,myargv));
}
*/

#ifdef __cplusplus
}
#endif


#ifdef __cplusplus

/*int main(int argc, char *argv[]) {
    return(ubcsatmain(argc,argv));
}*/

/*
int main(int argc, char *argv[]) {
  return(ubcsat::sparrowmain(argc,argv));
}
*/

#ifndef TLIMIT
#define TLIMIT 120
#endif


namespace ubcsat {
    
UINT32 getNewSeed(struct mt19937p *mt) {
    UINT32 newSeed;
    newSeed = genrand(mt);
    return newSeed;
}
    
int main2(int argc, char *argv[], int rank, int tGroups, char pAlg[][20]) {
	int n=atoi(argv[1]);
	Cooperation c(n,tGroups,rank);

    c.tLimit=TLIMIT;
    c.tStart=MPI_Wtime();
	////cinfo=&c;
	int ncores=n;
	//signal(SIGTERM, exitInfo);
	//omp_set_num_threads(n);
    
    //"sparrow",
	//char *pAlg[8]={"paws","g2wsat+p","adaptg2wsat","adaptg2wsat+p","g2wsat","saps","rsaps","adaptnovelty+"};
    //char *pAlg[9]={"sparrow","paws","g2wsat+p","adaptg2wsat","adaptg2wsat+p","g2wsat","saps","rsaps","adaptnovelty+"};
    
	int nTotalPar=argc-1;

    char *algName=new char[100];
    char sparrow11[]="sparrow11";

    if(strcmp(sparrow11, pAlg[rank]) == 0) {
        const char *sparrowTMP="sparrow";
        strcpy(algName,sparrowTMP);
        nTotalPar+=2;
        argc+=2;
    }
    else {
        strcpy(algName,pAlg[rank]);
    }
    
	char **gInfo=new char*[nTotalPar];
	char pSeed[]="-seed";
    
	for(int i=0,j=0;i<argc;i++) {
		if(strcmp(argv[i],pSeed) == 0) {
			c.InitSeed(atoi(argv[i+1]));
		}
		if(i!=1) {
			gInfo[j]=argv[i];
			j++;
		}
	}
	if(c.globalSeed==-1) {
		////InitSeed(c.globalSeed);
		c.InitSeed(c.globalSeed);
	}
    /*UINT32 lSeeds[n];
     for(int i=0;i<=n;i++) {
     lSeeds[i]=getNewSeed(&c.mt);
     }*/
    UINT32 lSeed=0;
    for(int i=0;i<=rank;i++)
        lSeed=getNewSeed(&c.mt);
	StartRunClock();
    //#pragma omp parallel
    //	{
   // std::cout<<"Rank: "<<rank<<" Group: "<<c.gNumber<<" idGroup: "<<c.idGroup<<std::endl;

    std::cout<<"Alg["<<rank<<"]: "<<pAlg[rank]<<" Seed: "<<lSeed
             <<" Group: "<<c.gNumber<<" idGroup: "<<c.idGroup
             <<std::endl;
    //int cThread=1;
    //int cThread=rank;
    int cThread;//=omp_get_thread_num();
    cThread=rank;


    char *pPar[nTotalPar];
    //c.Solver[cThread].iSeed=lSeeds[cThread];
    iSeed=lSeed;
    int indexAlg=-1;
    for(int i=0;i<nTotalPar;i++) {
        char ptest[]="-alg";
        if(strcmp(gInfo[i],ptest) == 0) {
            //std::cout<<"parameter found ... "<<argv[i+1]<<std::endl;
            //pPar[i]=gInfo[i]; pPar[i+1]=pAlg[cThread];
            pPar[i]=gInfo[i]; pPar[i+1]=algName;
            indexAlg=i+1;
            i++;
        }
        else if(strcmp(gInfo[i],pSeed) == 0) {
            char tmpSeed[100];
            sprintf(tmpSeed,"%d",iSeed);
            pPar[i]=gInfo[i]; pPar[i+1]=tmpSeed;
            i++;
        }
        else pPar[i]=gInfo[i];
        ////std::cout<<"par["<<i<<"]: "<<pPar[i]<<std::endl;
    }

    idThread=cThread;
    //c.write=&c.write[cThread];
    //c.read=&c.read;
    /*checking the input for each algorithm
    for(int i=0;i<nTotalPar;i++)
        std::cout<<pPar[i]<<" - ";
    std::cout<<std::endl;
    */
    Solve(nTotalPar,pPar,&c);
    //	}
    StopRunClock();
	//std::cout<<"Time: "<<fTotalTime<<std::endl;
	bool solved=false;
	for(int i=0;i<c.aGroup;i++) {
		if(c.iBestNumFalse[i] == 0) solved=true;
	}
	//c.PrintSol();
    
    if(rank==0) {
        double tEnd=MPI_Wtime();
        std::cout<<"TIME= "<<(tEnd-c.tStart)<<std::endl;
    }   
    return 0;
}


}

//def => use a default list
//alg1, cores1, ..., algN, coresN
//algN Number => use algorithms position number (available for testing only)
//alg, alg, ..., alg. => will try to allocate equal resources to each algorithms
//char **readAlgs(char **aList, int _tElems, int tCores) {
void readAlgs(char **aList, int _tElems, int tCores, char output[][20]) {
    int tElems=_tElems-4;
    int fElem=4;
    char *def="def";
    char *algN="algN";
    const int tdAlg=28;
    const char *dAlg[tdAlg]={"sparrow11","paws","adaptg2wsat","g2wsat","adaptnovelty+", "crwalk","danov+", "dcrwalk","ddfw","gsat","gsat-tabu", "gwsat","hsat","hwsat","irots","novelty+","novelty++","novelty+p","rgsat","rnovelty","rnovelty+","rots","samd","urwalk","vw1","vw2","walksat","walksat-tabu"};
    if(strcmp(def, aList[fElem]) == 0) {
        int i=0;
        while(true) {
            for(int j=0;j<tdAlg;i++,j++) {
                if(i>tCores) return;
                strcpy(output[i],dAlg[j]);
            }
        }
    }
    if(strcmp(algN,aList[fElem]) == 0) {
        if(tElems != 2) {
            std::cout<<"Usage error 3"<<std::endl;
            MPI_Abort(MPI_COMM_WORLD, 911);
        }
        std::cout<<"Ele: "<<aList[fElem+1]<<std::endl;
        int aNumber=atoi(aList[fElem+1]);
        for(int i=0;i<tCores;i++) {
            strcpy(output[i],dAlg[aNumber]);
        }
        return;
    }
    
    if(tElems >= 1) {
        bool opt=true;
        if(tElems >= 2 && atoi(aList[fElem+1]) != 0) opt=false;
        if(opt) {
            int index=0;
            while(true) {
                for(int j=0;j<tElems;j++,index++) {
                    if(index>tCores) return;
                    strcpy(output[index],aList[fElem+j]);
                }
            }
        }
    }
    
    int j=0;
    for(int i=0;i<tElems;i+=2) {
        int tAlg_i=atoi(aList[fElem+i+1]);
        for(int p=0;p<tAlg_i;p++,j++) {
            if(j>tCores) {
                std::cout<<"Usage error 2"<<std::endl;
                MPI_Abort(MPI_COMM_WORLD, 911);
            }
            strcpy(output[j],aList[fElem+i]);
        }
    }
    if(j!=tCores) {
        std::cout<<"Usage error 3 assigned resources: "<<j<<" total resources: "<<tCores<<std::endl;
        MPI_Abort(MPI_COMM_WORLD, 911);
    }
}

//only need to check how to remove all the printf listed for each individual solver....
//It's sooo important to check the input seed number, this number should be an interger seed_i such that 0<seed_i<Max_int(32 bits).... otherwise the solver might crash...
int main(int argc, char *argv[]) {
    
    int rank, size;
    
    MPI_Init( &argc, &argv );
    
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    
	//std::cout<<argv[0]<<std::endl;
	//std::cout<<argv[1]<<" -- "<<argv[2]<<" -- "<<argv[3]<<std::endl;
    char *pmaxcores="-maxcores";
    int initPar=1; int maxcores=10000;
    
    if ( strcmp(pmaxcores,argv[1]) == 0 ) {
        maxcores=atoi(argv[2]);
        initPar=3;
    }
    char pseed[100], pcores[100];
    char *inputFile=argv[initPar]; initPar++;
    int mncores=atoi(argv[initPar]); initPar++;
    int nseed=atoi(argv[initPar]); initPar++;
    sprintf(pseed,"%d",(nseed<=0?1:nseed));
    //sprintf(pcores,"%d",(mncores>maxcores?maxcores:mncores));
    sprintf(pcores,"%d",size);
    
    int tcores=size;
    
    char pAlg[tcores][20];
    readAlgs(argv, argc, tcores,pAlg);
    if(rank==0) {
        for(int i=0;i<tcores;i++) {
            std::cout<<pAlg[i]<<" ";
        }
        std::cout<<std::endl;
    }
    
    ///std::cout<<"c Input seed "<<nseed<<" used seed: "<<pseed<<std::endl;
    //char *pseed=argv[3]; char *inputFile=argv[1]; char *pcores=argv[2];
    int time=1000;
    int cores=0;
    char *pargv[] = {"ubcsat",
        pcores,"-alg","coop-ubcsat","-i",inputFile,
        "-cutoff","max", 
#ifdef SHARE_INFO
        "-srestart","1000000",
#endif
        "-seed",pseed,
        "-solve",
        "-r", "out", "null", "-r", "stats", "null",
        "-v","sat11"
    };
#ifdef SHARE_INFO
    int tparam=19;
#else
    //int tparam=11;
    int tparam=17;
#endif
    //char **tmp = readAlgs(argv, argc, tcores);
    
    int tmpMain=ubcsat::main2(tparam,pargv, rank, mncores, pAlg);
    MPI_Finalize( );
    return tmpMain;
    //return main2(tparam,pargv, rank);
    //return ubcsatMain(tparam, pargv);
    //return 1;
}

int main3(int argc, char *argv[]) {
    int rank, size;
    
    MPI_Init( &argc, &argv );
    
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Comm_size( MPI_COMM_WORLD, &size );
    
    for(int i=0;i<argc;i++)
        std::cout<<argv[i]<<" ";
    std::cout<<std::endl;

    int tmp=ubcsat::Solve(argc,argv,NULL);
    MPI_Finalize( );
    return tmp;

}

#else

int main(int argc, char *argv[]) {
    return(ubcsatmain(argc,argv));
}

/*int main(int argc, char *argv[]) {
  return(sparrowmain(argc,argv));
}*/

#endif
