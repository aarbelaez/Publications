

#ifndef Cooperation__H
#define Cooperation__H

#include "ubcsat.h"
#include "vector"
#include "mt19937p.h"

#define TRUE 1
#define FALSE 0

enum sState { SLEEP, SEND, SWAIT };

#define SOLUTION 2
#define WINNER_SIGNAL 3
#define WINNER_CONFIRMATION 4
#define ASKVAL_SIGNAL 5

#define SEND_MSG 0
#define RECV_MSG 1
#define WINNER_MSG 2
#define WINNER_RECV_MSG 3
#define WINNER_ACK 4

#define GET_VAL_MSG 5
#define VAL_MSG 6
#define OPT_MSG 7

namespace ubcsat {
    
class CoreInfo {
public:
    double qValue;
    int iNumFalse;
    int idCore;
    int restart;
    /*void setInfo(int _iNumFalse, int _idCore) {
        iNumFalse=_iNumFalse;
        idCore=_idCore;
        restart=0;
    }*/
    void setInfo(double _qValue, int _idCore) {
     //iNumFalse=_iNumFalse;
        qValue=_qValue;
        idCore=_idCore;
        restart=0;
    }

};

class Cooperation {
public:
	int nThreads;
	bool End;
	int winnerThread;
	struct mt19937p mt; //global random generation ...> used to create a different random intial seed numbers for each thread.
	UINT32 globalSeed;
	void setEnd(void);
	bool isSolution(void);
	void InitSeed(int);
	std::vector<double>HammingDis;

	Cooperation(int,int,int);
	~Cooperation(void);
	
	int HammingDistance(int,int);
	void printHammingDistanceStats(void);
	void addHammingDistance(void);
	void SetFixedVariables(std::vector<int>);
	
	//sharing..
	//UNIT32 **aVarValue;
	UINT32 **aBestVarValue;
	UINT32 *iBestNumFalse;
	BOOL *isLocalMinimum;
	BOOL *initVars;
	
	SINT32 **bestScore;
	UINT32 **FlipCount;
	UINT32 *MaxFlips;
	BOOL **BestImp;
    
    UINT32 *LastAdaptStep;
    UINT32 *LastAdaptNumFalse;
    int *NumFalseClauses;
    int **ClauseFailure;
    UINT32 **FalseList;
    UINT32 **FalseListPos;
    UINT32 **aVarInfo;
    

	void UpdateStep(bool);
	void PrintBest(int);
	void PrintSol(void);
	void RestartBest(int);
    void InitInfo(int);
    void RestartClause(int);
    
    void iFalseClauses(int);
    
    double tLimit;
    double tStart;
    
    sState sInfo;
    
    MPI_Status status, statusR, statusR2, statusS;
    MPI_Request request, requestR, requestR2, requestS;
    int tFlag, rOK, cOK;
    
    int* bSend;
    int** bRecv;
    int ackrecv;
    int vSend;
    //UINT32 buffer[1001];
    void ReceiveCheck(int);
    void SenderMachine(int);
    void SenderMachine2(int, bool);
    void ReceiveMachine(int);
    void ReceiveMachine2(int);
    void StopThreads(void);
    void AckWinner(int);
    
    
    //algorithms per group
    int aGroup;
    //id in the group
    int idGroup;
    //group number
    int gNumber;
    //rank thread
    int rNumber;
    
    
    void GetObjFun(void);
    void RestartConfirmation(void);
    //restarting algorithms flag
    /*
    int *algIds;*/
    CoreInfo *algIds;
    //quality value
    double qValue;
    double alpha;
    double *qValueVector;
    
    //int compare(const void *, const void *);
    
};

}
#endif
