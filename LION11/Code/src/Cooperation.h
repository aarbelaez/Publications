

#ifndef Cooperation__H
#define Cooperation__H

#include "ubcsat.h"
#include "vector"


class Cooperation {
public:
	int nThreads;
	Ubcsat *Solver;
	bool End;
	int winnerThread;
	struct mt19937p mt; //global random generation ...> used to create different random intial seed numbers for each thread.
	UINT32 globalSeed;
	void setEnd(void);
	bool isSolution(void);
	void InitSeed(int);
	std::vector<double>HammingDis;

	Cooperation(int);
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
	
	void UpdateStep(int);
	void PrintBest(int);
	void PrintSol(void);
	void RestartBest(int);
	
	omp_lock_t read;
	omp_lock_t *write;
};

#endif
