

#ifndef UBCSAT_vars_H
#define UBCSAT_vars_H

#include <omp.h>
#include "mt19937p.h"

typedef struct typeUBCSATHEAP {
  char *pHeap;
  char *pFree;
  size_t iBytesFree;
} UBCSATHEAP;


/*void CreateTrigger2(const char *sID,
                   enum EVENTPOINT eEventPoint,
                   FXNPTR pProcedure,
                   char *sDependencyList,
                   char *sDeactivateList);*/

#define CreateTrigger( A, B, C, D, E ) CreateTrigger2(A, B, &Ubcsat::C, D, E );

class Cooperation;

class Ubcsat {
public:
struct mt19937p mt;
//ubcsat-internals variables
UINT32 aActiveCalcColumns[MAXITEMLIST];
FXNPTR aActiveProcedures[NUMEVENTPOINTS][MAXFXNLIST];
ALGORITHM aAlgorithms[MAXNUMALG];
REPORTCOL aColumns[MAXITEMLIST];
DYNAMICPARM aDynamicParms[MAXDYNAMICPARMS];
UINT32 aNumActiveProcedures[NUMEVENTPOINTS];
UINT32 aOutputColumns[MAXITEMLIST];
REPORT aReports[MAXREPORTS];
UINT32 aRTDColumns[MAXITEMLIST];
REPORTSTAT aStats[MAXITEMLIST];
char **aTotalParms;
TRIGGER aTriggers[MAXITEMLIST];
BOOL aParmValid[MAXTOTALPARMS];
BOOL bReportClean;
BOOL bReportFlush;
BOOL bReportEcho;
BOOL bRestart;
BOOL bSolutionFound;
BOOL bSolveMode;
BOOL bTerminateAllRuns;
BOOL bTerminateRun;
BOOL bWeighted;
FLOAT fDummy;
FLOAT fFlipsPerSecond;
FLOAT fBestScore;
FLOAT fTargetW;
SINT32 iBestScore;
UINT32 iCutoff;
UINT32 iFind;
UINT32 iFindUnique;
UINT32 iFlipCandidate;
UINT32 iNumActiveCalcColumns;
UINT32 iNumAlg;
UINT32 iNumDynamicParms;
UINT32 iNumOutputColumns;
UINT32 iNumReports;
UINT32 iNumRTDColumns;
UINT32 iNumRuns;
UINT32 iNumSolutionsFound;
UINT32 iNumStatsActive;
UINT32 iNumTotalParms;
UINT32 iPeriodicRestart;
PROBABILITY iProbRestart;
UINT32 iRun;
UINT32 iRunProceduresLoop;
UINT32 iRunProceduresLoop2;
UINT32 iSeed;
UINT32 iStagnateRestart;
UINT32 iStep;
UINT32 iTarget;
FLOAT fTimeOut;
FLOAT fGlobalTimeOut;
ITEMLIST listColumns;
ITEMLIST listStats;
ITEMLIST listTriggers;
ALGORITHM *pActiveAlgorithm;
ALGPARMLIST parmAlg;
ALGPARMLIST parmHelp;
ALGPARMLIST parmIO;
ALGPARMLIST parmUBCSAT;
char *pVersion;
char *sAlgName;
char *sCommentString;
char *sFilenameIn;
char *sFilenameParms;
char *sFilenameVarInit;
char *sFilenameSoln;
char sMasterString[MAXITEMLISTSTRINGLENGTH];
//const char sNull = 0;
char sParmLine[MAXPARMLINELEN];
char sStringParm[MAXPARMLINELEN];
char *sVarName;

BOOL bReportStateLMOnly;
FLOAT fReportStateQuality;
BOOL bReportBestStepVars;
BOOL bReportOptClausesSol;
UINT32 iReportFalseHistCount;
BOOL bReportDistanceLMOnly;
UINT32 iReportDistHistCount;
BOOL bReportStateQuality;
UINT32 iReportStateQuality;
BOOL bReportTriggersAll;

void AddContainerItem(ITEMLIST *pList,const char *sID, const char *sList);
void AddItem(ITEMLIST *pList,const char *sID);
ALGPARM *AddParmCommon(ALGPARMLIST *pParmList,const char *sSwitch,const char *sTerseDescription,const char *sVerboseDescription,const char *sTriggers);
void AddReportParmCommon(REPORT *pRep, const char *sParmName);
void DeActivateTriggerID(UINT32 iFxnID, const char *sItem);
ALGORITHM *FindAlgorithm(const char *sFindName, const char *sFindVar, BOOL bFindWeighted);
UINT32 MatchParameter(char *sSwitch,char *sParm);
void ParseParameters(ALGPARMLIST *pParmList);
void SetAlgorithmDefaultReports();
void SetDefaultParms(ALGPARMLIST *pParmList);
UINT32 SetColStatFlags (char *sStatsParms);


void SetCurVarState(VARSTATE vsIn);
VARSTATE NewVarState();
VARSTATE NewCopyVarState(VARSTATE vsCopy);
BOOL SetCurVarStateString(VARSTATE vsIn, char *sVarState);
void SetArrayFromVarState(UINT32 *aOut, VARSTATE vsIn);
UINT32 HammingDistVarState(VARSTATE vsA, VARSTATE vsB);
BOOL IsVarStateEqual(VARSTATE vsA, VARSTATE vsB);
BOOL IsVarStateInList(VARSTATELIST *vsList, VARSTATE vsIn);
UINT32 MinHammingVarStateList(VARSTATELIST *vsList, VARSTATE vsIn);
void AddToVarStateList(VARSTATELIST *vsList, VARSTATE vsAdd);
BOOL AddUniqueToVarStateList(VARSTATELIST *vsList, VARSTATE vsAdd);

//ubcsat-globals.h functions 
ALGORITHM *CreateAlgorithm (const char *sName, const char *sVariant, BOOL bWeighted, 
                            const char *sDescription, 
                            const char *sAuthors,
                            const char *sHeuristicTriggers,
                            const char *sDataTriggers,
                            const char *sDefaultOutput,
                            const char *sDefaultStats);


/*
    CopyParameters()      copy the parameters from one algorithm to another
*/

void CopyParameters(ALGORITHM *pDest, const char *sName, const char *sVar, BOOL bWeighted);

/*
    InheritDataTriggers()   copy the data triggers from one algorithm to another
*/

void InheritDataTriggers(ALGORITHM *pDest, const char *sName, const char *sVar, BOOL bWeighted);

/*
    CreateTrigger()       add a new trigger to the UBCSAT system
*/

void CreateTrigger2(const char *sID,
                   enum EVENTPOINT eEventPoint,
                   FXNPTR pProcedure,
                   char *sDependencyList,
                   char *sDeactivateList);

/*
    CreateContainerTrigger()    add a new container trigger to the UBCSAT system
*/

void CreateContainerTrigger(const char *sID, const char *sList);


/*  
    ActivateTriggers()     Explicitly Activate specific trigger(s) [not normally necessary]
*/

void ActivateTriggers(char *sTriggers);

/*  
    DeActivateTriggers()   Explicitly DeActivate specific trigger(s) [not normally necessary]
*/

void DeActivateTriggers(char *sTriggers);


/*
    AddParm????()         adds a parameter to an algorithm (many different types)
*/


void AddParmProbability(ALGPARMLIST *pParmList,
                  const char *sSwitch, 
                  const char *sName, 
                  const char *sDescription,
                  const char *sTriggers,
                  PROBABILITY *pProb,
                  FLOAT fProb);

void AddParmUInt(ALGPARMLIST *pParmList, 
                  const char *sSwitch, 
                  const char *sName, 
                  const char *sDescription,
                  const char *sTriggers,
                  UINT32 *pInt,
                  UINT32 iDefInt);

void AddParmSInt(ALGPARMLIST *pParmList, 
                  const char *sSwitch, 
                  const char *sName, 
                  const char *sDescription,
                  const char *sTriggers,
                  SINT32 *pSInt,
                  SINT32 iDefSInt);

void AddParmBool(ALGPARMLIST *pParmList, 
                  const char *sSwitch, 
                  const char *sName, 
                  const char *sDescription,
                  const char *sTriggers,
                  UINT32 *pBool,
                  BOOL bDefBool);

void AddParmFloat(ALGPARMLIST *pParmList, 
                  const char *sSwitch, 
                  const char *sName, 
                  const char *sDescription,
                  const char *sTriggers,
                  FLOAT *pFloat,
                  FLOAT fDefFloat);

void AddParmString(ALGPARMLIST *pParmList, 
                  const char *sSwitch, 
                  const char *sName, 
                  const char *sDescription,
                  const char *sTriggers,
                  char **pString,
                  char *sDefString);

                  
/*
    CreateReport()        add a new report to the system
*/

REPORT *CreateReport(const char *sID, 
                     const char *sDescription, 
                     const char *sVerboseDescription, 
                     const char *sOutputFile, 
                     const char *sTriggers);

/*
    AddReportParm???()    add a parameter to a report
*/

void AddReportParmUInt(REPORT *pRep, const char *sParmName, UINT32 *pParmValUInt, UINT32 iDefault);
void AddReportParmFloat(REPORT *pRep, const char *sParmName, FLOAT *pParmValFloat, FLOAT fDefault);
void AddReportParmString(REPORT *pRep, const char *sParmName, const char *pDefault);

/*
    AddColumn????()       add a column of data for output & rtd reports
*/

void AddColumnUInt(const char *sID, 
                   const char *sDescription, 
                   char *sHeader1,  
                   char *sHeader2,  
                   char *sHeader3, 
                   char *sPrintFormat, 
                   UINT32 *pCurValue,
                   char *sTriggers,
                   enum COLTYPE eColType);

void AddColumnFloat(const char *sID, 
                    const char *sDescription, 
                    char *sHeader1,  
                    char *sHeader2,  
                    char *sHeader3, 
                    char *sPrintFormat,
                    FLOAT *pCurValue,
                    char *sTriggers,
                    enum COLTYPE eColType);

void AddColumnComposite(const char *sID, 
                        const char *sList);

/*
    AddStatCol()       add a column statistic, providing stats on columns of data 
*/

void AddStatCol(const char *sID, 
             const char *sBaseDescription, 
             const char *sDefParm,
             BOOL bSortByStep);

void AddContainerStat(const char *sID, 
                      const char *sList);

/*
    AddStatCustom()     add a custom statistic, which can be calculated via triggers
*/

void AddStatCustom(const char *sID, 
                   const char *sCustomDescription, 
                   const char *sBaseDescription, 
                   const char *sPrintCustomFormat,
                   void *pCurValue,
                   enum CDATATYPE eCustomType,
                   const char *sDataColumn,
                   const char *sTriggers);


/*
    IsLocalMinimum()      returns TRUE if currently in a local minimum
*/

BOOL IsLocalMinimum(BOOL bUseWeighted);


//ubcsat triggers.c

/***** Trigger ReadCNF *****/

void ReadCNF();

UINT32 iNumVars;
UINT32 iNumClauses;
UINT32 iNumLits;

UINT32 *aClauseLen;
LITTYPE **pClauseLits;

FLOAT *aClauseWeight;
FLOAT fTotalWeight;

UINT32 iVARSTATELen;


/***** Trigger LitOccurence *****/

void CreateLitOccurence();

UINT32 *aNumLitOcc;
UINT32 *aLitOccData;
UINT32 **pLitClause;


/***** Trigger CandidateList *****/

void CreateCandidateList();

UINT32 *aCandidateList;
UINT32 iNumCandidates;
UINT32 iMaxCandidates;


/***** Trigger InitVarsFromFile *****/
/***** Trigger DefaultInitVars *****/
  
void InitVarsFromFile();
void DefaultInitVars();

UINT32 *aVarInit;
UINT32 iInitVarFlip;
BOOL bVarInitGreedy;


/***** Trigger DefaultStateInfo *****/

void CreateDefaultStateInfo();
void InitDefaultStateInfo();

UINT32 *aNumTrueLit;
UINT32 *aVarValue;
UINT32 iNumFalse;
FLOAT fSumFalseW;

//for parallelization by fixing values to some variables
UINT32 *iVarFixValue;


/***** Trigger DefaultFlip[W] *****/

void DefaultFlip();
void DefaultFlipW();


/***** Trigger CheckTermination *****/

void CheckTermination();


/***** Trigger FalseClauseList *****/
/***** Trigger Flip+FalseClauseList[W] *****/

void CreateFalseClauseList();
void InitFalseClauseList();
void UpdateFalseClauseList();
//void InitFalseClauseList();
void FlipFalseClauseList();
void FlipFalseClauseListW();

UINT32 *aFalseList;
UINT32 *aFalseListPos;
UINT32 iNumFalseList;


/***** Trigger VarScore[W] *****/
/***** Trigger Flip+VarScore[W] *****/

void CreateVarScore();
void InitVarScore();
void UpdateVarScore();
void FlipVarScore();
void CreateVarScoreW();
void InitVarScoreW();
void UpdateVarScoreW();
void FlipVarScoreW();

SINT32 *aVarScore;
FLOAT *aVarScoreW;


/***** Trigger MakeBreak[W] *****/
/***** Trigger Flip+MakeBreak[W] *****/

void CreateMakeBreak();
void InitMakeBreak();
void UpdateMakeBreak();
void FlipMakeBreak();
void CreateMakeBreakW();
void InitMakeBreakW();
void UpdateMakeBreakW();
void FlipMakeBreakW();

UINT32 *aBreakCount;
UINT32 *aMakeCount;
UINT32 *aCritSat;
FLOAT *aBreakCountW;
FLOAT *aMakeCountW;


/***** Trigger VarInFalse *****/
/***** Trigger Flip+VarInFalse *****/
/*
    aVarInFalseList[j]    variable # for the jth variable that appears in false clauses
    aVarInFalseListPos[j] for variable[j], position it occurs in aVarInFalseList
    iNumVarsInFalseList   # variables that appear in false clauses 
*/

void CreateVarInFalse();
void InitVarInFalse();
void UpdateVarInFalse();
void FlipVarInFalse();

UINT32 iNumVarsInFalseList;
UINT32 *aVarInFalseList;
UINT32 *aVarInFalseListPos;


/***** Trigger VarLastChange *****/
/*
    aVarLastChange[j]     the step # of the most recent time variable[j] was flipped
    iVarLastChangeReset   the step # of the last time all aVarLastChange values were reset
*/
void CreateVarLastChange();
void InitVarLastChange();
void UpdateVarLastChange();

UINT32 *aVarLastChange;
UINT32 iVarLastChangeReset;


/***** Trigger TrackChanges[W] *****/
/*
    iNumChanges           # of changes to aVarScore[] values this step
    aChangeList[j]        variable # of the jth variable changed this step
    aChangeOldScore[j]    the previous score of variable[j]
    aChangeLastStep[j]    the step of the last change for variable[j]
*/

void CreateTrackChanges();
void InitTrackChanges();
void UpdateTrackChanges();
void FlipTrackChanges();
void FlipTrackChangesFCL();

void CreateTrackChangesW();
void InitTrackChangesW();
void UpdateTrackChangesW();
void FlipTrackChangesW();
void FlipTrackChangesFCLW();

UINT32 iNumChanges;
UINT32 *aChangeList;
SINT32 *aChangeOldScore;
UINT32 *aChangeLastStep;

UINT32 iNumChangesW;
UINT32 *aChangeListW;
FLOAT *aChangeOldScoreW;
UINT32 *aChangeLastStepW;


/***** Trigger DecPromVars[W] *****/

void CreateDecPromVars();
void InitDecPromVars();
void UpdateDecPromVars();

void CreateDecPromVarsW();
void InitDecPromVarsW();
void UpdateDecPromVarsW();

UINT32 *aDecPromVarsList;
UINT32 iNumDecPromVars;

UINT32 *aDecPromVarsListW;
UINT32 iNumDecPromVarsW;


/***** Trigger BestScoreList *****/

void CreateBestScoreList();
void InitBestScoreList();
void UpdateBestScoreList();

UINT32 iNumBestScoreList;
UINT32 *aBestScoreList;
UINT32 *aBestScoreListPos;


/***** Trigger ClausePenaltyFL[W] *****/

void CreateClausePenaltyFL();
void InitClausePenaltyFL();
void InitClausePenaltyFLW();

FLOAT *aClausePenaltyFL;
BOOL bClausePenaltyCreated;
BOOL bClausePenaltyFLOAT;
FLOAT fBasePenaltyFL;
FLOAT fTotalPenaltyFL;



/***** Trigger MakeBreakPenaltyFL *****/

void CreateMakeBreakPenaltyFL();
void InitMakeBreakPenaltyFL();
void UpdateMakeBreakPenaltyFL();
void FlipMBPFLandFCLandVIF();
void FlipMBPFLandFCLandVIFandW();

FLOAT *aMakePenaltyFL;
FLOAT *aBreakPenaltyFL;


/***** Trigger ClausePenaltyINT *****/

void CreateClausePenaltyINT();
void InitClausePenaltyINT();
void InitClausePenaltyINTW();

UINT32 *aClausePenaltyINT;
UINT32 iInitPenaltyINT;
UINT32 iBasePenaltyINT;
UINT32 iTotalPenaltyINT;



/***** Trigger MakeBreakPenalty *****/

void CreateMakeBreakPenaltyINT();
void InitMakeBreakPenaltyINT();
void UpdateMakeBreakPenaltyINT();
void FlipMBPINTandFCLandVIF();
void FlipMBPINTandFCLandVIFandW();

UINT32 *aMakePenaltyINT;
UINT32 *aBreakPenaltyINT;


/***** Trigger NullFlips *****/

UINT32 iNumNullFlips;

void InitNullFlips();
void UpdateNullFlips();


/***** Trigger LocalMins *****/

UINT32 iNumLocalMins;

void InitLocalMins();
void UpdateLocalMins();


/***** Trigger LogDist *****/

void CreateLogDist();

UINT32 *aLogDistValues;
UINT32 iNumLogDistValues;
UINT32 iLogDistStepsPerDecade;


/***** Trigger BestFalse *****/

void InitBestFalse();
void UpdateBestFalse();

UINT32 iBestNumFalse;
UINT32 iBestStepNumFalse;
FLOAT fBestSumFalseW;
UINT32 iBestStepSumFalseW;


/***** Trigger SaveBest *****/

void CreateSaveBest();
void UpdateSaveBest();

VARSTATE vsBest;


/***** Trigger StartFalse *****/

void UpdateStartFalse();

UINT32 iStartNumFalse;
FLOAT fStartSumFalseW;


/***** Trigger CalcImproveMean *****/

void CalcImproveMean();

FLOAT fImproveMean;
FLOAT fImproveMeanW;


/***** Trigger FirstLM *****/

void InitFirstLM();
void UpdateFirstLM();
void CalcFirstLM();

UINT32 iFirstLM;
UINT32 iFirstLMStep;
FLOAT fFirstLMW;
UINT32 iFirstLMStepW;


/***** Trigger FirstLMRatio *****/

void CalcFirstLMRatio();

FLOAT fFirstLMRatio;
FLOAT fFirstLMRatioW;


/***** Trigger TrajBestLM *****/

void UpdateTrajBestLM();
void CalcTrajBestLM();

UINT32 iTrajBestLMCount;
FLOAT fTrajBestLMSum;
FLOAT fTrajBestLMSum2;
UINT32 iTrajBestLMCountW;
FLOAT fTrajBestLMSumW;
FLOAT fTrajBestLMSum2W;

FLOAT fTrajBestLMMean;
FLOAT fTrajBestLMMeanW;
FLOAT fTrajBestLMCV;
FLOAT fTrajBestLMCVW;

/***** Trigger NoImprove *****/

void CheckNoImprove();

UINT32 iNoImprove;


/***** Trigger StartSeed *****/

void StartSeed();

UINT32 iStartSeed;


/***** Trigger CountRandom *****/


/***** Trigger CheckTimeout *****/

void CheckTimeout();


/***** Trigger CheckForRestarts *****/

void CheckForRestarts();


/***** Trigger FlipCounts *****/

void CreateFlipCounts();
void InitFlipCounts();
void UpdateFlipCounts();

UINT32 *aFlipCounts;


/***** Trigger FlipCountStats *****/

void FlipCountStats();

FLOAT fFlipCountsMean;
FLOAT fFlipCountsCV;
FLOAT fFlipCountsStdDev;


/***** Trigger BiasCounts *****/

void CreateBiasCounts();
void PreInitBiasCounts();
void UpdateBiasCounts();
void FinalBiasCounts();

UINT32 *aBiasTrueCounts;
UINT32 *aBiasFalseCounts;


/***** Trigger BiasStats *****/

void BiasStats();

FLOAT fMeanFinalBias;
FLOAT fMeanMaxBias;


/***** Trigger UnsatCounts *****/

void CreateUnsatCounts();
void InitUnsatCounts();
void UpdateUnsatCounts();

UINT32 *aUnsatCounts;


/***** Trigger UnsatCountStats *****/

void UnsatCountStats();

FLOAT fUnsatCountsMean;
FLOAT fUnsatCountsCV;
FLOAT fUnsatCountsStdDev;


/***** Trigger NumFalseCounts *****/

void CreateNumFalseCounts();
void InitNumFalseCounts();
void UpdateNumFalseCounts();

UINT32 *aNumFalseCounts;
UINT32 *aNumFalseCountsWindow;


/***** Trigger DistanceCounts *****/

void CreateDistanceCounts();
void InitDistanceCounts();
void UpdateDistanceCounts();

UINT32 *aDistanceCounts;
UINT32 *aDistanceCountsWindow;


/***** Trigger ClauseLast *****/

void CreateClauseLast();
void InitClauseLast();
void UpdateClauseLast();

UINT32 *aClauseLast;


/***** Trigger SQGrid *****/

void CreateSQGrid();
void InitSQGrid();
void UpdateSQGrid();
void FinishSQGrid();

FLOAT *aSQGridW;
UINT32 *aSQGrid;


/***** Trigger PenaltyStats *****/

void CreatePenaltyStats();
void InitPenaltyStats();
void UpdatePenaltyStatsStep();
void UpdatePenaltyStatsRun();

FLOAT *aPenaltyStatsMean;
FLOAT *aPenaltyStatsStddev;
FLOAT *aPenaltyStatsCV;

FLOAT *aPenaltyStatsSum;
FLOAT *aPenaltyStatsSum2;

FLOAT *aPenaltyStatsMeanSum;
FLOAT *aPenaltyStatsMeanSum2;
FLOAT *aPenaltyStatsStddevSum;
FLOAT *aPenaltyStatsStddevSum2;
FLOAT *aPenaltyStatsCVSum;
FLOAT *aPenaltyStatsCVSum2;


/***** Trigger VarFlipHistory *****/

void CreateVarFlipHistory();
void UpdateVarFlipHistory();

UINT32 *aVarFlipHistory;
UINT32 iVarFlipHistoryLen;


/***** Trigger MobilityWindow *****/

void CreateMobilityWindow();
void InitMobilityWindow();
void UpdateMobilityWindow();

UINT32 *aMobilityWindowVarChange;
UINT32 *aMobilityWindow;
FLOAT *aMobilityWindowSum;
FLOAT *aMobilityWindowSum2;


/***** Trigger MobilityFixedFrequencies *****/

void CreateMobilityFixedFrequencies();
void InitMobilityFixedFrequencies();
void UpdateMobilityFixedFrequencies();

UINT32 *aMobilityFixedFrequencies;


/***** Trigger AutoCorr *****/

void CreateAutoCorr();
void InitAutoCorr();
void UpdateAutoCorr();
void CalcAutoCorr();

UINT32 iAutoCorrMaxLen;
FLOAT fAutoCorrCutoff;
UINT32 iAutoCorrLen;
FLOAT *aAutoCorrValues;
FLOAT *aAutoCorrStartBuffer;
FLOAT *aAutoCorrEndCircBuffer;
FLOAT fAutoCorrSum;
FLOAT fAutoCorrSum2;
FLOAT *aAutoCorrCrossSum;

/***** Trigger AutoCorrOne *****/

void InitAutoCorrOne();
void UpdateAutoCorrOne();
void CalcAutoCorrOne();

FLOAT fAutoCorrOneVal;
FLOAT fAutoCorrOneEst;
FLOAT fAutoCorrOneStart;
FLOAT fAutoCorrOneLast;
FLOAT fAutoCorrOneSum;
FLOAT fAutoCorrOneSum2;
FLOAT fAutoCorrOneCrossSum;


/***** Trigger BranchFactor *****/

void BranchFactor();
void BranchFactorW();

FLOAT fBranchFactor;
FLOAT fBranchFactorW;


/****** Trigger StepsUpDownSide *****/

void InitStepsUpDownSide();
void UpdateStepsUpDownSide();

UINT32 iNumUpSteps;
UINT32 iNumDownSteps;
UINT32 iNumSideSteps;
UINT32 iNumUpStepsW;
UINT32 iNumDownStepsW;
UINT32 iNumSideStepsW;

/****** Trigger NumRestarts *****/

void NumRestarts();

UINT32 iNumRestarts;

/***** Trigger LoadKnownSolutions *****/

void LoadKnownSolutions();

VARSTATELIST vslKnownSoln;
BOOL bKnownSolutions;


/***** Trigger SolutionDistance *****/

void CreateSolutionDistance();
void UpdateSolutionDistance();

VARSTATE vsSolutionDistance;
UINT32 iSolutionDistance;


/***** Trigger FDCRun *****/

void CreateFDCRun();
void InitFDCRun();
void UpdateFDCRun();
void CalcFDCRun();

FLOAT fFDCRun;

FLOAT fFDCRunHeightDistanceSum;
FLOAT fFDCRunHeightSum;
FLOAT fFDCRunHeightSum2;
FLOAT fFDCRunDistanceSum;
FLOAT fFDCRunDistanceSum2;
UINT32 iFDCRunCount;



/***** Trigger DynamicParms *****/

void DynamicParms();



/***** Trigger FlushBuffers *****/

void FlushBuffers();



/***** Trigger CheckWeighted *****/

void CheckWeighted();



/***** Trigger UniqueSolutions *****/

void CreateUniqueSolutions();
void UpdateUniqueSolutions();

VARSTATELIST vslUnique;
VARSTATE vsCheckUnique;
UINT32 iNumUniqueSolutions;
UINT32 iLastUnique;


/***** Trigger VarsShareClauses *****/

UINT32 *aNumVarsShareClause;
UINT32 *aVarsShareClauseData;
UINT32 **pVarsShareClause;
UINT32 iNumShareClauses;

void CreateVarsShareClauses();

//ubcsat-internal.c  functions
void ActivateDynamicParms();
void ActivateAlgorithmTriggers();
void ActivateColumnID(UINT32 iColID, const char *sItem);
void ActivateStatID(UINT32 iStatID, const char *sItem);
void ActivateTriggerID(UINT32 iFxnID, const char *sItem);

void ActivateColumns(char *sColumns);
void ActivateStats(char *sStats);

void AddAllocateRAMColumnID(UINT32 j, const char *sItem);
void AddOutputColumnID(UINT32 j, const char *sItem);
void AddParameters();
void AddParmReport(ALGPARMLIST *pParmList,const char *sSwitch,const char *sName,const char *sDescription,const char *sTriggers);
void AddReportTriggers();
void AddRTDColumnID(UINT32 j, const char *sItem);
void CalculateStats(FLOAT *fMean, FLOAT *fStddev, FLOAT *fCV, FLOAT fSum, FLOAT fSum2, UINT32 iCount);
FLOAT CorrelationCoeff(FLOAT fSumA, FLOAT fSumA2,FLOAT fSumB, FLOAT fSumB2, FLOAT fSumAB, UINT32 iCount);
void CheckPrintHelp();
UINT32 FindItem(ITEMLIST *pList,char *sID);
void HelpBadParm(char *sParm);
void ParseAllParameters(int argc, char *argv[]);
void PrintAlgParmSettings(REPORT *pRep, ALGPARMLIST *pParmList);
void ParseItemList(ITEMLIST *pList, char *sItems, CALLBACKPTR ItemFunction);
void PrintUBCSATHeader(REPORT *pRep);
void SetupUBCSAT();



//reports.c varibles
REPORT *pRepHelp;
REPORT *pRepErr;

REPORT *pRepOut;
REPORT *pRepRTD;
REPORT *pRepStats;
REPORT *pRepState;
REPORT *pRepModel;
REPORT *pRepSolution;
REPORT *pRepUniqueSol;
REPORT *pRepBestSol;
REPORT *pRepBestStep;
REPORT *pRepTrajBestLM;
REPORT *pRepOptClauses;
REPORT *pRepFalseHist;
REPORT *pRepDistance;
REPORT *pRepDistHist;
REPORT *pRepCNFStats;
REPORT *pRepFlipCounts;
REPORT *pRepBiasCounts;
REPORT *pRepUnsatCounts;
REPORT *pRepVarLast;
REPORT *pRepClauseLast;
REPORT *pRepSQGrid;
REPORT *pRepPenalty;
REPORT *pRepPenMean;
REPORT *pRepPenStddev;
REPORT *pRepPenCV;
REPORT *pRepMobility;
REPORT *pRepMobFixed;
REPORT *pRepMobFixedFreq;
REPORT *pRepAutoCorr;
REPORT *pRepTriggers;
REPORT *pRepSATComp;

void AddReports();

//ubcsat-reports.c definitions
/*  
    This file contains the code to make the various reports work
*/

/***** Trigger ReportOut *****/

BOOL bReportOutputSuppress;

void ReportOutSetup();
void ReportOutSplash();
void ReportOutHeader();
void ReportOutRow();
void ReportOutSuppressed();

/***** Trigger ReportStats *****/
void ReportStatsSetup();
void ReportStatsPrint();

/***** Trigger ReportRTD *****/
void ReportRTDSetup();
void ReportRTDPrint();

/***** Trigger ReportModelPrint *****/
void ReportModelPrint();

/***** Trigger ReportCNFStatsPrint *****/
void ReportCNFStatsPrint();

/***** Trigger ReportStatePrint *****/
void ReportStatePrint();

/***** Trigger ReportSolutionPrint *****/
void ReportSolutionPrint();

/***** Trigger ReportUniqueSolPrint *****/
void ReportUniqueSolPrint();

/***** Trigger ReportBestSolPrint *****/
void ReportBestSolPrint();

/***** Trigger ReportBestStepPrint *****/
void ReportBestStepPrint();

/***** Trigger ReportTrajBestLM *****/
void ReportTrajBestLMPostStep();
void ReportTrajBestLMPostRun();

/***** Trigger ReportUnsatClausesPrint *****/
void ReportUnsatClausesPrint();

/***** Trigger ReportFalseHistPrint *****/
void ReportFalseHistPrint();

/***** Trigger ReportDistHistPrint *****/
void ReportDistancePrint();

/***** Trigger ReportDistHistPrint *****/
void ReportDistHistPrint();

/***** Trigger ReportFlipCountsPrint *****/
void ReportFlipCountsPrint();

/***** Trigger ReportBiasCountsPrint *****/
void ReportBiasCountsPrint();

/***** Trigger ReportUnsatCountsPrint *****/
void ReportUnsatCountsPrint();

/***** Trigger ReportVarLastPrint *****/
void ReportVarLastPrint();

/***** Trigger ReportClauseLastPrint *****/
void ReportClauseLastPrint();

/***** Trigger ReportSQGridPrint *****/
void ReportSQGridPrint();

/***** Trigger ReportPenaltyPrint *****/
void ReportPenaltyCreate();
void ReportPenaltyPrintStep();
void ReportPenaltyPrintRun();
BOOL bReportPenaltyReNormBase;
BOOL bReportPenaltyReNormFraction;
UINT32 bReportPenaltyEveryLM;
FLOAT *aPenaltyStatsFinal;
FLOAT *aPenaltyStatsFinalSum;
FLOAT *aPenaltyStatsFinalSum2;

/***** Trigger ReportPenMeanPrint *****/
void ReportPenMeanPrint();

/***** Trigger ReportPenStddevPrint *****/
void ReportPenStddevPrint();

/***** Trigger ReportPenCVPrint *****/
void ReportPenCVPrint();

/***** Trigger ReportMobilityPrint *****/
void ReportMobilityPrint();
UINT32 iReportMobilityDisplay;
BOOL bReportMobilityNormalized;

/***** Trigger ReportMobFixedPrint *****/
void ReportMobFixedPrint();
UINT32 iMobFixedWindow;
BOOL bMobilityFixedIncludeStart;

/***** Trigger ReportMobFixedFreqPrint *****/
void ReportMobFixedFreqPrint();

/***** Trigger ReportAutoCorrPrint *****/
void ReportAutoCorrPrint();

/***** Trigger ReportTriggersPrint *****/
void ReportTriggersPrint();

/***** Trigger ReportSatCompetitionPrint *****/
void ReportSatCompetitionPrint();

/***** Trigger ActivateStepsFoundColumns *****/
void ActivateStepsFoundColumns();

/***** Trigger AllocateColumnRAM *****/
void AllocateColumnRAM();

/***** Trigger CalcPercentSolve *****/
void CalcPercentSolve();
FLOAT fPercentSuccess;

/***** Trigger ColumnRunCalculation *****/
void ColumnRunCalculation();

/***** Trigger ColumnStepCalculation *****/
void ColumnStepCalculation();

/***** Trigger ColumnInit *****/
void ColumnInit();

/***** Trigger InitSolveModev *****/
void InitSolveMode();

/***** Trigger UpdatePercents *****/
void UpdatePercents();

/***** Trigger MobilityColumn *****/
void InitMobilityColumnN();
void InitMobilityColumnX();
void UpdateMobilityColumn();
BOOL bMobilityColNActive;
BOOL bMobilityColXActive;
FLOAT fMobilityColNMean;
FLOAT fMobilityColXMean;
FLOAT fMobilityColNMeanNorm;
FLOAT fMobilityColXMeanNorm;
FLOAT fMobilityColNCV;
FLOAT fMobilityColXCV;

/***** Trigger UpdateTimes *****/
void UpdateTimes();

/***** Trigger SortByStepPerformance *****/
UINT32 *aSortedBySteps;
BOOL bSortedByStepsValid;
UINT32 *aSortedByCurrent;
UINT32 *aSortedByStepsAndFound;
void SortByCurrentColData(REPORTCOL *pCol);
void SortByCurrentColDataAndFound(REPORTCOL *pCol);
void SortByStepPerformance();

/***** Trigger CalcFPS *****/
void CalcFPS();

/***** Trigger StringAlgParms *****/
char *sStringAlgParms;
void StringAlgParms();



//ubcsat-help.c
BOOL bShowHelp;
BOOL bShowHelpA;
BOOL bShowHelpW;
BOOL bShowHelpP;
BOOL bShowHelpV;
BOOL bShowHelpT;
BOOL bShowHelpR;
BOOL bShowHelpC;
BOOL bShowHelpS;

void HelpNoAlgorithm();
void HelpBadReport(char *sParm);
void HelpPrintAlgorithms();
void HelpPrintAlgorithmsW();
void HelpPrintParms();
void HelpShowBasic();
void HelpShowVerbose();
void HelpShowTerse();
void HelpPrintReports();
void HelpPrintColumns();
void HelpPrintStats();
void HelpPrintParameter(ALGPARM *pCurParm, BOOL bAlgOffset);
void HelpPrintSpecialParameters();
void HelpPrintAlgorithm(ALGORITHM *pAlg, BOOL bTerse, BOOL bVerbose);
void HelpPrintReport(REPORT *pRep);
//void CheckPrintHelp();
BOOL bHelpHeaderShown;

void HelpShowHeader();
char sHelpString[HELPSTRINGLENGTH];
void SetHelpStringAlg(ALGORITHM *pAlg);
void SetHelpStringRep(REPORT*);
void HelpPrintParameters(ALGPARMLIST*);
void HelpPrintParametersTerse(ALGPARMLIST*);
void HelpPrintParametersTerse2(ALGPARMLIST*);
void HelpPrintAlgParameters(ALGORITHM*);
void HelpPrintAlgParametersTerse(ALGORITHM*);
void HelpPrintSpecialParametersTerse();

//ubcsat-internal.c  (local definitions)
void AddDynamicParm(void *pTarget, enum CDATATYPE eDataType, UINT32 *pBase, FLOAT fFactor);
void CheckInvalidParamters();
void CheckParamterFile(int iCommandLineCount,char **aCommandLineArgs);
void ClearActiveProcedures();
ALGPARM *FindParm(ALGPARMLIST *pParmList, char *sSwitch);
ALGORITHM *GetAlgorithm();
void SetupUbcsat();


//ubcsat-triggers.c (local definitions)
void AddDataTriggers();
void FlipVarScoreFalseClauseList();

char sLine[MAXCNFLINELEN];

//ubcsat-io.h

void RandomSeed(UINT32 iSeed);
UINT32 RandomMax();
UINT32 RandomInt(UINT32 iMax);
BOOL RandomProb(PROBABILITY);
FLOAT RandomFloat();

FLOAT ProbToFloat(PROBABILITY iProb);
PROBABILITY FloatToProb(FLOAT fProb);
UINT32 ProbToInvInt(PROBABILITY iProb);

void ActivateReportTriggers();
void AbnormalExit();
void CleanExit();

void CloseSingleFile(FILE *filToClose);

void SetupFile(FILE **fFil,const char *sOpenType, const char *sFilename, FILE *filDefault, BOOL bAllowNull);

UINT32 iNumRandomCalls;
void SetupCountRandom();
void InitCountRandom();

//void InitVarsFromFile();

FILE *filReportPrint;


char *sFilenameRandomData;
char *sFilenameAbort;
void CreateFileRandom();
void CloseFileRandom();
void FileAbort();

//UINT32 iNumRandomCalls;
FXNRAND32 fxnRandOrig;

UINT32 CountRandom();
void CloseReports();
UINT32 FileRandomUInt32();

UINT32 Ubcsat_genrand_int32();

FXNRAND32 fxnRandUInt32;


//variables ubcsat-reports
void PrintColHeaders(REPORT *pRep,UINT32 iNumCols, UINT32 *aCols);
void PrintRow(REPORT *pRep, UINT32 iRow, UINT32 iNumCols, UINT32 *aCols);
FLOAT GetRowElement(REPORTCOL *pCol,UINT32 iRowRequested, BOOL bSorted, BOOL bSortByStep);
void PrintRTDRow(UINT32);
int CompareSortedUInt(const void*, const void*);
int CompareSortedSInt(const void*, const void*);
int CompareSortedFloat(const void*, const void*);
REPORTCOL *pSortCol;
UINT32 *aFoundData;
UINT32 *auiSortColData;
SINT32 *asiSortColData;
FLOAT *afSortColData;

int CompareFoundSortedUInt(const void *a, const void *b);
int CompareFoundSortedSInt(const void*, const void*);
int CompareFoundSortedFloat(const void*, const void*);


//ubcsat-mem.c
UINT32 iNumHeap;
size_t iLastRequestSize;
UINT32 iLastHeap;

UBCSATHEAP aHeap[MAXHEAPS];

void* AllocateRAM(size_t);
void AdjustLastRAM(size_t);
void SetString(char**, const char*);
void FreeRAM();


//algorithms.h
void AddAlgorithms();

/* gsat.c */

void AddGSat();

/* gwsat.c */

void AddGWSat();
PROBABILITY iWp;

/* gsat-tabu.c */

void AddGSatTabu();
UINT32 iTabuTenure;

/* hsat.c */

void AddHSat();
void PickHSat();
void PickHSatW();

/* hwsat.c */

void AddHWSat();

/* walksat.c */

void AddWalkSat();
UINT32 PickClauseWCS();

/* walksat-tabu.c */

void AddWalkSatTabu();
void PickWalkSatTabu();
UINT32 iWalkSATTabuClause;

/* novelty.c */

void AddNovelty();
void AddNoveltyPlus();
void AddNoveltyPlusPlus();
void PickNoveltyPlusW();
void PickNoveltyPlusPlus();
void PickNoveltyPlusPlusW();

void PickNoveltyVarScore();
void PickNoveltyPlusVarScore();
void PickNoveltyPlusPlusVarScore();

PROBABILITY iNovNoise;
PROBABILITY iDp;

/* novelty+p.c */

void AddNoveltyPlusP();
void PickNoveltyPlusP();

/* rnovelty.c */

void AddRNovelty();
void AddRNoveltyPlus();

/* adaptnovelty.c */

void AddAdaptNoveltyPlus();
void InitAdaptNoveltyNoise();
void AdaptNoveltyNoiseAdjust();
UINT32 iLastAdaptStep;
UINT32 iLastAdaptNumFalse;
FLOAT fLastAdaptSumFalseW;
UINT32 iInvPhi;
UINT32 iInvTheta;
FLOAT fAdaptPhi;
FLOAT fAdaptTheta;

/* saps.c */

void AddSAPS();

/* paws.c */

void AddPAWS();
PROBABILITY iPAWSFlatMove;

/* ddfw.c */

void AddDDFW();

/* g2wsat.c */

void AddG2WSat();

/* vw.c */

void AddVW();

/* rots.c */

void AddRoTS();
void PickRoTS();

UINT32 iTabuTenureInterval;
UINT32 iTabuTenureLow;
UINT32 iTabuTenureHigh;

/* irots.c */

void AddIRoTS();

/* samd.c */

void AddSAMD();

/* random.c */

void AddRandom();

/* derandomized.c */

void AddDerandomized();

/* rgsat.c */

void AddRGSat();


//random.c
void PickURWalk();
void PickCRWalk();
void SchoeningRestart();


//adaptnovelty.c
//UINT32 iInvPhi=5;               /* = 1/phi   */
//UINT32 iInvTheta=6;             /* = 1/theta */

/*FLOAT fAdaptPhi;
FLOAT fAdaptTheta;

UINT32 iLastAdaptStep;
UINT32 iLastAdaptNumFalse;
FLOAT fLastAdaptSumFalseW;
*/

//void InitAdaptNoveltyNoise();
void AdaptNoveltyNoise();
//void AdaptNoveltyNoiseAdjust();
void AdaptNoveltyNoiseW();



//derandomized.c
void PickDCRWalk();
void CreateClausePickCount();
void InitClausePickCount();
void UpdateClausePickCount();
void CreateNextClauseLit();
void InitNextClauseLit();
void UpdateNextClauseLit();

UINT32 iClausePick;
UINT32 *aClausePickCount;  
UINT32 *aNextClauseLit;


void PickDANOVP();
void InitAdaptNoveltyNoiseDet();
void AdaptNoveltyNoiseDet();

UINT32 iCountNovNoise0;
UINT32 iCountNovNoise1;

//ddfw.c
void SetupDDFW();
void DistributeDDFW();

UINT32 iDDFWInitWeight;
PROBABILITY iDDFW_TL;

//g2wsat.c
void PickG2WSat();
void PickG2WSatW();

void PickG2WSatNoveltyPlusOldest();
void PickG2WSatNoveltyPlusOldestW();

void PickG2WSatP();

void InitAdaptG2WSatNoise();
void AdaptG2WSatNoise();

//gsat.c
void PickGSatSimple();
void PickGSatWithBSL();
void PickGSatW();

//gsat-tabu.c
//UINT32 iTabuTenure;

void PickGSatTabu();
void PickGSatTabuW();

//hsat.c
//void PickHSat();
//void PickHSatW();

//hwsat.c
void PickHWSat();
void PickHWSatW();

//irots.c
UINT32 iIrotsEscapeSteps;
UINT32 iIrotsPerturbSteps;

UINT32 iLSTabuTenure;
UINT32 iPerturbTabuTenure;

PROBABILITY iIrotsNoise;

UINT32 iLSTabuTenureLow;
UINT32 iLSTabuTenureHigh;
UINT32 iPerturbTabuTenureLow;
UINT32 iPerturbTabuTenureHigh;

UINT32 iIrotsLSBestStep;
UINT32 iIrotsLSBestValue;
FLOAT fIrotsLSBestValueW;

UINT32 iIrotsSavedValue;
FLOAT fIrotsSavedValueW;

UINT32 iIrotsMode;

UINT32 *aIrotsBackup;

void InitIRoTSParms();
void InitIRoTS();
void PostStepIRoTS();

void CreateIRoTSBackup();
void IRoTSBackup();
void IRoTSRestore();


//mylocal.c
void AddWalkSatTabuNoNull();
void PickWalkSatTabuNoNull();
void AddAgeStat();
void UpdateCurVarAge();

UINT32 iCurVarAge;                                                /* variable to store current variable age */
void AddLocal();

//novelty.c
//PROBABILITY iNovNoise;
//PROBABILITY iDp;

void PickNovelty();
void PickNoveltyPlus();
void PickNoveltyW();
//void PickNoveltyPlusW();

//void PickNoveltyPlusPlus();
//void PickNoveltyPlusPlusW();

//void PickNoveltyVarScore();
//void PickNoveltyPlusVarScore();
//void PickNoveltyPlusPlusVarScore();

//novelty+p
//void PickNoveltyPlusP();
void InitLookAhead();
void CreateLookAhead();
SINT32 BestLookAheadScore(UINT32 iLookVar);

UINT32 *aIsLookAhead;
UINT32 *aLookAheadList;
SINT32 *aLookAheadScoreChange;
void UCreateLookAhead();

//paws.c
void PickPAWS();
void PostFlipPAWS();

UINT32 iPAWSMaxInc;
//PROBABILITY iPAWSFlatMove;

UINT32 iPawsSmoothCounter;


/***** Trigger PenClauseList *****/

void CreatePenClauseList();
void InitPenClauseList();

UINT32 *aPenClauseList;
UINT32 *aPenClauseListPos;
UINT32 iNumPenClauseList;

void SmoothPAWS();
void ScalePAWS();


//rgsat.c
void PickRGSat();
void PickRGSatW();

//rnovelty.c
void PickRNovelty();
void PickRNoveltyPlus();

void PickRNoveltyCore();

//rots.c
void InitRoTS();
//void PickRoTS();
void PickRoTSW();


//samd.c
void SAMDUpdateVarLastChange();
void SAMDUpdateVarLastChangeW();

//saps.c
FLOAT fAlpha;
FLOAT fRho;
FLOAT fPenaltyImprove;
PROBABILITY iPs;
PROBABILITY iRPs;

//const FLOAT fMaxClausePenalty = 1000.0f;

void PickSAPS();
void PostFlipSAPS();

void PostFlipSAPSWSmooth();

void InitRSAPS();
void PickRSAPS();
void PostFlipRSAPS();

void PickSAPSNR();
void PostFlipSAPSNR();

void SmoothSAPS();
void AdjustPenalties();
void ScaleSAPS();
void SmoothSAPSWSmooth();
void RSAPSAdaptSmooth();

//vw.c
FLOAT fVW2Smooth;
FLOAT fVW2WeightFactor;
void PickVW1();
void PickVW2();

/***** Trigger VW2Weights *****/

void CreateVW2Weights();
void InitVW2Weights();
void UpdateVW2Weights();

FLOAT *aVW2Weights;
FLOAT fVW2WeightMean;

//walksat.c
void PickWalkSatSKC();
void PickWalkSatSKCW();

//walksat-tabu.c
//void PickWalkSatTabu();
void PickWalkSatTabuW();
//void PickWalkSatTabuNoNull();

//UINT32 iWalkSATTabuClause;

//gwsat.cc
//PROBABILITY iWp;

void PickGWSat();
void PickGWSatW();


//thread info
int idThread;
Ubcsat(void) {
	iInvPhi=5;
	iInvTheta=6;
	fxnRandUInt32 = &Ubcsat::Ubcsat_genrand_int32;
	idThread=-1;
	iVarFixValue=NULL;
}

omp_lock_t *write;
omp_lock_t *read;
bool Solve(int argc, char *argv[], Cooperation *coop);



void sVars(int vars[],int m) {
#pragma critical (tmp)
{
  //parallelization.
	if(iVarFixValue==NULL) {
		iVarFixValue = (UINT32*)AllocateRAM((iNumVars+1)*sizeof(UINT32));
	}

	int tvars=m,n=idThread;
	for(int i=0;i<=iNumVars;i++) {
		iVarFixValue[i]=3;
	}
	while(true) {
		tvars--;
		if(n<2) {
			iVarFixValue[vars[tvars]]=n;
			std::cout<<n<<" ";
			break;
		}
		
		int tmp=n%2;
		std::cout<<tmp<<" ";
		n=n/2;
	}
	while(tvars>0) {
		tvars--;
		iVarFixValue[vars[tvars]]=0;
		std::cout<<"0 ";
	}
	std::cout<<std::endl;
}

}
};


#endif
