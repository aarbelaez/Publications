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

#include "ubcsat.h"

#ifdef __cplusplus 
namespace ubcsat {
#endif

void PickHWSat();

void AddHWSat() {

  ALGORITHM *pCurAlg;

  pCurAlg = CreateAlgorithm("hwsat","",0,
    "HWSAT: HSAT with random walk",
    "Gent, Walsh [Hybrid Problems... 95]",
    "PickHWSat",
    "DefaultProcedures,Flip+VarScore,VarLastChange,FalseClauseList",
    "default","default");
  
  AddParmProbability(&pCurAlg->parmList,"-wp","walk probability [default %s]","with probability PR, select a random variable from those~that appear in unsat clauses","",&iWp,0.10);

  CreateTrigger("PickHWSat",ChooseCandidate,PickHWSat,"","");

}

void PickHWSat() {
  UINT32 iClause;
  LITTYPE litPick;

  /* with probability (iWp) uniformly choose an unsatisfied clause,
     and then uniformly choose a literal from that clause */

  if (RandomProb(iWp)) {
    if (iNumFalse) {
      iClause = aFalseList[RandomInt(iNumFalse)];
      litPick = pClauseLits[iClause][RandomInt(aClauseLen[iClause])];
      iFlipCandidate = GetVarFromLit(litPick);
    } else {
      iFlipCandidate = 0;
    }
  } else {
    
     /* otherwise, perform a regular HSAT step */

     PickHSat();
  }
}

#ifdef __cplusplus

}
#endif
