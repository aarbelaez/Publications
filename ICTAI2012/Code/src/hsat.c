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

void PickHSat();

void AddHSat() {

  ALGORITHM *pCurAlg;

  pCurAlg = CreateAlgorithm("hsat","",0,
    "HSAT",
    "Gent, Walsh [AAAI 93]",
    "PickHSat",
    "DefaultProcedures,Flip+VarScore,VarLastChange",
    "default","default");
  
  CreateTrigger("PickHSat",ChooseCandidate,PickHSat,"","");

}

void PickHSat() {
  
  UINT32 j;
  SINT32 iScore;

  iNumCandidates = 0;
  iBestScore = (SINT32) iNumClauses;

  /* check score of all variables */

  for (j=1;j<=iNumVars;j++) {

    /* use cached value of score */

    iScore = aVarScore[j];

    /* build candidate list of best vars */

    if (iScore < iBestScore) {
        iNumCandidates=0;
        aCandidateList[iNumCandidates++] = j;
        iBestScore = iScore;
    } else {
      if ((iScore == iBestScore) && (aVarLastChange[j] <= aVarLastChange[*aCandidateList])) {
        /* if 2 variables are tied, and one is 'older' then choose older var */
        if (aVarLastChange[j] < aVarLastChange[*aCandidateList]) {
          iNumCandidates=0;
        }
        aCandidateList[iNumCandidates++] = j;
      }
    }
  }
  
  /* select flip candidate uniformly from candidate list */
  
  if (iNumCandidates > 1) {
    iFlipCandidate = aCandidateList[RandomInt(iNumCandidates)];
  } else {
    iFlipCandidate = aCandidateList[0];
  }
}

#ifdef __cplusplus

}
#endif
