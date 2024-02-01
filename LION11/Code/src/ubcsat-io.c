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

/*  
    This file contains some of the i/o routines for
    file access and random number generators
*/

#define MERSENNE

#ifdef MERSENNE
  /*extern unsigned long genrand_int32();
  extern void init_genrand(unsigned long s);*/

  UINT32 Ubcsat::Ubcsat_genrand_int32() { return static_cast<UINT32>(genrand(&mt)); }
  //unsigned long Ubcsat::Ubcsat_genrand_int32() { return genrand_int32(); }
  //FXNRAND32 fxnRandUInt32 = &Ubcsat::Ubcsat_genrand_int32;
  //FXNRAND32 fxnRandUInt32 = genrand_int32;
  //#define fxnRandSeed(A) init_genrand(A)
  #define fxnRandSeed(A) sgenrand(A,&mt)
#else
  #ifdef WIN32
    FXNRAND32 fxnRandUInt32 = (UINT32 (*)()) rand;
    #define fxnRandSeed(A) srand(A)
  #else
    FXNRAND32 fxnRandUInt32 = (UINT32 (*)()) random;
    #define fxnRandSeed(A) srandom(A)
  #endif
#endif

FLOAT Ubcsat::ProbToFloat(PROBABILITY iProb) {
  return (iProb*(1.0/4294967295.0));
}

PROBABILITY Ubcsat::FloatToProb(FLOAT fProb) {
  return ((PROBABILITY) (fProb * 4294967295.0));
}

UINT32 Ubcsat::ProbToInvInt(PROBABILITY iProb) {
  FLOAT fInv;
  fInv = 4294967295.0 / iProb;
  return ((UINT32) (fInv + 0.5));
}

FLOAT Ubcsat::RandomFloat() {
  UINT32 iNum;
  FLOAT fNum;
  iNum = (*this.*fxnRandUInt32)();
  fNum = (FLOAT) iNum;
  fNum /= 4294967295.0;
  return(fNum);
}

UINT32 Ubcsat::RandomInt(UINT32 iMax) {
  return((*this.*fxnRandUInt32)() % iMax);
}

BOOL Ubcsat::RandomProb(PROBABILITY iProb) {
  if (iProb==0) {
    return(FALSE);
  }
  if ((*this.*fxnRandUInt32)() <= iProb) {
    return(TRUE);
  } else {
    return(FALSE);
  }
}

UINT32 Ubcsat::RandomMax() {
  return((*this.*fxnRandUInt32)());
}

void Ubcsat::RandomSeed(UINT32 iSeed) {
  fxnRandSeed(iSeed);
}
/*
UINT32 iNumRandomCalls;
FXNRAND32 fxnRandOrig;
*/
UINT32 Ubcsat::CountRandom() {
  iNumRandomCalls++;
  return((*this.*fxnRandOrig)());
}

void Ubcsat::SetupCountRandom() {
  fxnRandOrig = fxnRandUInt32;
  fxnRandUInt32 = &Ubcsat::CountRandom;
}

void Ubcsat::InitCountRandom() {
  iNumRandomCalls = 0;
}

void Ubcsat::SetupFile(FILE **fFil,const char *sOpenType, const char *sFilename, FILE *filDefault, BOOL bAllowNull) {

  if (*sFilename) {
    if (strcmp(sFilename,"null")==0) {
      if (bAllowNull) {
        (*fFil) = NULL;  
      } else {
        fprintf(stderr,"Fatal Error: Invalid use of null i/o\n");
        AbnormalExit();
      }
    } else {
      (*fFil) = fopen(sFilename,sOpenType);
      if ((*fFil) == NULL) {
        printf("Fatal Error: Invalid filename [%s] specified \n",sFilename);
        AbnormalExit();
      }
    }
  } else {
    (*fFil) = filDefault;
  }
}

void Ubcsat::ActivateReportTriggers() {

  UINT32 j;

  for (j=0;j<iNumReports;j++) {
    if (strcmp(aReports[j].sOutputFile,"null")==0) {
      aReports[j].bActive = FALSE;
      aReports[j].fileOut = 0;
    }
    if (aReports[j].bActive) {
      if (strcmp(aReports[j].sOutputFile,"stdout")==0) {
        aReports[j].fileOut = stdout;
      } else if (strcmp(aReports[j].sOutputFile,"stderr")==0) {
        aReports[j].fileOut = stderr;
      } else {
        if (aReports[j].bSpecialFileIO == FALSE) {
          aReports[j].fileOut = fopen(aReports[j].sOutputFile,"w");
          if (aReports[j].fileOut == 0) {
            printf("Fatal Error: Invalid filename [%s] specified \n",aReports[j].sOutputFile);
            AbnormalExit();
          }
        }
      }
      ActivateTriggers(aReports[j].sTriggers);
    }
  }

}

void Ubcsat::CloseSingleFile(FILE *filToClose) {
  if ((filToClose)&&(filToClose != stdout)&&(filToClose != stderr)) {
    fclose(filToClose);
  }
}

void Ubcsat::CloseReports() {
  UINT32 j;
  for (j=0;j<iNumReports;j++) {
    if (aReports[j].bActive) {
      if (aReports[j].bSpecialFileIO == FALSE) {
        CloseSingleFile(aReports[j].fileOut);
      }
    }
  }
}

void Ubcsat::AbnormalExit() {
  CloseReports();  
  FreeRAM();
  exit(1);
}

void Ubcsat::CleanExit() {
  CloseReports();
  FreeRAM();
}

FILE *filReportPrint;


FILE *filRandomData;
char *sFilenameRandomData;
char *sFilenameAbort;
BYTE *pRandomDataBuffer;
UINT32 iRandomBufferRemaining;
BYTE *pNextRandomData;
BOOL bCycleData;
UINT32 iCycleDataLen;

UINT32 Ubcsat::FileRandomUInt32() {
  UINT32 iReturn = 0;
  UINT32 j;
  if (iRandomBufferRemaining >= 4) {
    iReturn = *pNextRandomData++;
    iReturn = (iReturn << 8) | *pNextRandomData++;
    iReturn = (iReturn << 8) | *pNextRandomData++;
    iReturn = (iReturn << 8) | *pNextRandomData++;
    iRandomBufferRemaining -= 4;
  } else {
    j = 4;
    while (j--) {
      if (iRandomBufferRemaining==0) {
        if (bCycleData) {
          iRandomBufferRemaining = iCycleDataLen;
          pNextRandomData = pRandomDataBuffer;
        } else {
          iRandomBufferRemaining = fread(pRandomDataBuffer,1,RANDOMFILEBUFFERSIZE,filRandomData);
          if (iRandomBufferRemaining==0) {
            CloseSingleFile(filRandomData);
            SetupFile(&filRandomData,"rb",sFilenameRandomData,NULL,FALSE);
            iRandomBufferRemaining = fread(pRandomDataBuffer,1,RANDOMFILEBUFFERSIZE,filRandomData);
          }
          pNextRandomData = pRandomDataBuffer;
        }
      }
      iReturn = (iReturn << 8) | *pNextRandomData++;
      iRandomBufferRemaining--;
    }
  }
  return(iReturn);
}

void Ubcsat::CreateFileRandom() {

  pRandomDataBuffer = (BYTE*)AllocateRAM(RANDOMFILEBUFFERSIZE);
  
  SetupFile(&filRandomData,"rb",sFilenameRandomData,NULL,FALSE);

  if (filRandomData==NULL) {
    ReportPrint(pRepErr,"Error! Unable to read from random data file\n");
    AbnormalExit();
  }

  iRandomBufferRemaining = fread(pRandomDataBuffer,1,RANDOMFILEBUFFERSIZE,filRandomData);

  if ((iRandomBufferRemaining < RANDOMFILEBUFFERSIZE)||(feof(filRandomData))) {
    bCycleData = TRUE;
    iCycleDataLen = iRandomBufferRemaining;
  } else {
    bCycleData = FALSE;
  }

  pNextRandomData = pRandomDataBuffer;

  fxnRandUInt32 = &Ubcsat::FileRandomUInt32;
}

void Ubcsat::CloseFileRandom() {
  CloseSingleFile(filRandomData);
}

void Ubcsat::FileAbort() {

  FILE *fileAbort;
  
  fileAbort = fopen(sFilenameAbort,"r");

  if (fileAbort) {
    bTerminateAllRuns = TRUE;
    fclose(fileAbort);
  }
}

