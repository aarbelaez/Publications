
#GFLAGS=/Users/arbelaez/Home/KMedoids/Backhaul/gflags-install
#GFLAGS=/Users/aarbelaez/Home/Tools/gflags/gflags-install
GFLAGS=/home/aarbelaez/Backhaul/gflags-install

GFLAGS=/Users/alejandroarbelaez/Home/DISCUS/gflags-install
#LAMBDA=62.5

#OUTPUT=BH-LAMBDA_${LAMBDA}

OUTPUT=BH

#DEBUG=-DDEBUG=1

#DEBUG=-DDEBUG2=1 -DSCALE2=1 

DEBUG=-DSCALE2=1

#DEBUG=-DDEBUG2=1 -0fast

#FLAGS=-DDISTMATRIX=1 -DLAMBDA=${LAMBDA}

FLAGS=-DDISTMATRIX=1

all: Node.h MetroNode.h Solution.h Search.h Parameters.h BH.h LSearch.h read_files.h MNGraph.h TwoPred.h mt19937p.h Move.h FirstLevelList.h
	g++-12 ${DEBUG} ${FLAGS} -Wall -o ${OUTPUT} -I"${GFLAGS}/include" -g main.cpp TwoPred.cpp Node.cpp MetroNode.cpp LSearch.cpp read_files.cpp Solution.cpp mt19937p.c FirstLevelList.cpp ${GFLAGS}/lib/libgflags.a -fpermissive -fopenmp

all2: Node.h MetroNode.h Solution.h Search.h Parameters.h BH.h LSearch.h read_files.h MNGraph.h TwoPred.h
	g++ -Wall -o GenInstance -I"${GFLAGS}/include" -g GenInstances.cpp Node.cpp MetroNode.cpp LSearch.cpp read_files.cpp Solution.cpp ${GFLAGS}/lib/libgflags.a -fpermissive

clean:
	rm -rf ${OUTPUT}

#g++ -Wall -o BH -I"/Users/arbelaez/Home/KMedoids/Backhaul/gflags-install/include/" -g main.cpp Node.cpp MetroNode.cpp Solution.cpp tools.c ../gflags-install/lib/libgflags.a -fpermissive

