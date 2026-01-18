
#!/bin/bash
EXPERIMENTS_FILE=$1
#JAR_CPLEX_PATH=/home/cdloaiza/apps/ibm/ILOG/CPLEX_Studio1210/cplex/lib/cplex.jar
#JAR_GSON_PATH=/home/cdloaiza/jars/gson-2.6.2.jar
#CPLEX_PATH=/home/cdloaiza/apps/ibm/ILOG/CPLEX_Studio1210/cplex/bin/x86-64_linux/
JAR_CPLEX_PATH=/opt/ibm/ILOG/CPLEX_Studio1210/cplex/lib/cplex.jar
JAR_GSON_PATH=/home/nuser/jars/gson-2.6.2.jar
CPLEX_PATH=/opt/ibm/ILOG/CPLEX_Studio1210/cplex/bin/x86-64_linux/
javac -classpath $JAR_CPLEX_PATH:$JAR_GSON_PATH -d $PWD/bin $PWD/src/*/*.java
java -Djava.library.path=$CPLEX_PATH -classpath $JAR_CPLEX_PATH:$JAR_GSON_PATH:$PWD/bin core.Executor $EXPERIMENTS_FILE "`date`" | ts > ../data/cplex-`date +'%a-%b-%d-%H-%M-%S-%Z-%Y'`.log

