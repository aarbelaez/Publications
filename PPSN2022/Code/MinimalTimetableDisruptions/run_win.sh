
#!/bin/bash
EXPERIMENTS_FILE=$1
#JAR_CPLEX_PATH=/home/cdloaiza/apps/ibm/ILOG/CPLEX_Studio1210/cplex/lib/cplex.jar
#JAR_GSON_PATH=/home/cdloaiza/jars/gson-2.6.2.jar
#CPLEX_PATH=/home/cdloaiza/apps/ibm/ILOG/CPLEX_Studio1210/cplex/bin/x86-64_linux/
JAR_CPLEX_PATH="C:\Program Files\IBM\ILOG\CPLEX_Studio1210\cplex\lib\cplex.jar"
JAR_GSON_PATH="F:\Documents\jars\gson-2.8.6.jar"
CPLEX_PATH="C:\Program Files\IBM\ILOG\CPLEX_Studio1210\cplex\bin\x64_win64"
#javac -classpath $JAR_CPLEX_PATH;$JAR_GSON_PATH -d $PWD/bin $PWD/src/*/*.java
java  "-Djava.library.path=C:\Program Files\IBM\ILOG\CPLEX_Studio1210\cplex\bin\x64_win64" -Dfile.encoding=UTF-8 -classpath "C:\Users\cdlq1\Documents\git\Cesar\MinimalTimetableDisruptions\bin;C:\Program Files\IBM\ILOG\CPLEX_Studio1210\cplex\lib\cplex.jar;C:\Users\cdlq1\Documents\jars\gson-2.8.6.jar" core.Executor $EXPERIMENTS_FILE "`date`"
#java  "-Djava.library.path=C:\Program Files\IBM\ILOG\CPLEX_Studio1210\cplex\bin\x64_win64" -Dfile.encoding=UTF-8 -classpath "F:\git\Cesar\MinimalTimetableDisruptions\bin;C:\Program Files\IBM\ILOG\CPLEX_Studio1210\cplex\lib\cplex.jar;F:\Documents\jars\gson-2.8.6.jar" core.Executor $EXPERIMENTS_FILE "`date`"

