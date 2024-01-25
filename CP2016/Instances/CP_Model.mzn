%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%MODEL:	 Constraint Programming Formulation %
%AUTHORS:	Robinson Duque, Alejandro Arbelaez, Juan Francisco Díaz	%
%CONTACT:  	robinson.duque@correounivalle.edu.co			%
%LAST UPDATE:	October, 2016						%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%DATA
int:n; %Number of teams
set of int: TEAMS=1..n;
array[TEAMS] of int: iPoints; %Initial points
array[TEAMS,TEAMS] of int: fixtures;%Not used in this model
array[int,int] of int: games;%Games to play
array[int,int] of int: positionConstraints;%Position constraints
array[int,int] of int: relativeConstraints;%Position constraints
array[int,int] of int: finalPointConstraints;%Final point constraints


set of int: pt={0,1,3};
set of int: numberOfGames=1..(length(games) div 2);
%DECISION VARIABLES
array[numberOfGames, 1..2] of var pt: points;


%The addition of the points of team 1 and 2 must be a valid point assignment (0,3), (3,0) or (1,1):
constraint forall(g in numberOfGames)(
points[g,1]+points[g,2]>=2 /\ points[g,1]+points[g,2]<=3);


%The final points of team i are the addition of its initial points and the points it gets in the games.
array[TEAMS] of var int: fPoints;%Final points

constraint forall(i in TEAMS)( 
   fPoints[i] = iPoints[i]  + sum(j in numberOfGames where i=games[j,1]\/i=games[j,2])(
   if i=games[j,1] then
      points[j,1]
   else
      points[j,2]
   endif
   ));


%POSITION CALCULATION
array[TEAMS] of var TEAMS: finalPosition;%Final positions
array[TEAMS] of var TEAMS: bestPosition;%Best position
array[TEAMS] of var TEAMS: worstPosition;%worst position

%The worst position of team i is given by the number of teams with greater total points than team i
constraint forall(i,j in TEAMS)( 
  worstPosition[i]= sum(j in TEAMS)(bool2int(fPoints[j] >= fPoints[i])) 
);
%The best position of team i is given by the subtraction of the worst position and the number of teams with equal points as i.
constraint forall(i,j in TEAMS)( 
  bestPosition[i]= worstPosition[i] - sum(j in TEAMS where i!=j)(bool2int(fPoints[j] == fPoints[i])) 
);

%The position of a team is bounded by the worst and best position that it can take.
constraint forall(i in TEAMS)(finalPosition[i]>=bestPosition[i] /\  finalPosition[i]<=worstPosition[i]);
%All the final positions must be different
include "alldifferent.mzn";   
constraint alldifferent(finalPosition);




%POSITION CONSTRAINTS
int: numberPositionConstraints=length(positionConstraints) div 3; %Find out the number of position constraints

constraint forall(i in 1..numberPositionConstraints) (
  if positionConstraints[i,2] = 1 then
     finalPosition[positionConstraints[i,1]]=positionConstraints[i,3]
  elseif positionConstraints[i,2] = 2 then
     finalPosition[positionConstraints[i,1]]>positionConstraints[i,3]
  elseif positionConstraints[i,2] = 3 then
     finalPosition[positionConstraints[i,1]]>=positionConstraints[i,3]
  elseif positionConstraints[i,2] = 4 then
     finalPosition[positionConstraints[i,1]]<positionConstraints[i,3]
  else
     finalPosition[positionConstraints[i,1]]<=positionConstraints[i,3]  
  endif
);


%RELATIVE POSITION CONSTRAINTS
int: numberRelativeConstraints=length(relativeConstraints) div 3;  %Find out the number of relative position constraints
constraint forall(i in 1..numberRelativeConstraints) (
  if relativeConstraints[i,2] = 1 then
     fPoints[relativeConstraints[i,1]]=fPoints[relativeConstraints[i,3]]
  elseif relativeConstraints[i,2] = 2 then
     fPoints[relativeConstraints[i,1]]>fPoints[relativeConstraints[i,3]]
  elseif relativeConstraints[i,2] = 3 then
     fPoints[relativeConstraints[i,1]]>=fPoints[relativeConstraints[i,3]]
  elseif relativeConstraints[i,2] = 4 then
     fPoints[relativeConstraints[i,1]]<fPoints[relativeConstraints[i,3]]
  else
     fPoints[relativeConstraints[i,1]]<=fPoints[relativeConstraints[i,3]]
  endif
);




%FINAL POINT CONSTRAINTS
int: numberFinalPointConstraints=length(finalPointConstraints) div 3;  %Find out the number of final point constraints
constraint forall(i in 1..numberFinalPointConstraints) (
  if finalPointConstraints[i,2] = 1 then
     fPoints[finalPointConstraints[i,1]]=finalPointConstraints[i,3]
  elseif finalPointConstraints[i,2] = 2 then
     fPoints[finalPointConstraints[i,1]]>finalPointConstraints[i,3]
  elseif finalPointConstraints[i,2] = 3 then
     fPoints[finalPointConstraints[i,1]]>=finalPointConstraints[i,3]
  elseif finalPointConstraints[i,2] = 4 then
     fPoints[finalPointConstraints[i,1]]<finalPointConstraints[i,3]
  else
     fPoints[finalPointConstraints[i,1]]<=finalPointConstraints[i,3]
  endif
);



output["Game Results:\nTeam1 \tpts \tTeam2 \tpts: \n" ]++
      [show_int(2, games[i,j])++"\t" ++ show_int(2, points[i,j])++"\t" ++ if j=2 then "\n" else " " endif | i in numberOfGames, j in 1..2]++
      ["Initial Points=\(iPoints)\n"]++
      ["Final Positions=\(finalPosition)\n"]++
      ["Final Points=\(fPoints)\n"];


solve satisfy;
