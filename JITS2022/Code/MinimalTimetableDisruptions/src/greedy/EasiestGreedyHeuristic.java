package greedy;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import core.InstanceMTD;
import heuristics.HeuristicsVariablesSet;
import heuristics.Stop;
import ilog.concert.IloException;
import ilog.concert.IloIntVar;
import ilog.concert.IloNumVar;
import utils.ToolsMTD;

/**
 * A simple greedy algorithm in which we charge only the necessary to reach the next station
 * @author cedaloaiza
 *
 */
public class EasiestGreedyHeuristic {
	
	private InstanceMTD instance;
	private HeuristicsVariablesSet variables;
	private StrategyRecharge rechargeStrategy;
	
	private StrategySolveRobustness robustSolver;
	
	private double totalAddedEnergy;
	private double totalCt;
	private double totalWaitingTime;
	
	private double currentEnergy;
	private double currentTime;
	
	public Set<Integer> openStations;
	
	
	
	public EasiestGreedyHeuristic(InstanceMTD instance, StrategyRecharge rechargeStrategy, 
			HeuristicsVariablesSet variables) {
		this.instance = instance;
		this.rechargeStrategy = rechargeStrategy;
		this.variables = variables;
		
		openStations = new HashSet<Integer>();
		
	}
	
	public void greedyAlgorithm() {
		
		
		ArrayList<Integer> busIds = new ArrayList<>();
		for (int b = 0; b < instance.b; b++) {
			busIds.add(b);
		}
		//Collections.shuffle(busIds);
		
		for (int b: busIds) {
			travelFrom(b, 0);	
		}
		variables.fixXs();
		variables.computeNumberOpenStations();
		variables.print("initialSolution");
		solveOverlappingConflicts();
		
	}
	
	public void travelFrom(int b, int ki) {
		CurrentStatus status = new CurrentStatus();
		status.reset();
		reset(b);
		for (int k = ki; k < instance.paths[b].length; k++) {
			updateStopVariables(b, k);
		}
		//System.out.printf("Last arrivalTime[%s] = %s\n", b, currentTime);
		//System.out.printf("Last arrivalEnergy[%s] = %s\n", b, currentEnergy);
	} 
	
	
	/**
	 * Update the variables of the stop to this route.
	 * If it requires extra energy from the previous open stop, it returns it
	 * @param b
	 * @param k
	 * @return Extra energy required from previous open stop
	 */
	public double updateStopVariables(int b, int k) {
		int station = instance.paths[b][k];
		//System.out.printf("[%s][%s]\n", b, k);
		//System.out.println(currentTime);
		//System.out.println(instance.T[b][k]);
		//System.out.println(currentEnergy);
		if (currentTime < instance.originalTimetable[b][k] - instance.DTmax) {
			//System.out.printf("Arriving early in [%s,%s]\n", b, k);
			double wastedTime = (instance.originalTimetable[b][k] - instance.DTmax) - currentTime;
			totalWaitingTime += wastedTime;
			currentTime = instance.originalTimetable[b][k] - instance.DTmax;
			
		}	else if (currentTime > instance.originalTimetable[b][k] + instance.DTmax) {
			System.out.printf("[%s][%s]\n", b, k);
			System.out.println("Current time: " + currentTime);
			System.out.println("Max arrival: " + (instance.originalTimetable[b][k] + instance.DTmax));
			System.out.println("Total added energy: " + totalAddedEnergy);
			System.out.println("Total ct: " + totalCt);
			System.out.println("Total waiting time: " + totalWaitingTime);
			System.out.println("The algorithm cannot find a solution for this instance");
			System.exit(1);
		}	
		variables.c[b][k] = currentEnergy;
		variables.arrivalTime[b][k] = currentTime;
		//System.out.printf("c[%s]: %s\n", k, variables.c[b][k]);
		//System.out.printf("t[%s]: %s\n", k, variables.arrivalTime[b][k]);
		if (k != instance.paths[b].length - 1) {
			int nextStation = instance.paths[b][k + 1];
			rechargeStrategy.updateStop(b, k, currentEnergy, currentTime);
			variables.ct[b][k] = variables.e[b][k] / instance.chargingRate;
			//System.out.printf("Original ct[%s]: %s\n", k, variables.ct[b][k]);
			//System.out.printf("Original e[%s]: %s\n", k, variables.e[b][k]);
			if (variables.e[b][k] > 0) {
				openStations.add(station);
				variables.x[station] = true;
				variables.xBStation[b][station] = true;
				variables.xBStop[b][k] = true;
			} else if (variables.e[b][k] == 0) {
				variables.xBStation[b][station] = false;
				variables.xBStop[b][k] = false;
			} else {
				System.out.println("Negative recharge!!!");
				System.exit(1);
			}
			//if (variables.ct[b][k] > 0) {
				boolean wereThereConflicts = false;
				boolean wereExploitingDwellTimes = false;
				double potentialNextArrivalTime = -1;
				double originalCt = -1;
				double originalE = -1;
				boolean wereCtNEchanged = false;
				boolean wereSomethingChangedByRobust = false;
				do {
					wereExploitingDwellTimes = false;
					//System.out.println(variables.ct[b][k]);
					if (variables.ct[b][k] > 0) {
						double newCurrentTime = newArrivalTimeFromConflicts(b, k);
						wereThereConflicts = variables.arrivalTime[b][k] != newCurrentTime;
						variables.arrivalTime[b][k] = newCurrentTime;
						//System.out.println("Conflicts?");
						if (wereThereConflicts) {
							System.out.printf("A conlict in [%s,%s]\n", b, k);
						}
					} else {
						wereThereConflicts = false;
					}
					
					//**Exploiting dwell times**
					/*
					if (potentialNextArrivalTime == -1 || wereThereConflicts) {
						potentialNextArrivalTime = variables.arrivalTime[b][k] + variables.ct[b][k] + instance.T[b][k];
					}
					//System.out.printf("%s < %s \n", potentialNextArrivalTime, instance.originalTimetable[b][k + 1] - instance.DTmax);
					if (potentialNextArrivalTime < instance.originalTimetable[b][k + 1] - instance.DTmax) {
						//System.out.println("Arriving early");
						double wastedTime = (instance.originalTimetable[b][k + 1] - instance.DTmax) - potentialNextArrivalTime;
						if (variables.xBStop[b][k]) {
							//System.out.printf("Exploiting dwell times in [%s,%s]\n", b, k);
							double maxAddingEnergy = Math.min(instance.Cmax - (variables.c[b][k] + variables.e[b][k]),
									instance.maxAddingEnergy - variables.e[b][k]);
							double maxAddingTime = maxAddingEnergy / instance.chargingRate;
							if (!wereCtNEchanged) {
								originalCt = variables.ct[b][k];
								originalE = variables.e[b][k];
							}
							double energyNeededToFinish = instance.addedEnergies.get(b) - (totalAddedEnergy + variables.e[b][k]);
							//double ctNeededToFinish = Math.max(energyNeededToFinish / instance.chargingRate, 0);
							double ctNeededToFinish = energyNeededToFinish / instance.chargingRate;
							variables.ct[b][k] += Math.min(Math.min(wastedTime, maxAddingTime), ctNeededToFinish);
							variables.e[b][k] = variables.ct[b][k] * instance.chargingRate;
							//System.out.printf("New ct[%s]: %s\n", k, variables.ct[b][k]);
							//System.out.printf("New e[%s]: %s\n", k, variables.e[b][k]);
							if (variables.e[b][k] < 0) {
								variables.e[b][k] = 0;//1;
								variables.ct[b][k] = variables.e[b][k] / instance.chargingRate;
							}							
							wereCtNEchanged = true;
							wereExploitingDwellTimes = true;
							potentialNextArrivalTime = instance.originalTimetable[b][k + 1] - instance.DTmax;
							if (variables.e[b][k] == 0) {
								variables.xBStop[b][k] = false;
								wereExploitingDwellTimes = false;
							}
						}
					} else if (wereCtNEchanged && wereThereConflicts) {
						variables.ct[b][k] = originalCt; 
						variables.e[b][k] = originalE;
						System.out.printf("Here. Really!?\n");
					}
					*/
					//**End of Exploiting dwell times**
					//**Robustness**
					//It is delaying arrival time
					if (robustSolver != null) {
						wereSomethingChangedByRobust = robustSolver.hasTimeRobustProblemsSolved(b, k);
					}
					/*
					System.out.printf("b: %s, k: %s\n", b, k);
					System.out.printf("wereThereConflicts: %s\n", wereThereConflicts);
					System.out.printf("wereExploitingDwellTimes: %s\n", wereExploitingDwellTimes);
					System.out.printf("wereSomethingChangedByRobust: %s\n", wereSomethingChangedByRobust);
					System.out.printf("e: %s\n", variables.e[b][k]);
					System.out.printf("ct: %s\n", variables.ct[b][k]);
					
					if (b == 15 && k == 225) {
						//System.exit(0);
					}
					*/
					//**End of robustness**
				} while (wereThereConflicts || wereExploitingDwellTimes || wereSomethingChangedByRobust);
			//}
			if (robustSolver != null) {
				double extraNeededEnergy = robustSolver.energyNeededToSolveRobustProblems(b, k);
				if (extraNeededEnergy > 0) {
					return extraNeededEnergy;
				}
			}
			
			// For next iteration
			currentEnergy = currentEnergy + variables.e[b][k] - instance.D[station][nextStation];
			currentTime = variables.arrivalTime[b][k] + variables.ct[b][k] + instance.T[b][k];
			
			if (currentEnergy < instance.Cmin) {
				return instance.Cmin - currentEnergy;
			}
			////
			//printVariables(b, k, station);
			//System.out.printf("instance.T[%s][%s]=%s\n", b, k, instance.T[b][k]);
			//System.out.printf("instance.D[%s][%s]=%s\n\n", station, nextStation, instance.D[station][nextStation]);
			
		} else {
			variables.xBStation[b][station] = false;
			variables.xBStop[b][k] = false;
			variables.e[b][k] = 0;
			variables.ct[b][k] = 0;
			//printVariables(b, k, station);
		}
		//printVariables(b, k, station);
		totalAddedEnergy += variables.e[b][k];
		totalCt += variables.ct[b][k];
		return 0;
	}
	
	/**
	 * Reset values when changing the bus
	 */
	public void reset(int b) {
		currentEnergy = instance.Cmax;
		currentTime = -100000;
		currentTime = Math.max(instance.originalTimetable[b][0] - instance.DTmax, 0);
		totalAddedEnergy = 0;
		totalCt = 0;
		totalWaitingTime = 0;
	}
	
	/**
	 * Returns the new arrivalTime_{bk} considering the conflicts with times already assigned
	 * @param currentB 
	 * @param currentK
	 * @return
	 */
	public double newArrivalTimeFromConflicts(int currentB, int currentK) {
		double currentArrivalTime = variables.arrivalTime[currentB][currentK];
		double latestConflictDeparture = currentArrivalTime;
		int currentStation = instance.paths[currentB][currentK];
		for (int b = 0; b < currentB; b++) {
			//int kLimit = b == currentB ? currentK : instance.paths[b].length;
			for (int k = 0; k < instance.paths[b].length; k++) {
				int candidateStation = instance.paths[b][k];
				if (currentStation != candidateStation || variables.ct[b][k] == 0) {
					//System.out.printf("b=%s, k=%s\n", b, k);
					continue;
				} else {
					double departureTimeB = currentArrivalTime + variables.ct[currentB][currentK];
					double departureTimeB2 = variables.arrivalTime[b][k] + variables.ct[b][k];
					/*
					if (currentB == 21 && currentK ==  348) {
						System.out.printf("current: b=%s, k=%s; candidate: b=%s, k=%s\n",
								currentB, currentK, b, k);
						System.out.printf("%s >= %s && %s < %s || %s > %s && %s <= %s\n",
								currentArrivalTime, variables.arrivalTime[b][k], currentArrivalTime, departureTimeB2,
							departureTimeB, variables.arrivalTime[b][k], departureTimeB, departureTimeB2);
					}
					*/
					if (currentArrivalTime >= variables.arrivalTime[b][k] && currentArrivalTime < departureTimeB2 ||
							variables.arrivalTime[b][k] >= currentArrivalTime && variables.arrivalTime[b][k] < departureTimeB ) {
						//System.out.printf("HOLA current: b=%s, k=%s; candidate: b=%s, k=%s\n",
						//		currentB, currentK, b, k);
						//System.out.println(departureTimeB2);
						latestConflictDeparture = departureTimeB2 > latestConflictDeparture ? departureTimeB2 : latestConflictDeparture; 
						
					}
				}
			}
		}
		//System.out.println(latestConflictDeparture);
		//System.out.println();
		return latestConflictDeparture;
	}
	
	/**
	 * @deprecated
	 */
	public void solveOverlappingConflicts() {
		for (int i = 0; i < instance.n; i++) {
			//System.out.println(i);
			Set<Stop> conflicts = new HashSet<Stop>();
			for(int b = 0; b < instance.b; b++) {
				for(int k = 0; k < instance.paths[b].length; k++) {
					int bStation = instance.paths[b][k];
					if (bStation != i || variables.ct[b][k] == 0) {
						//System.out.printf("b=%s, k=%s\n", b, k);
						continue;
					}
					if (b != instance.b - 1) {
						for(int b2 = b + 1; b2 < instance.b; b2++) {
							for(int m = 0; m < instance.paths[b2].length; m++) {
								int b2Station = instance.paths[b2][m];
								if (b2Station != i || variables.ct[b2][m] == 0) {
									continue;
								}
								double departureTimeB = variables.arrivalTime[b][k] + variables.ct[b][k];
								double departureTimeB2 = variables.arrivalTime[b2][m] + variables.ct[b2][m];
								if (variables.arrivalTime[b][k] >= variables.arrivalTime[b2][m] && variables.arrivalTime[b][k] < departureTimeB2 ||
										departureTimeB > variables.arrivalTime[b2][m] && departureTimeB <= departureTimeB2) {
									conflicts.add(new Stop(b, k, variables.arrivalTime[b][k]));
									conflicts.add(new Stop(b2, m, variables.arrivalTime[b2][m]));		
								}
								/*
								System.out.printf("%s >= %s && %s < %s || %s >= %s && %s < %s",
										variables.arrivalTime[b][k], variables.arrivalTime[b2][m], variables.arrivalTime[b][k], departureTimeB2,
										departureTimeB, variables.arrivalTime[b2][m], departureTimeB, departureTimeB2);
								*/
							}
						}
					}
				}
			}
			
			
			ArrayList<Stop> conflictsToOrder = new ArrayList<Stop>(conflicts);
			conflictsToOrder.sort((s1, s2) -> s1.arrivalTime.compareTo(s2.arrivalTime));
			if (conflicts.size() > 0) {
				System.out.println(conflicts.size());
				for (Stop stop: conflictsToOrder) {
					System.out.printf("b=%s, k=%s, t=%s\n", stop.bus, stop.stop, stop.arrivalTime);		
				}
				System.out.println();
			}
		}
	}
	
	
	
	public void printVariables(int b, int k, int i) {
		System.out.printf("x[%s]=%s\n", i,  variables.x[i]);
		System.out.printf("xBStation[%s][%s]=%s\n", b, i,  variables.xBStation[b][i]);
		System.out.printf("xBStop[%s][%s]=%s\n", b, k,  variables.xBStop[b][k]);
		System.out.printf("c[%s][%s]=%s\n", b, k,  variables.c[b][k]);
		System.out.printf("e[%s][%s]=%s\n", b, k,  variables.e[b][k]);
		System.out.printf("ct[%s][%s]=%s\n", b, k,  variables.ct[b][k]);	
		int minutesT = (int) Math.round(variables.arrivalTime[b][k] / 60);
		int minutesOT = (int) Math.round(instance.originalTimetable[b][k] / 60);
		System.out.printf("arrivalTime[%s][%s]=%s / %s\n", b, k, variables.arrivalTime[b][k], 
				ToolsMTD.minutesToStringTime(minutesT));	
		System.out.printf("originalTime[%s][%s]=%s / %s\n", b, k,  instance.originalTimetable[b][k],
				ToolsMTD.minutesToStringTime(minutesOT));	
		System.out.println();
	}
	
	public HeuristicsVariablesSet getSolution() {
		return variables;
	}

	public void setRobustSolver(StrategySolveRobustness robustSolver) {
		this.robustSolver = robustSolver;
	}
	
	public void restoreTimeEnergyStatus(int b, int k) {
		currentEnergy = variables.c[b][k];
		//System.out.printf("Current energy by [%s][%s]: %s\n", b, k, currentEnergy);
		currentTime = variables.arrivalTime[b][k];
		variables.c[b][k] = 0;
		variables.arrivalTime[b][k] = 0;
	}

	public double getCurrentEnergy() {
		return currentEnergy;
	}
	
	
}
