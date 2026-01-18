package heuristics;

import java.util.LinkedList;
import java.util.ListIterator;


import utils.ToolsMTD;
import variables.VariablesSet;

public class OverlappingConflictSolver {
	
	OpenStationsStructure openStationsStructure;
	HeuristicsVariablesSet currentSolution;
	
	
	
	public OverlappingConflictSolver(CompleteSolution currentSolution) {
		super();
		this.currentSolution = currentSolution.normalVars;
		this.openStationsStructure = currentSolution.openStationsStructure;
	}
	
	/**
	 * Check if the new arrival time and charging time of bus b at stop k is causing overlapping conflicts
	 * @param arrivalTime
	 * @param chargingTime
	 * @param b
	 * @param k
	 * @param i
	 * @return
	 */
	public boolean checkConflicts(double arrivalTime, double chargingTime, int b, int k, int i) {
		boolean isOverlappingFeasible = true;
		//System.out.println("Number of stops in station " + i + ": " + openStationsStructure.getNumOpenStopsPerStations(i));
		if (openStationsStructure.getNumOpenStopsPerStations(i) > 0) {
			ListIterator<OpenStation> potentiallyConflictStops = openStationsStructure.getListIteratorOpenStopsPerStation(i);
			while (potentiallyConflictStops.hasNext()) {
				OpenStation posConflictStop = potentiallyConflictStops.next();
				if ((b == posConflictStop.bus && k == posConflictStop.stop) ||
					(ToolsMTD.round(chargingTime) == 0) || 
					(ToolsMTD.round(currentSolution.ct[posConflictStop.bus][posConflictStop.stop]) == 0)) {
					continue;
				}
				//System.out.println("considered");
				if (currentSolution.arrivalTime[posConflictStop.bus][posConflictStop.stop] >= 
						arrivalTime) {
					double latestArrivalTime = ToolsMTD.round(currentSolution.arrivalTime[posConflictStop.bus][posConflictStop.stop]);
					double busyTimeByEarliest = ToolsMTD.round(arrivalTime + chargingTime);
					isOverlappingFeasible = isOverlappingFeasible && (latestArrivalTime >= busyTimeByEarliest);
					
					//System.out.printf("Overlapping check: %s >= %s + %s [%s][%s]-[%s][%s]\n", latestArrivalTime,
					//		arrivalTime, chargingTime, posConflictStop.bus, posConflictStop.stop, b, k);
							
					
				} else if (arrivalTime >= currentSolution.arrivalTime[posConflictStop.bus][posConflictStop.stop]) {
					double latestArrivalTime = ToolsMTD.round(arrivalTime);
					double busyTimeByEarliest = ToolsMTD.round(currentSolution.arrivalTime[posConflictStop.bus][posConflictStop.stop] + 
							currentSolution.ct[posConflictStop.bus][posConflictStop.stop]);
					isOverlappingFeasible = isOverlappingFeasible && (latestArrivalTime >= busyTimeByEarliest);
					
					/*
					System.out.printf("Overlapping check: %s >= %s + %s [%s][%s]-[%s][%s]\n", latestArrivalTime,
							currentSolution.arrivalTime[posConflictStop.bus][posConflictStop.stop], 
							currentSolution.ct[posConflictStop.bus][posConflictStop.stop],
							b, k, posConflictStop.bus, posConflictStop.stop);
					*/
					
				}
			} 
		}
		return isOverlappingFeasible;
		
	}


}
