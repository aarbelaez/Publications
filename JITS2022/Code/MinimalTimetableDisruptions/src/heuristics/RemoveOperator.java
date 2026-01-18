package heuristics;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.Random;
import java.util.logging.Logger;

import core.InstanceMTD;
import utils.ToolsMTD;

public class RemoveOperator {
	
	HeuristicsVariablesSet currentSolution;
	InstanceMTD instance;
	
	double minEnergyNeeded;
	double minNeededChargingTime;
	double remainingEnergyPrev;
	double remainingEnergyNext;
	double remainingChargingTimePrev;
	
	Random rn;
	

	OpenStationsStructure openStationsStructure;
	HashMap<Integer, OpenStation> openStations;
	
	OpenStation selectedStation;
	
	OverlappingConflictSolver overlappingConflictSolver;
	
	CompleteSolution completeCurrentSolution;
	
	protected final static Logger LOGGER = Logger.getLogger(RemoveOperator.class.getName());
	
	HeuristicsVariablesSet updatingSolution;
	
	public RemoveOperator(CompleteSolution currentSolution, InstanceMTD instance, Random rn) {
		completeCurrentSolution = currentSolution;
		this.currentSolution = currentSolution.normalVars;
		this.instance = instance;
		this.openStationsStructure = currentSolution.openStationsStructure;
		this.rn = rn;
		overlappingConflictSolver = new OverlappingConflictSolver(currentSolution);
		LOGGER.setLevel(instance.loggerLevel);
	}
	
	public void resetOperator(CompleteSolution currentSolution) {
		completeCurrentSolution = currentSolution;
		this.currentSolution = currentSolution.normalVars;
		this.openStationsStructure = currentSolution.openStationsStructure;
		overlappingConflictSolver = new OverlappingConflictSolver(currentSolution);
		openStations = null;
		selectedStation = null;
		minEnergyNeeded = 0;
		minNeededChargingTime = 0;
		remainingEnergyPrev = 0;
		remainingEnergyNext = 0;
		remainingChargingTimePrev = 0;
	}
	
	/*
	public boolean isFeasible(int b, int selectedStopIndex) {
		openStations = openStationsStructure.getListIteratorOpenStations(b);
		int j = 0;
		while (openStations.hasNext()) {
			selectedStation = openStations.next();
			if (j == selectedStopIndex) {
				LOGGER.info(String.format("Selected stop: [%s][%s] station %s\n", b, selectedStation.stop,
						instance.paths[b][selectedStation.stop]));
				return checkFeasibility(b, selectedStation);
			}
			j++;	
		}
		return false;	
	}
	*/
	
	public boolean isFeasible2(OpenStation theSelectedStop) {
		
		this.selectedStation = theSelectedStop;
		
		minEnergyNeeded = 0;
		minNeededChargingTime = 0;
		remainingEnergyPrev = 0;
		remainingEnergyNext = 0;
		remainingChargingTimePrev = 0;
		
		
		return checkFeasibility(theSelectedStop);
		
		
	}
	
	/*
	public boolean isFeasibleStation(int i, int selectedStopIndex) {
		openStations = openStationsStructure.getListIteratorOpenStopsPerStation(i);
		int j = 0;
		while (openStations.hasNext()) {
			selectedStation = openStations.next();
			if (j == selectedStopIndex) {
				LOGGER.info(String.format("Selected stop: [%s][%s] station %s\n", selectedStation.bus, selectedStation.stop,
						instance.paths[selectedStation.bus][selectedStation.stop]));
				return checkFeasibility(selectedStation.bus, selectedStation);
			}
			j++;	
		}
		return false;
		
	}
	*/
	
	public boolean checkFeasibility(OpenStation selectedStation) {
		
		int b = selectedStation.bus;
		
		boolean energyTimeFeasibility = false;
		boolean isOverlappingFeasible = true;
		
		if (selectedStation.hasPrevious()) {
			OpenStation prevStation = selectedStation.previous();
			//System.out.printf("prev during checking [%s][%s]\n", b, prevStation.stop);
			
		//// Energy and time feasibility
			// The current energy with which the bus leaves the previous station
			double currentPrevEnergy = currentSolution.c[b][prevStation.stop] + currentSolution.e[b][prevStation.stop];
	
			boolean hasNext = false;
			minEnergyNeeded = currentSolution.e[b][selectedStation.stop];
			double addedNextEnergy = 0;
			double arrivalEnergyNext = 0;
			double arrivalTimeNext = 0;
			double ctNext = 0;
			int nextK = -1;
			if (selectedStation.hasNext()) {
				OpenStation nextStation = selectedStation.next();	
				nextK = nextStation.stop;
				//System.out.println("nextK: " + nextStation.stop);
				//System.out.printf("next on a middle stop during checking [%s][%s]\n", b, nextK);
				double neededEnergyToNext = selectedStation.E + nextStation.E;
				//System.out.printf("E%s = %s, E%s = %s\n", selectedStation.stop, selectedStation.E, nextK, nextStation.E);
				minEnergyNeeded = neededEnergyToNext + instance.Cmin - currentPrevEnergy;
				//System.out.printf("minEnergyNeeded = %s + %s - %s\n", neededEnergyToNext, instance.Cmin, currentPrevEnergy);
				minEnergyNeeded = Math.max(minEnergyNeeded, 0);
				hasNext = true;
				addedNextEnergy = currentSolution.e[b][nextStation.stop];
				arrivalEnergyNext = currentSolution.c[b][nextStation.stop];
				arrivalTimeNext = currentSolution.arrivalTime[b][nextStation.stop];
				ctNext = currentSolution.ct[b][nextStation.stop];
				//System.out.println("Removing middle stop...");
			}
			/*
			if (!hasNext) {
				System.out.println("Removing last stop...");
			}
			System.out.printf("e[%s][%s]=%s\n", b, selectedStation.stop, currentSolution.e[b][selectedStation.stop]);
			System.out.printf("minEnergyNeeded=%s\n", minEnergyNeeded);
			*/
			/*
			 * THIS MUST BE IMPROVED SOON!!!
			 */
			boolean minCapacityFeasibility = ToolsMTD.round(currentPrevEnergy + minEnergyNeeded ) <= instance.Cmax;
			minNeededChargingTime = minEnergyNeeded / instance.chargingRate;
			boolean minTimeFeasibility = ToolsMTD.round(selectedStation.s - minNeededChargingTime) >= 0;
			if (minCapacityFeasibility && minTimeFeasibility) {
				double selectedE = currentSolution.e[b][selectedStation.stop];
				//System.out.printf("%s < %s\n", minEnergyNeeded, selectedE);
				if (ToolsMTD.round(minEnergyNeeded) < ToolsMTD.round(selectedE)) {
					double p = rn.nextDouble();
					//System.out.println("We were setting remainings");
					remainingEnergyPrev = p*(selectedE - minEnergyNeeded);
					remainingEnergyNext = (1-p)*(selectedE - minEnergyNeeded);
					//System.out.printf("situ me:%s, rep:%s, ren:%s\n", minEnergyNeeded,
					//		remainingEnergyPrev, remainingEnergyNext);
					if (ToolsMTD.round(currentPrevEnergy + minEnergyNeeded + remainingEnergyPrev) > instance.Cmax) {
						remainingEnergyPrev = instance.Cmax - (currentPrevEnergy + minEnergyNeeded);
						remainingEnergyNext = selectedE - minEnergyNeeded - remainingEnergyPrev;
						//System.out.println("Energy overflowing");
						//System.out.printf("%s + %s + %s > %s", currentPrevEnergy, minEnergyNeeded,
						//		remainingEnergyPrev, instance.Cmax);
					}
					boolean energyFeasibility = true;
					boolean timeFeasibility = true;
					if (hasNext) {
						//energyFeasibility = addedNextEnergy + remainingEnergyNext <= instance.Cmax;
						energyFeasibility = energyFeasibility && 
								(ToolsMTD.round(arrivalEnergyNext - remainingEnergyNext) >= instance.Cmin);
						/*
						System.out.println("nextK: " + nextK);
						System.out.printf("%s - %s  >= %s\n", arrivalEnergyNext, remainingEnergyNext,
								 instance.Cmin);
						if (!energyFeasibility) {
							System.out.println("Cannot be?");
							currentSolution.print("holaco.txt");
							System.exit(0);
						}
						*/
						
						/**
						 * Overlapping conflicts
						 */
						double remainingChargingTimeNext = remainingEnergyNext / instance.chargingRate;
						double newArrivalTimeNext = arrivalTimeNext - remainingChargingTimeNext;
						double newCTNext = ctNext + remainingChargingTimeNext;
						//System.out.println(newCTNext);
						int stationID = instance.paths[b][nextK];
						isOverlappingFeasible = isOverlappingFeasible && overlappingConflictSolver.checkConflicts(
								newArrivalTimeNext, newCTNext, b, nextK, stationID);
						
						if (newCTNext > instance.maxChargingTime) {
							timeFeasibility = false;
						}
					} else {
						LOGGER.info("Removing last stop");
					}
					remainingChargingTimePrev = remainingEnergyPrev / instance.chargingRate;
					timeFeasibility = timeFeasibility && ToolsMTD.round(selectedStation.s - minNeededChargingTime - remainingChargingTimePrev) >= 0;
					energyTimeFeasibility = timeFeasibility && energyFeasibility;
				} 
				else if (ToolsMTD.round(minEnergyNeeded) > ToolsMTD.round(selectedE)) {
					remainingEnergyPrev = 0;
					remainingEnergyNext = 0;
					remainingChargingTimePrev = 0;
					energyTimeFeasibility = false;
				} else {
					remainingEnergyPrev = 0;
					remainingEnergyNext = 0;
					remainingChargingTimePrev = 0;
					energyTimeFeasibility = true;
				}
			} else {
				energyTimeFeasibility = false;
			}
		
		
			//// Overlapping feasibility
			int prevStationID = instance.paths[b][prevStation.stop];
			isOverlappingFeasible = isOverlappingFeasible && overlappingConflictSolver.checkConflicts(currentSolution.arrivalTime[b][prevStation.stop],
					currentSolution.ct[b][prevStation.stop] + minNeededChargingTime + remainingChargingTimePrev,
					b, prevStation.stop, prevStationID);
			
			
			double newPrevCt = currentSolution.ct[b][prevStation.stop] + minNeededChargingTime + remainingChargingTimePrev;
			//LOGGER.info(String.valueOf(newPrevCt));
			if (newPrevCt > instance.maxChargingTime) {
				energyTimeFeasibility = false;
			}
		} else {
			LOGGER.info("Removing first station");
			if (selectedStation.hasNext()) {
				OpenStation nextStation = selectedStation.next();	
				//System.out.printf("next on first stop during checking [%s][%s]\n", b, nextStation.stop);
				/**
				 * TODO
				 * This is unexplainable. Correction is next
				 */
				/*
				boolean energyFeasibility = ToolsMTD.round(currentSolution.c[b][selectedStation.stop] - nextStation.E) >=
						instance.Cmin;
				System.out.printf("%s - %s >= %s\n", currentSolution.c[b][selectedStation.stop], nextStation.E,
						instance.Cmin);
				*/
				// We use e[b][selectedStation] because un this case all the energy is going to the next station
				boolean energyFeasibility = ToolsMTD.round(currentSolution.c[b][nextStation.stop] - 
						currentSolution.e[b][selectedStation.stop]) >= instance.Cmin;
				//System.out.printf("%s - %s >= %s\n", currentSolution.c[b][nextStation.stop], currentSolution.e[b][selectedStation.stop],
				//		instance.Cmin);
				//boolean capacityFeasibility = ToolsMTD.round(currentSolution.c[b][nextStation.stop] + currentSolution.e[b][nextStation.stop] +
				//		currentSolution.e[b][selectedStation.stop]) <= instance.Cmax;
				boolean capacityFeasibility = true;
				energyTimeFeasibility = energyFeasibility && capacityFeasibility;
				
				if (energyTimeFeasibility) {
					minEnergyNeeded = 0;
					minNeededChargingTime = 0;
					remainingEnergyPrev = 0;
					remainingEnergyNext = currentSolution.e[b][selectedStation.stop];
					remainingChargingTimePrev = 0;
				}			
				/**
				 * Overlapping conflicts
				 */
				double arrivalTimeNext = currentSolution.arrivalTime[b][nextStation.stop];
				double ctNext = currentSolution.ct[b][nextStation.stop];
				double remainingChargingTimeNext = remainingEnergyNext / instance.chargingRate;
				double newArrivalTimeNext = arrivalTimeNext - remainingChargingTimeNext;
				//System.out.printf("original t in checking: t[%s][%s]=%s\n", b, nextStation.stop, arrivalTimeNext);
				//System.out.printf("New next t in checking: t[%s][%s]=%s \n", b, nextStation.stop, newArrivalTimeNext);
				double newCTNext = ctNext + remainingChargingTimeNext;
				int stationID = instance.paths[b][nextStation.stop];
				isOverlappingFeasible = isOverlappingFeasible && overlappingConflictSolver.checkConflicts(
						newArrivalTimeNext, newCTNext, b, nextStation.stop, stationID);
								
				if (newCTNext > instance.maxChargingTime) {
					energyTimeFeasibility = false;
				}
				
			} else {
				energyTimeFeasibility = false;
			}
		}
		
		
		
		//System.out.printf("end me:%s, rep:%s, ren:%s\n", minEnergyNeeded, remainingEnergyPrev, remainingEnergyNext);
		
		/*
		System.out.println("Fesasibility:");
		System.out.printf("Energy: %s\n", isEnergyFeasible);
		System.out.printf("Time: %s\n", isTimeFeasible);
		System.out.printf("Overlapping: %s\n", isOverlappingFeasible);
		*/
		/*
		if (energyTimeFeasibility && isOverlappingFeasible) {
			try {
				Thread.sleep(5*1000);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		*/
		//System.out.println(energyTimeFeasibility && isOverlappingFeasible);
		return energyTimeFeasibility && isOverlappingFeasible;
		
	}
	
	
	public void removeStation() {
		int b = selectedStation.bus;
		
		/*
		openStations = openStationsStructure.getListIteratorOpenStations(b);
		while (openStations.hasNext()) {
			OpenStation s = openStations.next();
			//System.out.println(s.stop);
			if (selectedStation.bus == s.bus && selectedStation.stop == s.stop) {
				break;
			}
		}
		*/
		
		int k = selectedStation.stop;
		
		//System.out.printf("removing [%s][%s]\n", b, k);
		
		double nextS = 100000;
		if (selectedStation.hasPrevious()) {
			OpenStation prevStation = selectedStation.previous();		
			
			int prevK = prevStation.stop;
			
			//System.out.printf("prev during performing [%s][%s]\n", b, prevK);
			//System.out.printf("e[%s][%s] += %s + %s\n", b, prevK, minEnergyNeeded, remainingEnergyPrev);
			
			currentSolution.e[b][prevK] += minEnergyNeeded + remainingEnergyPrev;
			currentSolution.ct[b][prevK] += minNeededChargingTime + remainingChargingTimePrev;
			
			for (int j = prevK+1; j <= k; j++) {
				//System.out.println(j);
				currentSolution.c[b][j] += minEnergyNeeded + remainingEnergyPrev;
				currentSolution.arrivalTime[b][j] +=  minNeededChargingTime + remainingChargingTimePrev;
				double delayAvailable = instance.DTmax - currentSolution.getDeltaT(b, j);
				nextS = delayAvailable < nextS ? delayAvailable : nextS;
			}
		}
		
		currentSolution.e[b][k] = 0;
		currentSolution.ct[b][k] = 0;
		currentSolution.xBStop[b][k] = false;
		
		int selectedStationId = instance.paths[b][k];
		
		if (!selectedStation.hasPrevious()) {
			openStationsStructure.headsIdPerBus.put(b, selectedStation.next().stop);
		}
		selectedStation.closeOpenStation();
		openStationsStructure.removeStopFromOpenStopsPerBus(b, k);
		openStationsStructure.removeStopFromOpenStopsPerStation(b, selectedStationId, k);
		
		
		if (openStationsStructure.getNumOpenStopsPerStations(selectedStationId) == 0) {
			currentSolution.x[selectedStationId] = false;
		}
		
		
		
		if (selectedStation.hasNext()) {
			OpenStation nextStation = selectedStation.next();
			nextStation.E += selectedStation.E;
			/*
			if (nextStation.E <= 0) {
				System.out.println("E 0 or bellow");
				System.exit(1);
			}
			*/
			nextStation.s = Math.min(nextStation.s, nextS);
			
			if (remainingEnergyNext > 0) {
				int nextK = nextStation.stop;
				//System.out.printf("next during performing [%s][%s]\n", b, nextK);
				currentSolution.e[b][nextK] += remainingEnergyNext;
				double remainingChargingTimeNext = remainingEnergyNext / instance.chargingRate;
				currentSolution.ct[b][nextK] += remainingChargingTimeNext;
				
				for (int j = k + 1; j <= nextK; j++) {
					//System.out.println(j);
					currentSolution.c[b][j] += -1 * remainingEnergyNext;
					//System.out.printf("original t in updating: t[%s][%s]=%s\n", b, j, currentSolution.arrivalTime[b][j]);
					currentSolution.arrivalTime[b][j] +=  -1 * remainingChargingTimeNext;
					/*
					if (j == nextK)
						System.out.printf("New next t in updating: t[%s][%s]=%s\n", b, j, currentSolution.arrivalTime[b][j]);
					*/
					double delayAvailable = instance.DTmax - currentSolution.getDeltaT(b, j);
					nextS = delayAvailable < nextS ? delayAvailable : nextS;
				}
				
				nextStation.s = nextS;
			}
		}
	}
	
	/**
	 * Return true if it is able to remove all buses in station i
	 * @param i removing station
	 * @param seed for all random coming computations
	 * @param iter current iterarion
	 * @return
	 */
	public boolean removeAllBuses(int i, int iter) {
		
		
		//rn.setSeed(seed);
		
		boolean feasible = true;
		
		int numOpenStopsPerStations = openStationsStructure.getNumOpenStopsPerStations(i);
		LOGGER.info("Number of open stops: " + numOpenStopsPerStations);
		if ( numOpenStopsPerStations > 0) {
			LinkedList<OpenStation> openStopsPerStation = openStationsStructure.openStopsPerStation.get(i);
			if (openStopsPerStation.size() == 0) {
				feasible = false;
			} else {
				LOGGER.info("Number of open stops: " + openStopsPerStation.size());
				
				 
 				int numBusesInI = openStopsPerStation.size();
 				OpenStation[] nopenStopsInI = openStopsPerStation.toArray(new OpenStation[numBusesInI]);
				//for (int k = 0; k < numBusesInI; k++) {
 				for (OpenStation selectedStop: nopenStopsInI) {
					//openStationsStructure.getListIteratorOpenStopsPerStation(k);
					/*
					System.out.println(k);
					feasible = feasible && removeOperators.get(k).isFeasibleStation(i, k);
					*/
					//OpenStation selectedStop = openStopsPerStation.getFirst();
					/*
					feasible = feasible && this.isFeasible2(selectedStop.bus, selectedStop.stop);			
					if (feasible) {
						this.removeStation();
					}
					*/
 					//System.out.println("Before...");
 					//this.completeCurrentSolution.printOpenStationStructure(selectedStop.bus);
 					int b = selectedStop.bus;
 					int k = selectedStop.stop;
 					openStations = openStationsStructure.openStopsPerBus.get(b);
 					
 					if (openStations.containsKey(k)) {
 						selectedStation = openStations.get(k);
 						LOGGER.info(String.format("Selected stop: [%s][%s] station %s", b, selectedStation.stop,
 								instance.paths[b][selectedStation.stop]));
						if (this.isFeasible2(selectedStation)) {
							this.removeStation();
						} else {
							return false;
							//System.out.println("WTF?");
							//System.exit(0);
						}
 					} else {
 						System.out.println("WTF?");
						System.exit(0);
 					}
					//System.out.println("After...");
					//this.completeCurrentSolution.printOpenStationStructure(selectedStop.bus);
					
					/*
					if (iter > 2500 && !completeCurrentSolution.checkFeasibility(instance, "partial")) {
						LOGGER.info("Unfeasible after a specific stop remove");
						System.exit(0);
					}
					*/
					
					
					
				}
			}
		} //else {
			//System.out.printf("Station %s is not open\n", i);
			//feasible = false;
		//}
		return feasible;
	}
	
//	/**
//	 * Return true if it is able to remove all buses in station i
//	 * @param removeOperator
//	 * @param i
//	 * @return
//	 */
//	public boolean checkAllBuses(int i, int seed) {
//		
//		rn.setSeed(seed);
//		
//		boolean feasible = true;
//		int numRemovals = 0;
//		
//		int numOpenStopsPerStations = openStationsStructure.getNumOpenStopsPerStations(i);
//		LOGGER.info("Number of open stops: " + numOpenStopsPerStations);
//		if ( numOpenStopsPerStations > 0) {
//			LinkedList<OpenStation> openStopsPerStation = openStationsStructure.openStopsPerStation.get(i);
//			if (openStopsPerStation.size() == 0) {
//				feasible = false;
//			} else {
//				LOGGER.info("Number of open stops: " + openStopsPerStation.size());
//				
//				 
// 				int numBusesInI = openStopsPerStation.size();
// 				OpenStation[] nopenStopsInI = openStopsPerStation.toArray(new OpenStation[numBusesInI]);
//				//for (int k = 0; k < numBusesInI; k++) {
// 				for (OpenStation selectedStop: nopenStopsInI) {
//					//openStationsStructure.getListIteratorOpenStopsPerStation(k);
//					/*
//					System.out.println(k);
//					feasible = feasible && removeOperators.get(k).isFeasibleStation(i, k);
//					*/
//					//OpenStation selectedStop = openStopsPerStation.getFirst();
//					feasible = feasible && this.isFeasible2(selectedStop.bus, selectedStop.stop);
//					/*
//					if (feasible) {
//						this.removeStation();
//					}
//					*/
// 					//System.out.println("Before...");
// 					//this.completeCurrentSolution.printOpenStationStructure(selectedStop.bus);
//					//System.out.println("After...");
//					//this.completeCurrentSolution.printOpenStationStructure(selectedStop.bus);
//					/*
//					if (iteration == 81408   && selectedStop.stop==195) { //!completeCurrentSolution.checkFeasibility(instance)) {
//						System.exit(0);
//					}
//					*/
//					
//				}
//			}
//		} // If there is no open stops with the station in primary we still want to check in backup and remove there 
//		//else {
//			//System.out.printf("Station %s is not open\n", i);
//			//feasible = false;
//		//}
//		return feasible;
//	}
}
