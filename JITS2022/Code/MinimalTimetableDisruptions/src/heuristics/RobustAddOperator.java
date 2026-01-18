package heuristics;

import java.util.HashMap;
import java.util.Random;

import core.InstanceMTD;
import utils.ToolsMTD;

public class RobustAddOperator extends AddOperator {

	RobustCompleteSolution completeCurrentSolution;
	AddOperator backupAddOperator;
	
	
	public RobustAddOperator(RobustCompleteSolution currentSolution, InstanceMTD instance, Random rn) {
		super(currentSolution, instance, rn);
		this.completeCurrentSolution = currentSolution;
		backupAddOperator = new AddOperator(currentSolution.robustRoute, instance, rn);
	}
	
	public void addStation(int i) {
		/*
		System.out.println("Adding primary station first --------------\n");
		super.resetOperator(completeCurrentSolution);
		isAddingPrimary = true;
		super.addStation(i);

		System.out.println("Adding backup station now -----------------\n");
		super.resetOperator(completeCurrentSolution.robustRoute);
		isAddingPrimary = false;
		super.addStation(i);
		*/
		
		if (openStationsStructure.getNumOpenStopsPerStations(i) == 0) {
			for (Stop stop: instance.stopsPerStation[i]) {

				LOGGER.info(String.format("Trying to addi: [%s][%s]", stop.bus, stop.stop));
				
				nextAddedEnergy = 0;
				nextAddedCtime = 0;
				newE = 0;
				prevAddednergy = 0;
				prevAddedCtime = 0;
				
				backupAddOperator.nextAddedEnergy = 0;
				backupAddOperator.nextAddedCtime = 0;
				backupAddOperator.newE = 0;
				backupAddOperator.prevAddednergy = 0;
				backupAddOperator.prevAddedCtime = 0;
				
				addSpecificStop(stop, i);			
			}
		}
		
		if (backupAddOperator.openStationsStructure.getNumOpenStopsPerStations(i) == 0) {
			for (Stop stop: instance.stopsPerStation[i]) {

				LOGGER.info(String.format("b Trying to addi: [%s][%s]", stop.bus, stop.stop));
				
				nextAddedEnergy = 0;
				nextAddedCtime = 0;
				newE = 0;
				prevAddednergy = 0;
				prevAddedCtime = 0;
				
				backupAddOperator.nextAddedEnergy = 0;
				backupAddOperator.nextAddedCtime = 0;
				backupAddOperator.newE = 0;
				backupAddOperator.prevAddednergy = 0;
				backupAddOperator.prevAddedCtime = 0;
				
				addSpecificStopBackup(stop, i);			
			}
		}
		
		
		
	}
	
	public void resetOperator(CompleteSolution currentSolution) {
		super.resetOperator(currentSolution);
		this.completeCurrentSolution = (RobustCompleteSolution) currentSolution;
		this.backupAddOperator.resetOperator(completeCurrentSolution.robustRoute);
	}
	
	public boolean checkRobustFeasibility(Stop stop, int b, int i, OpenStation nextOpenStation, OpenStation previousStation, 
			boolean isAddingPrimary, int mirrorStop) {
		
		boolean isFeasible = true;
		/*
		boolean isFeasible = super.checkFeasibility(stop, b, i, nextOpenStation, previousStation);
		
		if (!isFeasible) {
			return false;
		}
		*/
		
		int k = stop.stop;
		
		HeuristicsVariablesSet primaryVars = completeCurrentSolution.normalVars;
		HeuristicsVariablesSet backupVars = completeCurrentSolution.robustRoute.normalVars;
		
		
		// Path constraint
		if (isAddingPrimary && backupVars.xBStop[b][k]) {
			isFeasible = false;
		} else if (!isAddingPrimary && primaryVars.xBStop[b][k]) {
			isFeasible = false;
		}
		
		
		//It has a next
		if (nextOpenStation != null) {
			int nextK = nextOpenStation.stop;
			for (int ka = k + 1; ka <= nextK - 1; ka++) {
				if (!isAddingPrimary && primaryVars.xBStop[b][ka]) {
					double newBackupArrivalEnergy = ToolsMTD.round(backupVars.c[b][ka] + backupAddOperator.nextAddedEnergy);
					double newPrimaryArrivalEnergy = ToolsMTD.round(primaryVars.c[b][ka] + nextAddedEnergy);
					boolean canGotoBackup = newPrimaryArrivalEnergy >= newBackupArrivalEnergy;
					isFeasible = isFeasible && canGotoBackup;
				}
				
				if (isAddingPrimary && (backupVars.xBStop[b][ka] || mirrorStop == ka)) {
					double newPrimaryArrivalEnergy = ToolsMTD.round(primaryVars.c[b][ka + 1] + nextAddedEnergy);
					double newBackupArrivalEnergy = ToolsMTD.round(backupVars.c[b][ka + 1] + backupAddOperator.nextAddedEnergy);
					boolean canGotoPrimary = newBackupArrivalEnergy >= newPrimaryArrivalEnergy;
					isFeasible = isFeasible && canGotoPrimary;
				}
			}
			
			if (isAddingPrimary) {
				double newPrimaryArrivalTime = ToolsMTD.round(primaryVars.arrivalTime[b][nextK] + nextAddedCtime);
				double newBackupArrivalTime = ToolsMTD.round(backupVars.arrivalTime[b][nextK] + backupAddOperator.nextAddedCtime);
				boolean canGoToPrimaryT = newPrimaryArrivalTime <= newBackupArrivalTime;
				isFeasible = isFeasible && canGoToPrimaryT;
			} else {
				
			}
				
		}
		
		
		
		if (previousStation != null) {
			OpenStation prevStop = previousStation;
			int prevK = prevStop.stop;
			
			for (int m = prevK + 1; m <= k - 1; m++) {
				if (isAddingPrimary) {	
					if (backupVars.xBStop[b][m]) {
						double newPrimaryArrivalTime = ToolsMTD.round(primaryVars.arrivalTime[b][m + 1] - prevAddedCtime);
						double newBackupArrivalTime = ToolsMTD.round(backupVars.arrivalTime[b][m + 1] - backupAddOperator.prevAddedCtime);
						boolean canGoToPrimaryT = newBackupArrivalTime <= newPrimaryArrivalTime;
						isFeasible = isFeasible && canGoToPrimaryT;
					}
				} else {
					if (primaryVars.xBStop[b][m] || mirrorStop == m) {
						double newBackupArrivalTime = ToolsMTD.round(primaryVars.arrivalTime[b][m] - backupAddOperator.prevAddedCtime);
						double newPrimaryArrivalTime = ToolsMTD.round(backupVars.arrivalTime[b][m] - prevAddedCtime);
						boolean canGoToBackupT = newPrimaryArrivalTime <= newBackupArrivalTime;
						isFeasible = isFeasible && canGoToBackupT;
					}
				}
			}	
		}
		
		
		
		
		if (isAddingPrimary) {	
			double newPrimaryArrivalEnergy = ToolsMTD.round(primaryVars.c[b][k] - prevAddednergy);
			double newBackupArrivalEnergy = ToolsMTD.round(backupVars.c[b][k] - backupAddOperator.prevAddednergy);
			boolean canGoToBackup = newPrimaryArrivalEnergy >= newBackupArrivalEnergy;
			isFeasible = isFeasible && canGoToBackup;	
			double newPrimaryArrivalTime = ToolsMTD.round(primaryVars.arrivalTime[b][k] - prevAddedCtime);
			double newBackupArrivalTime = ToolsMTD.round(backupVars.arrivalTime[b][k] - backupAddOperator.prevAddedCtime);
			boolean canGoToBackupT = newPrimaryArrivalTime <= newBackupArrivalTime;
			isFeasible = isFeasible && canGoToBackupT;	
		} else {		
			try {
				double newBackupArrivalEnergyToK = ToolsMTD.round(backupVars.c[b][k + 1] + backupAddOperator.nextAddedEnergy);
				double newPrimaryArrivalEnergyToK = ToolsMTD.round(primaryVars.c[b][k + 1] + nextAddedEnergy);
				boolean canGoToPrimaryFromK = newBackupArrivalEnergyToK >= newPrimaryArrivalEnergyToK;
				isFeasible = isFeasible && canGoToPrimaryFromK;
				
				double newBackupArrivalTimeToK = ToolsMTD.round(backupVars.arrivalTime[b][k + 1] + backupAddOperator.nextAddedCtime);
				double oldPrimaryArrivalTimeToK = ToolsMTD.round(primaryVars.arrivalTime[b][k + 1] + nextAddedCtime);
				boolean canGoToPrimaryTimeFromK = newBackupArrivalTimeToK <= oldPrimaryArrivalTimeToK;
				isFeasible = isFeasible && canGoToPrimaryTimeFromK;
			} catch (ArrayIndexOutOfBoundsException e) {
				//Nothing to do
				//There is not need to check the try code
			}
			
			if (previousStation != null) {
				OpenStation prevStop = previousStation;
				int prevK = prevStop.stop;
				
				double newBackupArrivalEnergyToPrevK = ToolsMTD.round(backupVars.c[b][prevK + 1] - backupAddOperator.prevAddednergy);
				double newPrimaryArrivalEnergyToPrevK = ToolsMTD.round(primaryVars.c[b][prevK + 1] - prevAddednergy);
				boolean canGoToPrimaryFromPrevK = newBackupArrivalEnergyToPrevK >= newPrimaryArrivalEnergyToPrevK;
				isFeasible = isFeasible && canGoToPrimaryFromPrevK;
				//System.out.printf("IS CHECKING! stop %s [%s][%s] - %s\n", prevK + 1, b, k, canGoToPrimaryFromPrevK);
				//System.out.printf("%s >= %s\n", newBackupArrivalEnergyToPrevK, oldPrimaryArrivalEnergyToPrevK);
			}
		}
			
		
		return isFeasible;
	}
	
	public void addSpecificStop(Stop stop, int i) {
		
		int b = stop.bus;
		openStations = openStationsStructure.openStopsPerBus.get(b);
		OpenStation nextStation = openStations.get(openStationsStructure.headsIdPerBus.get(b));//openStationsStructure.getHeadInOpenStopPerBus(b);
		boolean willBeLast = true;
		//int prevK = nextStation.stop;
		LOGGER.info("head: " + nextStation.stop);
		OpenStation lastStop = null;
		while (nextStation != null) {	
			OpenStation previousStation = nextStation.previous();
			/*
			if (previousStation != null) {
				LOGGER.info("prev: " + previousStation.stop);
			}
			LOGGER.info("next: " + nextStation.stop);
			*/
			LOGGER.info(String.format("%s > %s\n", nextStation.stop, stop.stop));
			if (nextStation.stop > stop.stop) {
				LOGGER.info("ah?");
				if (previousStation != null) {
					OpenStation previousBackup = previousStation.nextBackup();
					if (stop.stop <= previousBackup.stop) {
						return;
					}
					
				}
				int stationsBetweenAddingAndNext = nextStation.stop - stop.stop - 1;
				if (stationsBetweenAddingAndNext == 0) { 
					return;
				}
				int newBackupStop = stop.stop + (rn.nextInt(stationsBetweenAddingAndNext) + 1);
				Stop bStop = new Stop(b, newBackupStop, -1);
				int bStation = instance.paths[b][newBackupStop];
				LOGGER.info(String.format("k:%s, bk:%s\n", stop.stop, newBackupStop));
				OpenStation previousNewBackup = null;
				OpenStation nextNewBackup = null;
				if (previousStation != null) {
					previousNewBackup = previousStation.nextBackup();
					nextNewBackup = previousNewBackup.next();
				} else {
					nextNewBackup = nextStation.nextBackup();
				} 
				if (checkFeasibility(stop, b, i, nextStation, previousStation) && 
						backupAddOperator.checkFeasibility(bStop, b, bStation, nextNewBackup, previousNewBackup) &&
						checkRobustFeasibility(stop, b, i, nextStation, previousStation, true, newBackupStop) &&
						checkRobustFeasibility(bStop, b, bStation, nextNewBackup, previousNewBackup, false, stop.stop)) {
					updateAfterAdd(stop, b, i, nextStation, previousStation);
					backupAddOperator.updateAfterAdd(bStop, b, bStation, nextNewBackup, previousNewBackup);
					linkNewPrimaryBackup(b, stop.stop, newBackupStop);
						
					/*
					if (!completeCurrentSolution.checkFeasibility(instance, "partial.log")) {
						LOGGER.info("Exiting after specific adding 1");
						//completeCurrentSolution.printOpenStationStructure(18);
						//backupAddOperator.openStationsStructure.print(18);
						System.out.println(completeCurrentSolution.robustRoute.openStationsStructure == backupAddOperator.openStationsStructure);
						System.exit(0);
					};
					*/
					
						
				}
				willBeLast = false;
				break;			
			}
			if (!nextStation.hasNext()) {
				lastStop = nextStation;
			}
			nextStation = nextStation.next();	
			//LOGGER.info("next: " + nextStation.stop);
		}
		// Last or first in a current single element list
		if (willBeLast) {
			nextStation = lastStop;
			
			//nextAddedEnergy = 0;
			//nextAddedCtime = 0;
			//openStations.previous();
			// The last open stop
			OpenStation previousStation = null;
			OpenStation nextStationToPass = null;
			
			int stationsBetweenAddingAndNext = -1;// nextStation.stop - stop.stop - 1;
			
			OpenStation previousNewBackup = null;
			OpenStation nextNewBackup = null;
			
			if (nextStation.stop < stop.stop) {
				LOGGER.info(String.format("Adding last stop: [%s][%s]\n", b, stop.stop));
				previousStation = nextStation;
				stationsBetweenAddingAndNext = instance.paths[b].length - stop.stop - 1;
								
				previousNewBackup = previousStation.nextBackup();
				nextNewBackup = previousNewBackup.next();
				
				if (stop.stop <= previousNewBackup.stop) {
					return;
				}
					
				
			} else {
				System.out.println("Error");
				System.exit(0);
			}
			
			
			if (stationsBetweenAddingAndNext == 0) { 
				return;
			}
			
			int newBackupStop = stop.stop + (rn.nextInt(stationsBetweenAddingAndNext) + 1);
			Stop bStop = new Stop(b, newBackupStop, -1);
			int bStation = instance.paths[b][newBackupStop];
			
			LOGGER.info(String.format("k:%s, bk:%s\n", stop.stop, newBackupStop));			
			
			if (checkFeasibility(stop, b, i, nextStationToPass, previousStation) && 
					backupAddOperator.checkFeasibility(bStop, b, bStation, nextNewBackup, previousNewBackup) &&
					checkRobustFeasibility(stop, b, i, nextStationToPass, previousStation, true, newBackupStop) &&
					checkRobustFeasibility(bStop, b, bStation, nextNewBackup, previousNewBackup, false, stop.stop)) {
				updateAfterAdd(stop, b, i, nextStationToPass, previousStation);
				backupAddOperator.updateAfterAdd(bStop, b, bStation, nextNewBackup, previousNewBackup);
				linkNewPrimaryBackup(b, stop.stop, newBackupStop);
				
				
				/*
				if (!completeCurrentSolution.checkFeasibility(instance, "partial.log")) {
					LOGGER.info("Exiting after specific adding 2");
					completeCurrentSolution.printOpenStationStructure(33);
					//backupAddOperator.openStationsStructure.print(39);
					System.out.println(completeCurrentSolution.robustRoute.openStationsStructure == backupAddOperator.openStationsStructure);
					System.exit(0);
				};
				*/
				
								
			}
			
		}
	}
	
	
public void addSpecificStopBackup(Stop stop, int i) {
		
		int b = stop.bus;
		openStations = backupAddOperator.openStationsStructure.openStopsPerBus.get(b);
		//LOGGER.info(String.valueOf(openStations.size()));
		OpenStation nextStation = openStations.get(backupAddOperator.openStationsStructure.headsIdPerBus.get(b));//openStationsStructure.getHeadInOpenStopPerBus(b);
		boolean willBeLast = true;
		OpenStation lastStop = null;
		//LOGGER.info("next: " + nextStation.stop);
		while (nextStation != null) {	
			OpenStation previousStation = nextStation.previous();
			/*
			if (previousStation != null) {
				LOGGER.info("prev: " + previousStation.stop);
			}
			LOGGER.info("next: " + nextStation.stop);
			*/
			//LOGGER.info("next: " + nextStation.stop);
			
			if (nextStation.stop > stop.stop) {
				
				OpenStation prymaryOfNext = nextStation.previousPrimary();
				if (stop.stop >= prymaryOfNext.stop) {
					return;
				}
				
				int initialStop = nextStation.hasPrevious() ? nextStation.previous().stop : 0;
				
				int stationsBetweenAddingAndPrev = stop.stop - initialStop - 1;
				if (stationsBetweenAddingAndPrev <= 0) { 
					return;
				}
				int newPrimarySpot = initialStop + (rn.nextInt(stationsBetweenAddingAndPrev) + 1);
				Stop pStop = new Stop(b, newPrimarySpot, -1);
				int pStation = instance.paths[b][newPrimarySpot];
				LOGGER.info(String.format("k:%s, bk:%s\n", newPrimarySpot, stop.stop));
				OpenStation previousNewPrimary = null;
				OpenStation nextNewPrimary = null;
				if (previousStation != null) {
					previousNewPrimary = previousStation.previousPrimary();
					nextNewPrimary = previousNewPrimary.next();
				} else {
					nextNewPrimary = nextStation.previousPrimary();
				} 
				if (checkFeasibility(pStop, b, pStation, nextNewPrimary, previousNewPrimary) && 
						backupAddOperator.checkFeasibility(stop, b, i, nextStation, previousStation) &&
						checkRobustFeasibility(pStop, b, pStation, nextNewPrimary, previousNewPrimary, true, stop.stop) &&
						checkRobustFeasibility(stop, b, i, nextStation, previousStation, false, newPrimarySpot)) {
					updateAfterAdd(pStop, b, pStation, nextNewPrimary, previousNewPrimary);
					backupAddOperator.updateAfterAdd(stop, b, i, nextStation, previousStation);
					linkNewPrimaryBackup(b, newPrimarySpot, stop.stop);
						
					/*
					if (!completeCurrentSolution.checkFeasibility(instance, "partial.log")) {
						LOGGER.info("Exiting after specific adding 1");
						//completeCurrentSolution.printOpenStationStructure(18);
						//backupAddOperator.openStationsStructure.print(18);
						System.out.println(completeCurrentSolution.robustRoute.openStationsStructure == backupAddOperator.openStationsStructure);
						System.exit(0);
					};
					*/
					
						
				}
				willBeLast = false;
				break;			
			}
			if (!nextStation.hasNext()) {
				lastStop = nextStation;
			}
			nextStation = nextStation.next();	
			//LOGGER.info("next: " + nextStation.stop);
		}
		// Last or first in a current single element list
		if (willBeLast) {
			nextStation = lastStop;
			
			//nextAddedEnergy = 0;
			//nextAddedCtime = 0;
			//openStations.previous();
			// The last open stop
			OpenStation previousStation = null;
			
			int stationsBetweenAddingAndNext = -1;// nextStation.stop - stop.stop - 1;
			
			OpenStation previousNewPrimary = null;
			OpenStation nextNewPrimary = null;
			
			//completeCurrentSolution.printOpenStationStructure(0);
			
			if (nextStation.stop < stop.stop) {
				LOGGER.info(String.format("Adding last stop: [%s][%s]\n", b, stop.stop));
				previousStation = nextStation;
				stationsBetweenAddingAndNext = instance.paths[b].length - stop.stop - 1;
								
				previousNewPrimary = previousStation.previousPrimary();
				nextNewPrimary = previousNewPrimary.next();
					
				
			} else {
				System.out.println("Error");
				System.exit(0);
			}
			
			
			if (stationsBetweenAddingAndNext <= 0) { 
				return;
			}
			
			int initialStop = previousStation.stop;
			
			int stationsBetweenAddingAndPrev = stop.stop - initialStop - 1;
			if (stationsBetweenAddingAndPrev == 0) { 
				return;
			}
			int newPrimarySpot = initialStop + (rn.nextInt(stationsBetweenAddingAndPrev) + 1);
			Stop pStop = new Stop(b, newPrimarySpot, -1);
			int pStation = instance.paths[b][newPrimarySpot];
			
			LOGGER.info(String.format("k:%s, bk:%s\n", newPrimarySpot, stop.stop));			
			
			if (checkFeasibility(pStop, b, pStation, nextNewPrimary, previousNewPrimary) && 
					backupAddOperator.checkFeasibility(stop, b, i, null, previousStation) &&
					checkRobustFeasibility(pStop, b, pStation, nextNewPrimary, previousNewPrimary, true, stop.stop) &&
					checkRobustFeasibility(stop, b, i, null, previousStation, false, newPrimarySpot)) {
				updateAfterAdd(pStop, b, pStation, nextNewPrimary, previousNewPrimary);
				backupAddOperator.updateAfterAdd(stop, b, i, null, previousStation);
				linkNewPrimaryBackup(b, newPrimarySpot, stop.stop);
				
				
				/*
				if (!completeCurrentSolution.checkFeasibility(instance, "partial.log")) {
					LOGGER.info("Exiting after specific adding 2");
					completeCurrentSolution.printOpenStationStructure(33);
					//backupAddOperator.openStationsStructure.print(39);
					System.out.println(completeCurrentSolution.robustRoute.openStationsStructure == backupAddOperator.openStationsStructure);
					System.exit(0);
				};
				*/
				
								
			}
			
		}
	}
	
	public void linkNewPrimaryBackup(int b, int pk, int bk) {
		HashMap<Integer, OpenStation> primaryOpenStations = openStationsStructure.openStopsPerBus.get(b);
		OpenStation primary = primaryOpenStations.get(pk);
		HashMap<Integer, OpenStation> backupOpenStations = backupAddOperator.openStationsStructure.openStopsPerBus.get(b);
		OpenStation backup = backupOpenStations.get(bk);
		primary.setNextBackup(backup);
		backup.setPreviousPrimary(primary);
		
		
	}

}
