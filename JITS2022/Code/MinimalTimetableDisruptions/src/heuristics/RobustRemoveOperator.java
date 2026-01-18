package heuristics;

import java.util.LinkedList;
import java.util.Random;

import core.InstanceMTD;
import utils.ToolsMTD;

/**
 * @author cedaloaiza
 *
 */
public class RobustRemoveOperator extends RemoveOperator {

	RobustCompleteSolution completeCurrentSolution;
	RemoveOperator backupRemoveOperator;

	
	public RobustRemoveOperator(RobustCompleteSolution currentSolution, InstanceMTD instance, Random rn) {
		super(currentSolution, instance, rn);
		this.completeCurrentSolution = currentSolution;
		backupRemoveOperator = new RemoveOperator(currentSolution.robustRoute, instance, rn);
	}
	
//	public boolean checkAllBuses(int i, int iteration) {
//		boolean feasible = true;
//		System.out.println("Removing primary station first");
//		super.resetOperator(completeCurrentSolution);
//		isRemovingPrimary = true;
//		feasible = super.checkAllBuses(i, iteration);
//		super.resetOperator(completeCurrentSolution.robustRoute);
//		isRemovingPrimary = false;
//		feasible = feasible && super.checkAllBuses(i, iteration);
//		//System.exit(0);
//		return feasible;
//	}
	
	public boolean removeAllBuses(int i, int iteration) {
			
			boolean feasible = true;
		
			/*
			System.out.println("Removing primary station first ------------------------");
			isRemovingPrimary = true;
			super.resetOperator(completeCurrentSolution);
			isFeasible = super.removeAllBuses(i, seed, iteration);

			System.out.println("Removing backup station now ------------------------");
			isRemovingPrimary = false;
			super.resetOperator(completeCurrentSolution.robustRoute);
			isFeasible = isFeasible & super.removeAllBuses(i, seed, iteration);

			*/
			
			
			LinkedList<OpenStation> openStopsPerStation = openStationsStructure.openStopsPerStation.get(i);
			if (openStopsPerStation == null || openStopsPerStation.size() == 0) {
				feasible = false;
			} else {
				LOGGER.info("Number of open stops: " + openStopsPerStation.size());
				
				 
 				int numBusesInI = openStopsPerStation.size();
 				OpenStation[] nopenStopsInI = openStopsPerStation.toArray(new OpenStation[numBusesInI]);
 				for (OpenStation selectedStop: nopenStopsInI) {

 					int b = selectedStop.bus;
 					int k = selectedStop.stop;
 					openStations = openStationsStructure.openStopsPerBus.get(b);
 					
 					if (openStations.containsKey(k)) {
 						selectedStation = openStations.get(k);
 						LOGGER.info(String.format("Selected primary stop: [%s][%s] station %s", b, selectedStation.stop,
 								instance.paths[b][selectedStation.stop]));
 						if (selectedStation.hasBackup()) {
	 						OpenStation backupSelectedStop = selectedStation.nextBackup();
	 						LOGGER.info(String.format("Selected corresponding backup stop: [%s][%s] station %s", b, backupSelectedStop.stop,
	 								instance.paths[b][backupSelectedStop.stop]));
							if (this.isFeasible2(selectedStation) && backupRemoveOperator.isFeasible2(backupSelectedStop)
									&& checkRobustFeasibility(selectedStop, true) && checkRobustFeasibility(backupSelectedStop, false)) {
								this.removeStation();
								backupRemoveOperator.removeStation();
							} else {
								return false;
							}					
 						} else {
							System.out.println("Every primary should have a backup");
							System.exit(0);
						}
 					} else {
 						System.out.println("WTF?");
						System.exit(0);
 					}				
				}
			}
			
			CompleteSolution backupSolution = completeCurrentSolution.robustRoute;
			LinkedList<OpenStation> backupOpenStopsPerStation = backupSolution.openStationsStructure.openStopsPerStation.get(i);
			if (backupOpenStopsPerStation == null || backupOpenStopsPerStation.size() == 0) {
				feasible = false;
			} else {
				LOGGER.info("Number of open stops: " + backupOpenStopsPerStation.size());
				
				 
 				int numBusesInI = backupOpenStopsPerStation.size();
 				OpenStation[] nopenStopsInI = backupOpenStopsPerStation.toArray(new OpenStation[numBusesInI]);
 				for (OpenStation selectedStop: nopenStopsInI) {

 					int b = selectedStop.bus;
 					int k = selectedStop.stop;
 					openStations = backupSolution.openStationsStructure.openStopsPerBus.get(b);
 					
 					if (openStations.containsKey(k)) {
 						selectedStation = openStations.get(k);
 						LOGGER.info(String.format("Selected backup stop: [%s][%s] station %s", b, selectedStation.stop,
 								instance.paths[b][selectedStation.stop]));
 						if (selectedStation.hasPrimary()) {
	 						OpenStation primarySelectedStop = selectedStation.previousPrimary();
	 						LOGGER.info(String.format("Corresponding primary: [%s][%s] station %s", primarySelectedStop.bus, 
	 								primarySelectedStop.stop, "x"));
							if (this.isFeasible2(primarySelectedStop) && backupRemoveOperator.isFeasible2(selectedStop)
									&& checkRobustFeasibility(primarySelectedStop, true) && checkRobustFeasibility(selectedStop, false)) {
								//completeCurrentSolution.stopsPerStationToString();
								this.removeStation();
								backupRemoveOperator.removeStation();
							} else {
								return false;
							}					
 						} else {
							System.out.println("Every backup should have a primary");
							System.exit(0);
						}
 					} else {
 						System.out.println("WTF?");
						System.exit(0);
 					}				
				}
			}
			

		return feasible;
	}
	
	public void resetOperator(CompleteSolution currentSolution) {
		super.resetOperator(currentSolution);
		this.completeCurrentSolution = (RobustCompleteSolution) currentSolution;
		this.backupRemoveOperator.resetOperator(completeCurrentSolution.robustRoute);
	}
	
	
	public boolean checkRobustFeasibility(OpenStation selectedStation, boolean isRemovingPrimary) {
		int b = selectedStation.bus;
		boolean isFeasible = true;
		//boolean isFeasible = super.checkFeasibility(selectedStation);
		HeuristicsVariablesSet primaryVars = completeCurrentSolution.normalVars;
		HeuristicsVariablesSet backupVars = completeCurrentSolution.robustRoute.normalVars;
		
		if (isRemovingPrimary) {
			//************Removing primary*********
			if (selectedStation.hasNext()) {
				OpenStation nextStation = selectedStation.next();
				double newPrimaryNextArrivalEnergy = ToolsMTD.round(primaryVars.c[b][nextStation.stop] - remainingEnergyNext);
				double newBackupNextArrivalEnergy = ToolsMTD.round(backupVars.c[b][nextStation.stop] - 
						backupRemoveOperator.remainingEnergyNext);
				boolean canGotoBackup = newPrimaryNextArrivalEnergy >= newBackupNextArrivalEnergy;
				isFeasible = isFeasible && canGotoBackup;
			}
		}	
		/**
		 * TODO
		 * We could use the backup/primary structure to not walk across all these stations without purpose
		 */
		if (selectedStation.hasPrevious()) {
			OpenStation prevStation = selectedStation.previous();
			//System.out.println("prev: " + prevStation.stop);
			//System.out.println("next: " + selectedStation.stop);
			for (int k = prevStation.stop + 1; k <= selectedStation.stop - 1; k++) {
				if (isRemovingPrimary) {
					if (backupVars.xBStop[b][k]) {
						double newPrimaryArrivalEnergy = ToolsMTD.round(primaryVars.c[b][k + 1] + minEnergyNeeded +
								remainingEnergyPrev);
						double newBackupArrivalEnergy = ToolsMTD.round(backupVars.c[b][k + 1] + backupRemoveOperator.minEnergyNeeded +
								backupRemoveOperator.remainingEnergyPrev);
						boolean canBacktoPrimary = newBackupArrivalEnergy >= newPrimaryArrivalEnergy;
						isFeasible = isFeasible && canBacktoPrimary;
					}
				}
				//************END OF: Removing primary*********
				//************Removing backup*********
				// Not needed since this primary is not existing anymore
				/*
				if (!isRemovingPrimary) {
					if (primaryVars.xBStop[b][k]) {
						double newBackupArrivalEnergy = ToolsMTD.round(backupVars.c[b][k] + minEnergyNeeded +
								remainingEnergyPrev);
						double oldPrimaryArrivalEnergy = ToolsMTD.round(primaryVars.c[b][k]);
						boolean canGotoBackup = oldPrimaryArrivalEnergy >= newBackupArrivalEnergy;
						isFeasible = isFeasible && canGotoBackup;
					}
				}
				*/
			}
			if (!isRemovingPrimary) {
				double newBackupArrivalTime = ToolsMTD.round(backupVars.arrivalTime[b][prevStation.stop + 1] + 
						backupRemoveOperator.minNeededChargingTime + backupRemoveOperator.remainingChargingTimePrev);
				double newPrimaryArrivalTime = ToolsMTD.round(primaryVars.arrivalTime[b][prevStation.stop + 1] +
						minNeededChargingTime + remainingChargingTimePrev);
				boolean canGotoPrimary = newBackupArrivalTime <= newPrimaryArrivalTime;
				isFeasible = isFeasible && canGotoPrimary;
			}
		}
		
		if (selectedStation.hasNext()) {
			OpenStation nextStation = selectedStation.next();
			double remainingChargingTimeNextPrimary = remainingEnergyNext / instance.chargingRate;
			double remainingChargingTimeNextBackup = backupRemoveOperator.remainingEnergyNext / instance.chargingRate;
			for (int k = selectedStation.stop + 1; k <= nextStation.stop - 1; k++) {
				// Not needed since this backup is not existing anymore
				/*
				if (isRemovingPrimary && backupVars.xBStop[b][k]) {
					double newPrimaryArrivalTime = ToolsMTD.round(primaryVars.arrivalTime[b][k + 1] -
							remainingChargingTimeNext);
					double oldBackupArrivalTime = ToolsMTD.round(backupVars.arrivalTime[b][k + 1]);
					boolean canGotoPrimary = oldBackupArrivalTime <= newPrimaryArrivalTime;
					isFeasible = isFeasible && canGotoPrimary;
				} else 
				*/
				if (!isRemovingPrimary && primaryVars.xBStop[b][k]) {
					double newBackupArrivalTime = ToolsMTD.round(backupVars.arrivalTime[b][k] -
							remainingChargingTimeNextBackup);
					double newPrimaryArrivalTime = ToolsMTD.round(primaryVars.arrivalTime[b][k] -
							remainingChargingTimeNextPrimary);
					boolean canGotoBackup = newPrimaryArrivalTime <= newBackupArrivalTime;
					isFeasible = isFeasible && canGotoBackup;
				}
			}
		}
		
		return isFeasible;
	}
	

}
