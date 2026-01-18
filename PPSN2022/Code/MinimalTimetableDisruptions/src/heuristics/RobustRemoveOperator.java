package heuristics;

import java.util.Random;

import core.InstanceMTD;
import utils.ToolsMTD;

/**
 * @author cedaloaiza
 *
 */
public class RobustRemoveOperator extends RemoveOperator {

	RobustCompleteSolution completeCurrentSolution;
	/**
	 * Whether we are removing a primary station or a backup one
	 */
	boolean isRemovingPrimary = true;
	
	public RobustRemoveOperator(RobustCompleteSolution currentSolution, InstanceMTD instance, Random rn) {
		super(currentSolution, instance, rn);
		this.completeCurrentSolution = currentSolution;
	}
	
	public boolean checkAllBuses(int i, int iteration) {
		boolean feasible = true;
		System.out.println("Removing primary station first");
		super.resetOperator(completeCurrentSolution);
		isRemovingPrimary = true;
		feasible = super.checkAllBuses(i, iteration);
		super.resetOperator(completeCurrentSolution.robustRoute);
		isRemovingPrimary = false;
		feasible = feasible && super.checkAllBuses(i, iteration);
		//System.exit(0);
		return feasible;
	}
	
	public boolean removeAllBuses(int i, int seed, int iteration) {

			System.out.println("Removing primary station first ------------------------");
			isRemovingPrimary = true;
			super.resetOperator(completeCurrentSolution);
			super.removeAllBuses(i, seed, iteration);
			/*
			if (iteration > 29000 && !completeCurrentSolution.checkFeasibility(instance, "partial")) {
				System.exit(0);
			}
			*/
			System.out.println("Removing backup station now ------------------------");
			isRemovingPrimary = false;
			super.resetOperator(completeCurrentSolution.robustRoute);
			super.removeAllBuses(i, seed, iteration);
			/*
			if (iteration > 29000 && !completeCurrentSolution.checkFeasibility(instance, "partial")) {
				System.exit(0);
			}
			*/
		//System.exit(0);
		return false;
	}
	
	public void resetOperator(CompleteSolution currentSolution) {
		super.resetOperator(currentSolution);
		this.completeCurrentSolution = (RobustCompleteSolution) currentSolution;
	}
	
	
	public boolean checkFeasibility(int b, OpenStation selectedStation) {
		boolean isFeasible = super.checkFeasibility(b, selectedStation);
		HeuristicsVariablesSet primaryVars = completeCurrentSolution.normalVars;
		HeuristicsVariablesSet backupVars = completeCurrentSolution.robustRoute.normalVars;
		
		if (isRemovingPrimary) {
			//************Removing primary*********
			if (openStations.hasNext()) {
				OpenStation nextStation = openStations.next();
				double newPrimaryNextArrivalEnergy = ToolsMTD.round(primaryVars.c[b][nextStation.stop] - remainingEnergyNext);
				double oldBackupNextArrivalEnergy = ToolsMTD.round(backupVars.c[b][nextStation.stop]);
				boolean canGotoBackup = newPrimaryNextArrivalEnergy >= oldBackupNextArrivalEnergy;
				isFeasible = isFeasible && canGotoBackup;
				openStations.previous();
			}
		}	
		/**
		 * TODO
		 * We could use the backup/primary structure to not walk across all these stations without purpose
		 */
		openStations.previous();
		if (openStations.hasPrevious()) {
			OpenStation prevStation = openStations.previous();
			//System.out.println("prev: " + prevStation.stop);
			//System.out.println("next: " + selectedStation.stop);
			for (int k = prevStation.stop + 1; k <= selectedStation.stop - 1; k++) {
				if (isRemovingPrimary) {
					if (backupVars.xBStop[b][k]) {
						double newPrimaryArrivalEnergy = ToolsMTD.round(primaryVars.c[b][k + 1] + minEnergyNeeded +
								remainingEnergyPrev);
						double oldBackupArrivalEnergy = ToolsMTD.round(backupVars.c[b][k + 1]);
						boolean canBacktoPrimary = oldBackupArrivalEnergy >= newPrimaryArrivalEnergy;
						isFeasible = isFeasible && canBacktoPrimary;
					}
				}
				//************END OF: Removing primary*********
				//************Removing backup*********
				if (!isRemovingPrimary) {
					if (primaryVars.xBStop[b][k]) {
						double newBackupArrivalEnergy = ToolsMTD.round(backupVars.c[b][k] + minEnergyNeeded +
								remainingEnergyPrev);
						double oldPrimaryArrivalEnergy = ToolsMTD.round(primaryVars.c[b][k]);
						boolean canGotoBackup = oldPrimaryArrivalEnergy >= newBackupArrivalEnergy;
						isFeasible = isFeasible && canGotoBackup;
					}
				}
			}
			if (!isRemovingPrimary) {
				double newBackupArrivalTime = ToolsMTD.round(backupVars.arrivalTime[b][prevStation.stop + 1] + 
						minNeededChargingTime + remainingChargingTimePrev);
				double oldPrimaryArrivalTime = ToolsMTD.round(primaryVars.arrivalTime[b][prevStation.stop + 1] );
				boolean canGotoPrimary = newBackupArrivalTime <= oldPrimaryArrivalTime;
				isFeasible = isFeasible && canGotoPrimary;
			}
			
			openStations.next();
		}
		openStations.next();
		
		if (openStations.hasNext()) {
			OpenStation nextStation = openStations.next();
			double remainingChargingTimeNext = remainingEnergyNext / instance.chargingRate;
			for (int k = selectedStation.stop + 1; k <= nextStation.stop - 1; k++) {
				if (isRemovingPrimary && backupVars.xBStop[b][k]) {
					double newPrimaryArrivalTime = ToolsMTD.round(primaryVars.arrivalTime[b][k + 1] -
							remainingChargingTimeNext);
					double oldBackupArrivalTime = ToolsMTD.round(backupVars.arrivalTime[b][k + 1]);
					boolean canGotoPrimary = oldBackupArrivalTime <= newPrimaryArrivalTime;
					isFeasible = isFeasible && canGotoPrimary;
				} else if (!isRemovingPrimary && primaryVars.xBStop[b][k]) {
					double newBackupArrivalTime = ToolsMTD.round(backupVars.arrivalTime[b][k] -
							remainingChargingTimeNext);
					double oldPrimaryArrivalTime = ToolsMTD.round(primaryVars.arrivalTime[b][k]);
					boolean canGotoBackup = oldPrimaryArrivalTime <= newBackupArrivalTime;
					isFeasible = isFeasible && canGotoBackup;
				}
			}
			openStations.previous();
		}
		
		return isFeasible;
	}
	

}
