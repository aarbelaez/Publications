package heuristics;

import java.util.Random;

import core.InstanceMTD;
import utils.ToolsMTD;

public class RobustAddOperator extends AddOperator {

	RobustCompleteSolution completeCurrentSolution;
	
	/**
	 * Whether we are adding a primary station or a backup one
	 */
	boolean isAddingPrimary = true;
	
	public RobustAddOperator(RobustCompleteSolution currentSolution, InstanceMTD instance, Random rn) {
		super(currentSolution, instance, rn);
		this.completeCurrentSolution = currentSolution;
	}
	
	public void addStation(int i) {
		System.out.println("Adding primary station first --------------\n");
		super.resetOperator(completeCurrentSolution);
		isAddingPrimary = true;
		super.addStation(i);
		/*
		if (!completeCurrentSolution.checkFeasibility(instance, "partial")) {
			System.exit(0);
		}
		*/
		System.out.println("Adding backup station now -----------------\n");
		super.resetOperator(completeCurrentSolution.robustRoute);
		isAddingPrimary = false;
		super.addStation(i);
		/*
		if (!completeCurrentSolution.checkFeasibility(instance, "partial")) {
			System.exit(0);
		}
		*/
	}
	
	public void resetOperator(CompleteSolution currentSolution) {
		super.resetOperator(currentSolution);
		this.completeCurrentSolution = (RobustCompleteSolution) currentSolution;
	}
	
	public boolean checkFeasibility(Stop stop, int b, int i, OpenStation nextOpenStation) {
		
		boolean isFeasible = super.checkFeasibility(stop, b, i, nextOpenStation);
		
		if (!isFeasible) {
			return false;
		}
		
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
					double newBackupArrivalEnergy = ToolsMTD.round(backupVars.c[b][ka] + nextAddedEnergy);
					double oldPrimaryArrivalEnergy = ToolsMTD.round(primaryVars.c[b][ka]);
					boolean canGotoBackup = oldPrimaryArrivalEnergy >= newBackupArrivalEnergy;
					isFeasible = isFeasible && canGotoBackup;
				}
				
				if (isAddingPrimary && backupVars.xBStop[b][ka]) {
					double newPrimaryArrivalEnergy = ToolsMTD.round(primaryVars.c[b][ka + 1] + nextAddedEnergy);
					double oldBackupArrivalEnergy = ToolsMTD.round(backupVars.c[b][ka + 1]);
					boolean canGotoPrimary = oldBackupArrivalEnergy >= newPrimaryArrivalEnergy;
					isFeasible = isFeasible && canGotoPrimary;
				}
			}
			
			if (isAddingPrimary) {
				double newPrimaryArrivalTime = ToolsMTD.round(primaryVars.arrivalTime[b][nextK] + nextAddedCtime);
				double oldBackupArrivalTime = ToolsMTD.round(backupVars.arrivalTime[b][nextK]);
				boolean canGoToPrimaryT = newPrimaryArrivalTime <= oldBackupArrivalTime;
				isFeasible = isFeasible && canGoToPrimaryT;
			} else {
				
			}
			
			openStations.previous();		
		}
		
		
		
		if (openStations.hasPrevious()) {
			OpenStation prevStop = openStations.previous();
			int prevK = prevStop.stop;
			
			for (int m = prevK + 1; m <= k - 1; m++) {
				if (isAddingPrimary) {	
					if (backupVars.xBStop[b][m]) {
						double newPrimaryArrivalTime = ToolsMTD.round(primaryVars.arrivalTime[b][m + 1] - prevAddedCtime);
						double oldBackupArrivalTime = ToolsMTD.round(backupVars.arrivalTime[b][m + 1]);
						boolean canGoToPrimaryT = oldBackupArrivalTime <= newPrimaryArrivalTime;
						isFeasible = isFeasible && canGoToPrimaryT;
					}
				} else {
					if (primaryVars.xBStop[b][m]) {
						double newBackupArrivalTime = ToolsMTD.round(primaryVars.arrivalTime[b][m] - prevAddedCtime);
						double oldPrimaryArrivalTime = ToolsMTD.round(backupVars.arrivalTime[b][m]);
						boolean canGoToBackupT = oldPrimaryArrivalTime <= newBackupArrivalTime;
						isFeasible = isFeasible && canGoToBackupT;
					}
				}
			}
			openStations.next();	
		}
		
		
		
		
		if (isAddingPrimary) {	
			double newPrimaryArrivalEnergy = ToolsMTD.round(primaryVars.c[b][k] - prevAddednergy);
			double oldBackupArrivalEnergy = ToolsMTD.round(backupVars.c[b][k]);
			boolean canGoToBackup = newPrimaryArrivalEnergy >= oldBackupArrivalEnergy;
			isFeasible = isFeasible && canGoToBackup;	
			double newPrimaryArrivalTime = ToolsMTD.round(primaryVars.arrivalTime[b][k] - prevAddedCtime);
			double oldBackupArrivalTime = ToolsMTD.round(backupVars.arrivalTime[b][k]);
			boolean canGoToBackupT = newPrimaryArrivalTime <= oldBackupArrivalTime;
			isFeasible = isFeasible && canGoToBackupT;	
		} else {		
			try {
				double newBackupArrivalEnergyToK = ToolsMTD.round(backupVars.c[b][k + 1] + nextAddedEnergy);
				double oldPrimaryArrivalEnergyToK = ToolsMTD.round(primaryVars.c[b][k + 1]);
				boolean canGoToPrimaryFromK = newBackupArrivalEnergyToK >= oldPrimaryArrivalEnergyToK;
				isFeasible = isFeasible && canGoToPrimaryFromK;
				
				double newBackupArrivalTimeToK = ToolsMTD.round(backupVars.arrivalTime[b][k + 1] + nextAddedCtime);
				double oldPrimaryArrivalTimeToK = ToolsMTD.round(primaryVars.arrivalTime[b][k + 1]);
				boolean canGoToPrimaryTimeFromK = newBackupArrivalTimeToK <= oldPrimaryArrivalTimeToK;
				isFeasible = isFeasible && canGoToPrimaryTimeFromK;
			} catch (ArrayIndexOutOfBoundsException e) {
				//Nothing to do
				//There is not need to check the try code
			}
			
			if (openStations.hasPrevious()) {
				OpenStation prevStop = openStations.previous();
				int prevK = prevStop.stop;
				
				double newBackupArrivalEnergyToPrevK = ToolsMTD.round(backupVars.c[b][prevK + 1] - prevAddednergy);
				double oldPrimaryArrivalEnergyToPrevK = ToolsMTD.round(primaryVars.c[b][prevK + 1]);
				boolean canGoToPrimaryFromPrevK = newBackupArrivalEnergyToPrevK >= oldPrimaryArrivalEnergyToPrevK;
				isFeasible = isFeasible && canGoToPrimaryFromPrevK;
				//System.out.printf("IS CHECKING! stop %s [%s][%s] - %s\n", prevK + 1, b, k, canGoToPrimaryFromPrevK);
				//System.out.printf("%s >= %s\n", newBackupArrivalEnergyToPrevK, oldPrimaryArrivalEnergyToPrevK);
				openStations.next();	
			}
		}
		
		if (nextOpenStation != null) {	
			openStations.next();
		}
			
		
		return isFeasible;
	}

}
