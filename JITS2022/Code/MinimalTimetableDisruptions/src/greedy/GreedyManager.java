package greedy;

import java.util.TreeMap;

import core.InstanceMTD;
import heuristics.HeuristicsVariablesSet;

public class GreedyManager {
	EasiestGreedyHeuristic primaryGreedy;
	EasiestGreedyHeuristic backupGreedy;
	InstanceMTD instance;
	
	private StrategyRechargeEveryNOnlyNeeded primaryOnlyNeededStrategy;
	private StrategyRechargeEveryNOnlyNeeded backupOnlyNeededStrategy;
	
	int firstShifing;
	int backupShifting;
	
	
	public GreedyManager(InstanceMTD instance, HeuristicsVariablesSet primaryVars, HeuristicsVariablesSet backupVars) {
		this.instance = instance;
		//StrategyRechargeOnlyNeeded primaryOnlyNeededStrategy = new StrategyRechargeOnlyNeeded(primaryVars, instance);
		
		primaryOnlyNeededStrategy = new StrategyRechargeEveryNOnlyNeeded(primaryVars, instance);
		backupOnlyNeededStrategy = new StrategyRechargeEveryNOnlyNeeded(backupVars, instance);
		int stepLength = 20;
		primaryOnlyNeededStrategy.stepLength = stepLength;
		backupOnlyNeededStrategy.stepLength = stepLength;
		
		StrategySolveRobustness solverToBackup = new StrategySolveRobustnessNaiveToBackup(primaryVars, backupVars);
		StrategySolveRobustness solverBackPrimary = new StrategySolveRobustnessNaiveBackPrimary(primaryVars, backupVars);
		
		primaryGreedy = new EasiestGreedyHeuristic(instance, primaryOnlyNeededStrategy, primaryVars);
		backupGreedy = new EasiestGreedyHeuristic(instance, backupOnlyNeededStrategy, backupVars);
		
		backupGreedy.setRobustSolver(solverToBackup);
		primaryGreedy.setRobustSolver(solverBackPrimary);
	}
	
	public void run() {
		int stepLength = primaryOnlyNeededStrategy.stepLength;
		for (int b = 0; b < instance.b; b++) {
			primaryGreedy.reset(b);
			backupGreedy.reset(b);
			setShiftings(b, primaryOnlyNeededStrategy.stepLength);

			primaryOnlyNeededStrategy.setShifting(firstShifing);
			backupOnlyNeededStrategy.setShifting(backupShifting);	
			
			for (int k = 0; k < instance.paths[b].length; k++) {
				System.out.printf("Updating stop [%s][%s]\n", b, k);
				System.out.printf("Primary energy: %s\n", primaryGreedy.getCurrentEnergy());
				System.out.printf("Backup energy: %s\n", backupGreedy.getCurrentEnergy());
				int primaryStep = (instance.paths[b].length + firstShifing - k) % stepLength;
				// primaryStep == 2 means that primary route will charge
				if (primaryStep == stepLength - 1) {
					primaryGreedy.updateStopVariables(b, k);
					double requiredExtraEnergy = backupGreedy.updateStopVariables(b, k);
					if (requiredExtraEnergy > 0) {
						primaryOnlyNeededStrategy.setRequiredExtraEnergy(requiredExtraEnergy); 
						k -= primaryOnlyNeededStrategy.stepLength + 1;
						backupGreedy.restoreTimeEnergyStatus(b, k + 1);
						primaryGreedy.restoreTimeEnergyStatus(b, k + 1);
						System.out.printf("Energy required: %s. Going back!\n", requiredExtraEnergy);
						/*
						try {
							Thread.sleep(5*1000);
						} catch (InterruptedException e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
						*/
						continue;
						
					}
				} else {
					//Not condition when we are not checking robustness, the order does not matter
					int station = instance.paths[b][k];
					
					if (primaryGreedy.openStations.contains(station)) {
						backupOnlyNeededStrategy.setChargingForbidden(true);
					};
					
					double requiredExtraEnergy = backupGreedy.updateStopVariables(b, k);
					backupOnlyNeededStrategy.setChargingForbidden(false);
					primaryGreedy.updateStopVariables(b, k);
					
					if (requiredExtraEnergy > 0) {
						backupOnlyNeededStrategy.setRequiredExtraEnergy(requiredExtraEnergy); 
						//int backupStep = (instance.paths[b].length + backupShifting - k) % stepLength;
						int backupStep = (instance.paths[b].length + firstShifing - k) % stepLength;
						int positionsAfterCharging = (stepLength - 1) - backupStep;
						System.out.println("Positions: " + positionsAfterCharging);
						if (positionsAfterCharging >= stepLength) {
							System.out.println("positionsAfterCharging too large: " + positionsAfterCharging);
							System.exit(0);
						}
						k -= primaryOnlyNeededStrategy.stepLength + 1 + positionsAfterCharging;
						backupGreedy.restoreTimeEnergyStatus(b, k + 1);
						primaryGreedy.restoreTimeEnergyStatus(b, k + 1);
						System.out.printf("Energy required bkp: %s. Going back!\n", requiredExtraEnergy);
						/*
						try {
							Thread.sleep(5*1000);
						} catch (InterruptedException e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
						*/
						continue;
					}
				}
			}
			//System.exit(0);
		}
	}
	
	/**
	 * Set the shiftings for primary and backup routes 
	 * The one which produce the earlier open stop for bus b will be for primary.
	 * For the backup we need the one that produces the next stop
	 * We consider a specific stepLength
	 * @return
	 */
	private void setShiftings(int b, int stepLength) {
		TreeMap<Integer, Integer> orderedShiftings = new TreeMap<Integer, Integer>();
		for (int s = 0; s < stepLength; s++) {
			int stopForShifting = stopWhereFirstOpenStop(s, b, stepLength);		
			orderedShiftings.put(stopForShifting, s);
			if (stopForShifting == -1) {
				System.out.println("ERROR when computing which shifting we use for robust routes (no shifting)");
				System.exit(1);
			}
			//System.out.printf("stop for shifting %s: %s\n", s, stopForShifting);
		}
		
		/*
		if (stopForShifting0 == stopForShifting1 || stopForShifting0 == stopForShifting2 ||
				stopForShifting1 == stopForShifting2 ) {
			System.out.println("ERROR when computing which shifting we use for robust routes (equal shinftings)");
			System.exit(1);
		}
		*/
		int i = 0;
		for (int key: orderedShiftings.keySet()) {
			if (i == 0) {
				firstShifing = orderedShiftings.get(key);
			} else if (i == 1) {
				backupShifting = orderedShiftings.get(key);
				break;
			}
			i++;
			//System.out.println("First shifting: " + firstShifing);
		}
		//System.exit(1);
	}
	
	/**
	 * Returns the first stop that must be opened for bus b for a particular shifting
	 * Returning -1 is an error
	 * @param shifting
	 * @param b
	 * @return
	 */
	private int stopWhereFirstOpenStop(int shifting, int b, int stepLength) {
		double battery_level = instance.Cmax;
		for (int k = 0; k < instance.paths[b].length; k++) {
			double distance = 0;
			int step = (instance.paths[b].length + shifting - k) % stepLength;
			for (int s = k; s <= k + step; s++) {
				try {
					int station = instance.paths[b][s];
					int nextStation = instance.paths[b][s + 1];
					distance += instance.D[station][nextStation];
				} catch (IndexOutOfBoundsException e) {
					return k;
				}
			}
			if (battery_level - distance < instance.Cmin) {
				//System.out.println("Step: " + step);
				return k;
			}
			int station = instance.paths[b][k];
			int nextStation = instance.paths[b][k + 1];
			battery_level -= instance.D[station][nextStation];
		}
		return -1;
	}

}
