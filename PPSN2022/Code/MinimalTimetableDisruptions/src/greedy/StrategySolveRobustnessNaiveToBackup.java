package greedy;

import heuristics.HeuristicsVariablesSet;

public class StrategySolveRobustnessNaiveToBackup implements StrategySolveRobustness {
	
	private HeuristicsVariablesSet primaryVars;
	private HeuristicsVariablesSet backupVars;
	
	public StrategySolveRobustnessNaiveToBackup(HeuristicsVariablesSet primaryVars,
			HeuristicsVariablesSet backupVars) {
		this.primaryVars = primaryVars;
		this.backupVars = backupVars;
	}

	@Override
	public double energyNeededToSolveRobustProblems(int b, int k) {
		double energyNeeded = 0;
		if (primaryVars.c[b][k] == 0) {
			return 0;
		}
		if (primaryVars.ct[b][k] == 0) {
			return 0;
		}
		if (primaryVars.c[b][k] < backupVars.c[b][k]) {
			energyNeeded = backupVars.c[b][k] - primaryVars.c[b][k];
		}
		return energyNeeded;
	}

	@Override
	public boolean hasTimeRobustProblemsSolved(int b, int k) {
		double delayRequired = 0;
		if (primaryVars.c[b][k] == 0) {
			return false;
		}
		if (primaryVars.ct[b][k] == 0) {
			return false;
		}
		if (primaryVars.arrivalTime[b][k] > backupVars.arrivalTime[b][k]) {
			//System.out.println("Solving to backup time robust problems");
			delayRequired = primaryVars.arrivalTime[b][k] - backupVars.arrivalTime[b][k];
			backupVars.arrivalTime[b][k] += delayRequired;
			return true;
		}
		return false;
	}

	

}
