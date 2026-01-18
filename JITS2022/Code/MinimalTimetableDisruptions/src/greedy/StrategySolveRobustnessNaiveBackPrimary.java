package greedy;

import heuristics.HeuristicsVariablesSet;

public class StrategySolveRobustnessNaiveBackPrimary implements StrategySolveRobustness {
	
	private HeuristicsVariablesSet primaryVars;
	private HeuristicsVariablesSet backupVars;
	
	public StrategySolveRobustnessNaiveBackPrimary(HeuristicsVariablesSet primaryVars,
			HeuristicsVariablesSet backupVars) {
		this.primaryVars = primaryVars;
		this.backupVars = backupVars;
	}
	
	@Override
	public double energyNeededToSolveRobustProblems(int b, int k) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public boolean hasTimeRobustProblemsSolved(int b, int k) {
		double delayRequired = 0;
		
		if (backupVars.c[b][k] == 0) {
			return false;
		}
		if (k == 0 || backupVars.ct[b][k - 1] == 0) {
			return false;
		}
		
		if (backupVars.arrivalTime[b][k] > primaryVars.arrivalTime[b][k]) {
			//System.out.printf("Solving back primary time robust problems [%s][%s]\n", b, k);
			delayRequired = backupVars.arrivalTime[b][k] - primaryVars.arrivalTime[b][k];
			primaryVars.arrivalTime[b][k] += delayRequired;
			return true;
		}
		return false;
	}

}
