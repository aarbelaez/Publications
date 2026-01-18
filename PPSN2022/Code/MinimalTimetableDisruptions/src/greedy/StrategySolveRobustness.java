package greedy;

/**
 * Strategis to solve robustness inconsistences once values to one stop has been set for primary and backup routes
 * @author cedaloaiza
 *
 */
public interface StrategySolveRobustness {
	
	/**
	 * Returns the required energy for switching between routes (Energy for being gotten in prev open stop)
	 * @param b
	 * @param k
	 * @return
	 */
	public double energyNeededToSolveRobustProblems(int b, int k);
	/**
	 * Delay the arrival time for switching between routes
	 * Returns true if the arrival time changed
	 * @param b
	 * @param k
	 * @return
	 */
	public boolean hasTimeRobustProblemsSolved(int b, int k);
	
}
