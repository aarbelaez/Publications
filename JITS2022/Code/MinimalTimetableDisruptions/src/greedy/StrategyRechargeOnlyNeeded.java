package greedy;

import core.InstanceMTD;
import heuristics.HeuristicsVariablesSet;

/**
 * In this strategy, a bus recharge only the min necessary to get the next stop 
 * @author cedaloaiza
 *
 */
public class StrategyRechargeOnlyNeeded extends StrategyRecharge {

	public StrategyRechargeOnlyNeeded(HeuristicsVariablesSet solution, InstanceMTD instance) {
		super(solution, instance);
	}

	@Override
	public void updateStop(int b, int k, double currentEnergy, double currentTime) {
		int station = instance.paths[b][k];
		int nextStation = instance.paths[b][k + 1];
		if (currentEnergy - instance.D[station][nextStation] < instance.Cmin) {
			solution.e[b][k] = instance.Cmin + instance.D[station][nextStation] - currentEnergy;
		} else {
			solution.e[b][k] = 0;
			
			//for exploiting dwell times
			/*
			double potentialNextArrivalTime = solution.arrivalTime[b][k] + instance.T[b][k];
			if (potentialNextArrivalTime < instance.originalTimetable[b][k + 1] - instance.DTmax && k > 0 //&&
					//instance.inputFolder.contains("limerick")
					) {
				solution.e[b][k] = 1;	
			}
			*/
			
			
			
		}
	}

}
