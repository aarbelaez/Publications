package greedy;

import core.InstanceMTD;
import heuristics.HeuristicsVariablesSet;

public class StrategyRechargeMaxAllowed extends StrategyRecharge {

	public StrategyRechargeMaxAllowed(HeuristicsVariablesSet solution, InstanceMTD instance) {
		super(solution, instance);
		// TODO Auto-generated constructor stub
	}

	@Override
	public void updateStop(int b, int k, double currentEnergy, double currentTime) {
		int station = instance.paths[b][k];
		int nextStation = instance.paths[b][k + 1];
		if (currentEnergy - instance.D[station][nextStation] < instance.Cmin) {
			solution.e[b][k] = instance.Cmax - currentEnergy;
		} else {
			solution.e[b][k] = 0;	
		}

	}

}
