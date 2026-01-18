package greedy;

import core.InstanceMTD;
import heuristics.HeuristicsVariablesSet;

public abstract class StrategyRecharge {
	
	HeuristicsVariablesSet solution;
	InstanceMTD instance;
	CurrentStatus status;
	
	
	
	



	public StrategyRecharge(HeuristicsVariablesSet solution, InstanceMTD instance) {
		super();
		this.solution = solution;
		this.instance = instance;
	}
	
	public void setStatus(CurrentStatus status) {
		this.status = status;
	}


	public abstract void updateStop(int b, int k, double currentEnergy, double currentTime);

}
