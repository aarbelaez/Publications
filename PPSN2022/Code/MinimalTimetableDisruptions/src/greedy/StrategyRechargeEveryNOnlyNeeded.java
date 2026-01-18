package greedy;

import core.InstanceMTD;
import heuristics.HeuristicsVariablesSet;

/**
 * In this strategy, a bus recharge only the min necessary to get three stops ahead
 * @author cedaloaiza
 *
 */
public class StrategyRechargeEveryNOnlyNeeded extends StrategyRecharge {

	public StrategyRechargeEveryNOnlyNeeded(HeuristicsVariablesSet solution, InstanceMTD instance) {
		super(solution, instance);
		historicExtraEnergy = new double[instance.b][];
		for (int b = 0; b < instance.b; b++) {
			historicExtraEnergy[b] = new double[instance.paths[b].length];
		}
	}
	
	/**
	 * Determines shifting (next or prev) of where to locate stations
	 */
	private int shifting = 0;
	
	private double requiredExtraEnergy = 0;
	
	private boolean isChargingForbidden = false;
	
	public int stepLength = 3;
	
	public double[][] historicExtraEnergy;
	
	public void setShifting(int shifting) {
		this.shifting = shifting;
	}

	public void setRequiredExtraEnergy(double requiredExtraEnergy) {
		this.requiredExtraEnergy = requiredExtraEnergy;
	}

	public void setChargingForbidden(boolean isChargingForbidden) {
		this.isChargingForbidden = isChargingForbidden;
	}

	@Override
	public void updateStop(int b, int k, double currentEnergy, double currentTime) {
		double distance = 0;
		int step = (instance.paths[b].length + shifting - k) % stepLength;
		
		for (int s = k; s <= k + step; s++) {
			try {
				int station = instance.paths[b][s];
				int nextStation = instance.paths[b][s + 1];
				distance += instance.D[station][nextStation];
			} catch (IndexOutOfBoundsException e) {
				//System.out.printf("Out of bound in [%s][%s]\n", b, s);
				//e.printStackTrace();
			}
		}
		//System.out.printf("k=%s, step=%s\n", k, step);
		//System.out.printf("%s < %s\n", currentEnergy - distance, instance.Cmin);
		if (currentEnergy - distance < instance.Cmin && !isChargingForbidden) {
			solution.e[b][k] = instance.Cmin + distance - currentEnergy   +   requiredExtraEnergy   +     historicExtraEnergy[b][k];
			if (requiredExtraEnergy < 0) {
				System.out.println("ERROR computing required extra energy for robustness");
				System.exit(0);
			} else if (requiredExtraEnergy > 0) {
				historicExtraEnergy[b][k] += requiredExtraEnergy;
				requiredExtraEnergy = 0;
			}
			//System.out.println("Current Energy: " + solution.c[b][k]);
			System.out.println("Recharging: " + solution.e[b][k]);
		} else {
			solution.e[b][k] = 0;	
			/*
			//for exploiting dwell times
			double potentialNextArrivalTime = solution.arrivalTime[b][k] + instance.T[b][k];
			if (potentialNextArrivalTime < instance.originalTimetable[b][k + 1] - instance.DTmax) {
				solution.e[b][k] = 1;	
			}
			*/		
		}
		
	}

}
