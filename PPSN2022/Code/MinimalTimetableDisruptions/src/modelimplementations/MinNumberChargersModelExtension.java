package modelimplementations;

import ilog.concert.IloException;
import ilog.concert.IloLinearNumExpr;
import modelinterface.ModelExtension;
import variables.StationVariablesSet;

public class MinNumberChargersModelExtension extends ModelExtension {
	
	StationVariablesSet stationVariablesSet;
	
	public MinNumberChargersModelExtension(StationVariablesSet stationVariablesSet) {
		this.stationVariablesSet = stationVariablesSet;
	}

	@Override
	public void addConstraintsGlobally() throws IloException {
		IloLinearNumExpr minChargers = cplex.linearNumExpr();
		for (int i = 0; i < stationVariablesSet.x.length; i++) {
			minChargers.addTerm(1.0, stationVariablesSet.x[i]);
		}
		cplex.addLe(minChargers, instance.minNumberChargers);
	}

	

}
