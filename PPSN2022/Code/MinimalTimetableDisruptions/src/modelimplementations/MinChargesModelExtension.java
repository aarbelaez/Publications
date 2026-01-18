package modelimplementations;

import ilog.concert.IloException;
import ilog.concert.IloLinearNumExpr;
import modelinterface.RouteModelExtension;
import variables.BasicVariablesSet;

public class MinChargesModelExtension extends RouteModelExtension {

BasicVariablesSet variablesSet;
	
	public MinChargesModelExtension(BasicVariablesSet variablesSet) {
		this.variablesSet = variablesSet;
	}
	
	public void addConstraintsPerBus(int b) throws IloException {
		IloLinearNumExpr minCharges = cplex.linearNumExpr();
		for (int k = 0; k < instance.paths[b].length; k++) {
			minCharges.addTerm(1.0, variablesSet.xBStop[b][k]);
		}
		cplex.addGe(minCharges, instance.requiredXStops.get(b));
	};

	@Override
	public void addConstraintsPerStop(int b, int k, int m, int i, int j) throws IloException {
		// TODO Auto-generated method stub

	}

	@Override
	public void addConstraintsPerStation(int b, int i) throws IloException {
		// TODO Auto-generated method stub

	}

}
