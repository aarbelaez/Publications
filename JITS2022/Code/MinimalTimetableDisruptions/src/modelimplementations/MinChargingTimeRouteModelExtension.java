package modelimplementations;

import ilog.concert.IloException;
import ilog.concert.IloLinearNumExpr;
import modelinterface.RouteModelExtension;
import variables.BasicVariablesSet;

public class MinChargingTimeRouteModelExtension extends RouteModelExtension {

	BasicVariablesSet variablesSet;
	
	public MinChargingTimeRouteModelExtension(BasicVariablesSet variablesSet) {
		this.variablesSet = variablesSet;
	}
	@Override
	public void addConstraintsPerStop(int b, int k, int m, int i, int j) throws IloException {
		IloLinearNumExpr minCtConstraint = cplex.linearNumExpr();
		minCtConstraint.addTerm(1.0, variablesSet.ct[b][k]);
		minCtConstraint.addTerm(-1*instance.minCt, variablesSet.xBStop[b][k]);	
		cplex.addGe(minCtConstraint, 0); 	
	}

	@Override
	public void addConstraintsPerStation(int b, int i) throws IloException {
		// TODO Auto-generated method stub

	}

}
