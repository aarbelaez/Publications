package modelimplementations;

import ilog.concert.IloException;
import ilog.concert.IloLinearNumExpr;
import modelinterface.RouteModelExtension;
import variables.BasicVariablesSet;

public class RobustSkipChargeRouteModelExtension extends RouteModelExtension {
	
	BasicVariablesSet routeVariablesSet;
	BasicVariablesSet backupRouteVariablesSet;
	
	public RobustSkipChargeRouteModelExtension(BasicVariablesSet routeVariablesSet,
			BasicVariablesSet backupRouteVariablesSet) {
		this.routeVariablesSet = routeVariablesSet;
		this.backupRouteVariablesSet = backupRouteVariablesSet;
	}

	@Override
	public void addConstraintsPerStop(int b, int k, int m, int i, int j) throws IloException {
		// TODO Auto-generated method stub

	}

	@Override
	public void addConstraintsPerStation(int b, int i) throws IloException {
		// TODO Auto-generated method stub

	}
	
	public void addConstraintsBetweenSeparatedStops(int b, int k, int m, int i, int j) throws IloException {
		if (i == j && m >= k) {
			IloLinearNumExpr skipStationConstraint = cplex.linearNumExpr();
			skipStationConstraint.addTerm(1.0, backupRouteVariablesSet.xBStop[b][m]);
			skipStationConstraint.addTerm(1.0, routeVariablesSet.xBStop[b][k]);
			cplex.addLe(skipStationConstraint, 1);	
		}
	};

}
