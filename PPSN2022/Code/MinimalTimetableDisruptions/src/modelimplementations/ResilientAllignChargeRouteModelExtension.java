package modelimplementations;

import ilog.concert.IloException;
import ilog.concert.IloLinearNumExpr;
import modelinterface.RouteModelExtension;
import variables.BasicVariablesSet;

public class ResilientAllignChargeRouteModelExtension extends RouteModelExtension {

	BasicVariablesSet routeVariablesSet;
	BasicVariablesSet backupRouteVariablesSet;
	
	public ResilientAllignChargeRouteModelExtension(BasicVariablesSet routeVariablesSet,
			BasicVariablesSet backupRouteVariablesSet) {
		this.routeVariablesSet = routeVariablesSet;
		this.backupRouteVariablesSet = backupRouteVariablesSet;
	}
	
	@Override
	public void addConstraintsPerStop(int b, int k, int m, int i, int j) throws IloException {
		
		//double bigM = instance.M;
		double bigM = (instance.Cmax - instance.Cmin) + instance.maxD;
		//double bigM = 1000000;
		

		
		IloLinearNumExpr allignRouteConstraint = cplex.linearNumExpr();
		allignRouteConstraint.addTerm(1.0, routeVariablesSet.c[b][k]);
		allignRouteConstraint.addTerm(-1.0, backupRouteVariablesSet.c[b][m]);
		allignRouteConstraint.addTerm(-1.0, backupRouteVariablesSet.e[b][m]);
		allignRouteConstraint.addTerm(bigM, routeVariablesSet.xBStop[b][k]);
		cplex.addLe(allignRouteConstraint, -1*instance.D[i][j] + bigM); 
		

	}

	@Override
	public void addConstraintsPerStation(int b, int i) throws IloException {
		

	}

}
