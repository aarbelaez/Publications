package modelimplementations;

import ilog.concert.IloException;
import ilog.concert.IloLinearNumExpr;
import modelinterface.RouteModelExtension;
import variables.BasicVariablesSet;

public class ResilientRobustAllignChargeRouteModelExtension extends RouteModelExtension {

	BasicVariablesSet routeVariablesSet;
	BasicVariablesSet backupRouteVariablesSet;
	
	public ResilientRobustAllignChargeRouteModelExtension(BasicVariablesSet routeVariablesSet,
			BasicVariablesSet backupRouteVariablesSet) {
		this.routeVariablesSet = routeVariablesSet;
		this.backupRouteVariablesSet = backupRouteVariablesSet;
	}
	
	@Override
	public void addConstraintsPerStop(int b, int k, int m, int i, int j) throws IloException {
		
		double bigM3 = (instance.Cmax - instance.Cmin) + (instance.chargingRate*instance.maxChargingTime);
		
		
		//double bigM3 = 10000000;
		
		IloLinearNumExpr allignRouteConstraintEarlyBackupPrevention = cplex.linearNumExpr();
		allignRouteConstraintEarlyBackupPrevention.addTerm(1.0, routeVariablesSet.c[b][k]);
		allignRouteConstraintEarlyBackupPrevention.addTerm(-1.0, backupRouteVariablesSet.c[b][m]);
		allignRouteConstraintEarlyBackupPrevention.addTerm(-1.0, backupRouteVariablesSet.e[b][m]);
		allignRouteConstraintEarlyBackupPrevention.addTerm(-1*bigM3, routeVariablesSet.xBStop[b][k]);
		cplex.addGe(allignRouteConstraintEarlyBackupPrevention, -1*instance.D[i][j] + -1*bigM3); 
		
		

	}

	@Override
	public void addConstraintsPerStation(int b, int i) throws IloException {
		

	}

}
