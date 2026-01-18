package modelimplementations;

import ilog.concert.IloException;
import ilog.concert.IloLinearNumExpr;
import modelinterface.RouteModelExtension;
import variables.BasicVariablesSet;

public class RobustChargeBackBackupModelExtension extends RouteModelExtension {
	BasicVariablesSet routeVariablesSet;
	BasicVariablesSet backupRouteVariablesSet;
	
	public RobustChargeBackBackupModelExtension(BasicVariablesSet routeVariablesSet,
			BasicVariablesSet backupRouteVariablesSet) {
		this.routeVariablesSet = routeVariablesSet;
		this.backupRouteVariablesSet = backupRouteVariablesSet;
	}
	
	@Override
	public void addConstraintsPerStop(int b, int k, int m, int i, int j) throws IloException {
		
		double bigM3 = instance.Cmax - instance.Cmin;
		
		
		//double bigM3 = 10000000;
		
		IloLinearNumExpr allignRouteConstraintEarlyBackupPrevention = cplex.linearNumExpr();
		allignRouteConstraintEarlyBackupPrevention.addTerm(1.0, backupRouteVariablesSet.c[b][k]);
		allignRouteConstraintEarlyBackupPrevention.addTerm(-1.0, routeVariablesSet.c[b][k]);
		allignRouteConstraintEarlyBackupPrevention.addTerm(-1*bigM3, backupRouteVariablesSet.xBStop[b][m]);
		cplex.addGe(allignRouteConstraintEarlyBackupPrevention, -1*bigM3); 
		
		

	}

	@Override
	public void addConstraintsPerStation(int b, int i) throws IloException {
		// TODO Auto-generated method stub

	}
}
