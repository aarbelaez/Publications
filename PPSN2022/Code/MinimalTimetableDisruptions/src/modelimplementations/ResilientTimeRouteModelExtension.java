package modelimplementations;

import ilog.concert.IloException;
import ilog.concert.IloLinearNumExpr;
import modelinterface.RouteModelExtension;
import variables.BasicVariablesSet;

public class ResilientTimeRouteModelExtension extends RouteModelExtension {
	
	BasicVariablesSet routeVariablesSet;
	BasicVariablesSet backupRouteVariablesSet;
	
	public ResilientTimeRouteModelExtension(BasicVariablesSet routeVariablesSet,
			BasicVariablesSet backupRouteVariablesSet) {
		this.routeVariablesSet = routeVariablesSet;
		this.backupRouteVariablesSet = backupRouteVariablesSet;
	}

	@Override
	public void addConstraintsPerStop(int b, int k, int m, int i, int j) throws IloException {
		
		
		//double bigM = instance.M;
		double bigM = ((instance.originalTimetable[b][m] + instance.DTmax)  + instance.maxT) - 
	   		(instance.originalTimetable[b][k] - instance.DTmax);
		//double bigM = 1000000;
		
		IloLinearNumExpr timeContraint = cplex.linearNumExpr();
		timeContraint.addTerm(1.0, routeVariablesSet.arrivalTime[b][k]);
		timeContraint.addTerm(-1.0, backupRouteVariablesSet.arrivalTime[b][m]);
		timeContraint.addTerm(-1*bigM, routeVariablesSet.xBStop[b][m]);
		cplex.addGe(timeContraint, instance.T[b][m] - bigM); 
	}

	@Override
	public void addConstraintsPerStation(int b, int i) throws IloException {
		// TODO Auto-generated method stub

	}

}
