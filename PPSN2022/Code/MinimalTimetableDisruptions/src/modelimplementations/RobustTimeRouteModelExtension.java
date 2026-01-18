package modelimplementations;

import ilog.concert.IloException;
import ilog.concert.IloLinearNumExpr;
import modelinterface.RouteModelExtension;
import variables.BasicVariablesSet;

/**
 * With this constraint we prevent the backup route to arrive early at the primary stations.
 * This is important for the buses reaction, which must not depend on previous states
 * @author César Loaiza
 *
 */
public class RobustTimeRouteModelExtension extends RouteModelExtension {

	BasicVariablesSet routeVariablesSet;
	BasicVariablesSet backupRouteVariablesSet;
	
	public RobustTimeRouteModelExtension(BasicVariablesSet routeVariablesSet,
			BasicVariablesSet backupRouteVariablesSet) {
		this.routeVariablesSet = routeVariablesSet;
		this.backupRouteVariablesSet = backupRouteVariablesSet;
	}

	@Override
	public void addConstraintsPerStop(int b, int k, int m, int i, int j) throws IloException {
		
		
		//double bigM = instance.M;
		//double bigM = (instance.originalTimetable[b][k] + instance.DTmax )  - 
	   	//	(instance.originalTimetable[b][k] - instance.DTmax);
		double bigM = 2*instance.DTmax;
		//double bigM = 10000000;
		
		IloLinearNumExpr timeContraint = cplex.linearNumExpr();
		timeContraint.addTerm(1.0, routeVariablesSet.arrivalTime[b][k]);
		timeContraint.addTerm(-1.0, backupRouteVariablesSet.arrivalTime[b][k]);
		//timeContraint.addTerm(-1.0, backupRouteVariablesSet.ct[b][m]);
		timeContraint.addTerm(bigM, routeVariablesSet.xBStop[b][k]);
		//cplex.addLe(timeContraint, instance.T[b][m] + bigM); 
		cplex.addLe(timeContraint, bigM); 
	}

	@Override
	public void addConstraintsPerStation(int b, int i) throws IloException {
		// TODO Auto-generated method stub

	}

}
