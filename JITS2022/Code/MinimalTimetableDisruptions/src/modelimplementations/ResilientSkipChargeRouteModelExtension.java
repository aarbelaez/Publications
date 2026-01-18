package modelimplementations;

import ilog.concert.IloException;
import ilog.concert.IloLinearNumExpr;
import modelinterface.RouteModelExtension;
import variables.BasicVariablesSet;
import variables.StationChargeVariablesSet;

public class ResilientSkipChargeRouteModelExtension extends RouteModelExtension {

	StationChargeVariablesSet routeVariablesSet;
	StationChargeVariablesSet backupRouteVariablesSet;
	
	public ResilientSkipChargeRouteModelExtension(StationChargeVariablesSet routeVariablesSet,
			StationChargeVariablesSet backupRouteVariablesSet) {
		this.routeVariablesSet = routeVariablesSet;
		this.backupRouteVariablesSet = backupRouteVariablesSet;
	}
	
	@Override
	public void addConstraintsPerStop(int b, int k, int m, int i, int j) throws IloException {
		
		//double bigM = instance.M;
		double bigM = (instance.Cmax - instance.Cmin) + instance.maxD;
		//double bigM = 1000000;
		
		/*
		IloLinearNumExpr skipStationConstraint = cplex.linearNumExpr();
		skipStationConstraint.addTerm(1.0, backupRouteVariablesSet.c[b][k]);
		skipStationConstraint.addTerm(-1.0, backupRouteVariablesSet.c[b][m]);
		skipStationConstraint.addTerm(bigM, routeVariablesSet.xBStation[b][j]);
		cplex.addLe(skipStationConstraint, -1*instance.D[i][j] + bigM); 
		*/
		
		/*
		double bigM2 = instance.maxChargingTime*instance.chargingRate;
		
		IloLinearNumExpr skipStationConstraint = cplex.linearNumExpr();
		skipStationConstraint.addTerm(1.0, backupRouteVariablesSet.e[b][m]);
		skipStationConstraint.addTerm(bigM2, routeVariablesSet.xBStation[b][j]);
		cplex.addLe(skipStationConstraint, bigM2);
		*/
		

	}

	@Override
	public void addConstraintsPerStation(int b, int i) throws IloException {
		
		IloLinearNumExpr skipStationConstraint = cplex.linearNumExpr();
		skipStationConstraint.addTerm(1.0, backupRouteVariablesSet.xBStation[b][i]);
		skipStationConstraint.addTerm(1.0, routeVariablesSet.xBStation[b][i]);
		cplex.addLe(skipStationConstraint, 1);

	}

}
