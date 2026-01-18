package modelimplementations;

import ilog.concert.IloException;
import ilog.concert.IloLinearNumExpr;
import ilog.cplex.IloCplex;
import modelinterface.RouteModelExtension;
import utils.ToolsMTD;
import variables.BasicVariablesSet;
import variables.StationChargeVariablesSet;
import variables.StationVariablesSet;

public class StationChargeRouteModelExtension extends RouteModelExtension {
	
	BasicVariablesSet variablesSet;
	StationVariablesSet stationVariablesSet;
	StationChargeVariablesSet stationChargeVariablesSet;
	
	public StationChargeRouteModelExtension(BasicVariablesSet variablesSet, StationVariablesSet stationVars,
			StationChargeVariablesSet stationChargeVariablesSet) {
		this.variablesSet = variablesSet;
		this.stationVariablesSet = stationVars;
		this.stationChargeVariablesSet = stationChargeVariablesSet;
	}
	
	@Override
	public void defineVariables(IloCplex cplex) throws IloException {
		stationChargeVariablesSet.setCplex(cplex);
		stationChargeVariablesSet.defineVariables();
	}

	@Override
	public void addConstraintsPerStop(int b, int k, int m, int i, int j) throws IloException {
	    IloLinearNumExpr chargingStationsConstraint = cplex.linearNumExpr();
		chargingStationsConstraint.addTerm(1.0, stationChargeVariablesSet.xBStation[b][i]);
		chargingStationsConstraint.addTerm(-1.0, variablesSet.xBStop[b][k]);
		cplex.addGe(chargingStationsConstraint, 0); // CONSTRAINT (16)
	}

	@Override
	public void addConstraintsPerStation(int b, int i) throws IloException {
		if (ToolsMTD.stationInPath(b, i, instance.paths)) {
			IloLinearNumExpr chargingStationsConstraint = cplex.linearNumExpr();
			chargingStationsConstraint.addTerm(1.0, stationVariablesSet.x[i]);
			chargingStationsConstraint.addTerm(-1.0, stationChargeVariablesSet.xBStation[b][i]);
			cplex.addGe(chargingStationsConstraint, 0); // CONSTRAINT (7)
		}

	}

}
