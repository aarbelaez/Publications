package modelimplementations;

import ilog.concert.IloException;
import ilog.concert.IloLinearNumExpr;
import modelinterface.RouteModelExtension;
import variables.BasicVariablesSet;
import variables.StationVariablesSet;

public class InstallChargerRouteModelExtension extends RouteModelExtension {

	BasicVariablesSet variablesSet;
	StationVariablesSet stationVariablesSet;
	
	public InstallChargerRouteModelExtension(BasicVariablesSet variablesSet, StationVariablesSet stationVars) {
		this.variablesSet = variablesSet;
		this.stationVariablesSet = stationVars;
	}
	
	@Override
	public void addConstraintsPerStop(int b, int k, int m, int i, int j) throws IloException {
		IloLinearNumExpr chargingStationsConstraint = cplex.linearNumExpr();
		chargingStationsConstraint.addTerm(1.0, stationVariablesSet.x[i]);
		chargingStationsConstraint.addTerm(-1.0, variablesSet.xBStop[b][k]);
		cplex.addGe(chargingStationsConstraint, 0); 

	}

	@Override
	public void addConstraintsPerStation(int b, int i) throws IloException {
		// TODO Auto-generated method stub

	}
	
	public void setWarmStarts() throws IloException {
		if (stationVariablesSet.initialSolution != null) {
			stationVariablesSet.setWarmStart();
		}
		if (variablesSet.initialSolution != null) {
			variablesSet.setWarmStart();
		}
	};

}
