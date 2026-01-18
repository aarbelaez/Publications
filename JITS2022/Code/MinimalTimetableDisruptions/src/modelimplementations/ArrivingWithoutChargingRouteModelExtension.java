package modelimplementations;
import ilog.concert.IloException;
import ilog.concert.IloLinearNumExpr;
import ilog.cplex.IloCplex;
import modelinterface.RouteModelExtension;
import variables.BasicVariablesSet;
import variables.StartingChargeTimeVariablesSet;

public class ArrivingWithoutChargingRouteModelExtension extends RouteModelExtension {

	BasicVariablesSet basicVariablesSet;
	StartingChargeTimeVariablesSet startingChargeTimeVariablesSet;
	
	public ArrivingWithoutChargingRouteModelExtension(BasicVariablesSet basicVariablesSet,
			StartingChargeTimeVariablesSet startingChargeTimeVariablesSet) {
		this.basicVariablesSet = basicVariablesSet;
		this.startingChargeTimeVariablesSet = startingChargeTimeVariablesSet;
	}

	@Override
	public void addConstraintsPerStop(int b, int k, int m, int i, int j) throws IloException {
		IloLinearNumExpr timeContraint = cplex.linearNumExpr();
		timeContraint.addTerm(1.0, basicVariablesSet.arrivalTime[b][k]);
		timeContraint.addTerm(-1.0, startingChargeTimeVariablesSet.st[b][m]);
		timeContraint.addTerm(-1.0, basicVariablesSet.ct[b][m]);
		if (instance.T[b][m] < instance.SM) { // security margin
			timeContraint.addTerm(-1*instance.SM, basicVariablesSet.xBStop[b][m]);
			cplex.addGe(timeContraint, 0);
		} else {
			cplex.addGe(timeContraint, instance.T[b][m]); // CONSTRAINT (3)	
		}		
		
		/*
		IloLinearNumExpr securityMarginContraint = cplex.linearNumExpr();
		securityMarginContraint.addTerm(1.0, arrivalTime[bu][k]);
		securityMarginContraint.addTerm(-1.0, arrivalTime[bu][m]);
		securityMarginContraint.addTerm(-1.0, ct[bu][m]);
		securityMarginContraint.addTerm(-1*instance.SM, xBStop[bu][m]);
		cplex.addGe(securityMarginContraint, 0); // CONSTRAINT (17)	
		*/
		
		
		IloLinearNumExpr startingChargeTimeConstraint= cplex.linearNumExpr();
		startingChargeTimeConstraint.addTerm(1.0, basicVariablesSet.arrivalTime[b][k]);
		startingChargeTimeConstraint.addTerm(-1.0, startingChargeTimeVariablesSet.st[b][k]);
		cplex.addLe(startingChargeTimeConstraint, 0);

	}

	@Override
	public void addConstraintsPerStation(int b, int i) throws IloException {
		// TODO Auto-generated method stub

	}
	
	@Override
	public void defineVariables(IloCplex cplex) throws IloException {
		startingChargeTimeVariablesSet.setCplex(cplex);
		startingChargeTimeVariablesSet.defineVariables();
	}

}
