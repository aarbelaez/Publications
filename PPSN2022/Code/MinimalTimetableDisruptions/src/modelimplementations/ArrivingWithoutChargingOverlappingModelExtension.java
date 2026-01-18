package modelimplementations;
import ilog.concert.IloException;
import ilog.concert.IloLinearNumExpr;
import modelinterface.OverlappingModelExtension;
import variables.BasicVariablesSet;
import variables.OverlappingVariablesSet;
import variables.StartingChargeTimeVariablesSet;

public class ArrivingWithoutChargingOverlappingModelExtension extends OverlappingModelExtension {
	
	BasicVariablesSet routeVariablesSet;
	OverlappingVariablesSet overlappingVariablesSet;
	StartingChargeTimeVariablesSet startingChargeTimeVariablesSet;
	
	
	
	public ArrivingWithoutChargingOverlappingModelExtension(BasicVariablesSet routeVariablesSet,
			OverlappingVariablesSet overlappingVariablesSet,
			StartingChargeTimeVariablesSet startingChargeTimeVariablesSet) {
		this.routeVariablesSet = routeVariablesSet;
		this.overlappingVariablesSet = overlappingVariablesSet;
		this.startingChargeTimeVariablesSet = startingChargeTimeVariablesSet;
	}

	@Override
	public void addConstraintsPerStop(int b, int d, int k, int m, int i, int j) throws IloException {
		IloLinearNumExpr sameStationSameTimeConstraint1 = cplex.linearNumExpr();
		sameStationSameTimeConstraint1.addTerm(1.0, startingChargeTimeVariablesSet.st[b][k]);
		sameStationSameTimeConstraint1.addTerm(-1.0, startingChargeTimeVariablesSet.st[d][m]);
		sameStationSameTimeConstraint1.addTerm(-1.0, routeVariablesSet.ct[d][m]);
		sameStationSameTimeConstraint1.addTerm(-1 * instance.SM, routeVariablesSet.xBStop[d][m]);
		sameStationSameTimeConstraint1.addTerm(instance.M, overlappingVariablesSet.zbd.get(b).get(d).get(k).get(m));	
		cplex.addGe(sameStationSameTimeConstraint1, 0);     // CONSTRAINT(11)
		IloLinearNumExpr sameStationSameTimeConstraint2 = cplex.linearNumExpr();
		sameStationSameTimeConstraint2.addTerm(1.0, startingChargeTimeVariablesSet.st[d][m]);
		sameStationSameTimeConstraint2.addTerm(-1.0, startingChargeTimeVariablesSet.st[b][k]);
		sameStationSameTimeConstraint2.addTerm(-1.0, routeVariablesSet.ct[b][k]);
		sameStationSameTimeConstraint2.addTerm(-1 * instance.SM, routeVariablesSet.xBStop[b][k]);
		sameStationSameTimeConstraint2.addTerm(instance.M, overlappingVariablesSet.zdb.get(b).get(d).get(k).get(m));
		cplex.addGe(sameStationSameTimeConstraint2, 0);   // CONSTRAINT(12)

	}

	@Override
	public void addConstraintsPerStation(int b, int d, int i, int j) throws IloException {
		// TODO Auto-generated method stub

	}

}
