package modelimplementations;
import ilog.concert.IloException;
import ilog.concert.IloLinearNumExpr;
import modelinterface.OverlappingModelExtension;
import variables.BasicVariablesSet;
import variables.OverlappingVariablesSet;

public class SecurityMarginTimeOverlappingModelExtension extends OverlappingModelExtension {

	OverlappingVariablesSet overlappingVariablesSet;
	BasicVariablesSet routeVariablesSet;
	
	public SecurityMarginTimeOverlappingModelExtension(OverlappingVariablesSet overlappingVariablesSet,
			BasicVariablesSet routeVariablesSet) {
		this.overlappingVariablesSet = overlappingVariablesSet;
		this.routeVariablesSet = routeVariablesSet;
	}

	@Override
	public void addConstraintsPerStop(int b, int d, int k, int m, int i, int j) throws IloException {
		IloLinearNumExpr sameStationSameTimeConstraint1 = cplex.linearNumExpr();
		sameStationSameTimeConstraint1.addTerm(1.0, routeVariablesSet.arrivalTime[b][k]);
		sameStationSameTimeConstraint1.addTerm(-1.0, routeVariablesSet.arrivalTime[d][m]);
		sameStationSameTimeConstraint1.addTerm(-1.0, routeVariablesSet.ct[d][m]);
		sameStationSameTimeConstraint1.addTerm(-1 * instance.SM, routeVariablesSet.xBStop[d][m]);
		sameStationSameTimeConstraint1.addTerm(instance.M, overlappingVariablesSet.zbd.get(b).get(d).get(k).get(m));	
		cplex.addGe(sameStationSameTimeConstraint1, 0);     // CONSTRAINT(11)
		//System.out.println(paths[bu].length + " " + paths[d].length);
		//System.out.printf("b:%s, d:%s, i:%s, j:%s, k:%s, m:%s, originalTimetable[b][k]=%s, originalTimetable[d][m]=%s  \n",
		//	bu, d, i, j,  k, m, originalTimetable[bu][k], originalTimetable[d][m]);
		IloLinearNumExpr sameStationSameTimeConstraint2 = cplex.linearNumExpr();
		sameStationSameTimeConstraint2.addTerm(1.0, routeVariablesSet.arrivalTime[d][m]);
		sameStationSameTimeConstraint2.addTerm(-1.0, routeVariablesSet.arrivalTime[b][k]);
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
