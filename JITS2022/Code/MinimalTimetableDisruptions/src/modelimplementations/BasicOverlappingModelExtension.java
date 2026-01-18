package modelimplementations;
import ilog.concert.IloException;
import ilog.concert.IloLinearNumExpr;
import ilog.cplex.IloCplex;
import modelinterface.OverlappingModelExtension;
import variables.BasicVariablesSet;
import variables.OverlappingVariablesSet;
import variables.StationChargeVariablesSet;

public class BasicOverlappingModelExtension extends OverlappingModelExtension {
	
	OverlappingVariablesSet overlappingVariablesSet;
	StationChargeVariablesSet routeVariablesSet;
	
	public BasicOverlappingModelExtension(OverlappingVariablesSet overlappingVariablesSet,
			StationChargeVariablesSet routeVariablesSet) {
		this.overlappingVariablesSet = overlappingVariablesSet;
		this.routeVariablesSet = routeVariablesSet;
	}

	@Override
	public void addConstraintsPerStation(int b, int d, int i, int j) throws IloException {
		IloLinearNumExpr useSameStationConstraint1 = cplex.linearNumExpr();
		//System.out.printf("b:%s, d:%s, i:%s, j:%s\n", bu, d, k, m);
		useSameStationConstraint1.addTerm(1.0, overlappingVariablesSet.Z.get(b).get(d).get(i));
		useSameStationConstraint1.addTerm(-1.0, routeVariablesSet.xBStation[b][i]);
		cplex.addLe(useSameStationConstraint1, 0);
		IloLinearNumExpr useSameStationConstraint2 = cplex.linearNumExpr();
		useSameStationConstraint2.addTerm(1.0, overlappingVariablesSet.Z.get(b).get(d).get(i));
		useSameStationConstraint2.addTerm(-1.0, routeVariablesSet.xBStation[d][j]);
		cplex.addLe(useSameStationConstraint2, 0);
		IloLinearNumExpr useSameStationConstraint3 = cplex.linearNumExpr();
		useSameStationConstraint3.addTerm(-1.0, overlappingVariablesSet.Z.get(b).get(d).get(i));
		useSameStationConstraint3.addTerm(1.0, routeVariablesSet.xBStation[d][j]);
		useSameStationConstraint3.addTerm(1.0, routeVariablesSet.xBStation[b][i]);
		cplex.addLe(useSameStationConstraint3, 1);	
		
	}
	
	@Override
	public void addConstraintsPerStop(int b, int d, int k, int m, int i, int j) throws IloException {
		IloLinearNumExpr sameStationSameTimeConstraint3 = cplex.linearNumExpr();
		sameStationSameTimeConstraint3.addTerm(1.0, overlappingVariablesSet.zbd.get(b).get(d).get(k).get(m));
		sameStationSameTimeConstraint3.addTerm(1.0, overlappingVariablesSet.zdb.get(b).get(d).get(k).get(m));
		sameStationSameTimeConstraint3.addTerm(1.0, overlappingVariablesSet.Z.get(b).get(d).get(i));
		cplex.addLe(sameStationSameTimeConstraint3, 2);	                        // CONSTRAINT(13)
		
		/*
		IloLinearNumExpr sameStationSameTimeConstraint4 = cplex.linearNumExpr();
		sameStationSameTimeConstraint4.addTerm(1.0, zbd.get(zbd.size() - 1));
		sameStationSameTimeConstraint4.addTerm(1.0, zdb.get(zdb.size() - 1));
		sameStationSameTimeConstraint4.addTerm(1.0, Z.get(bu).get(d).get(i).get(j));
		cplex.addGe(sameStationSameTimeConstraint4, 2);	                        // CONSTRAINT(13 + 1)
		*/

	}

	@Override
	public void defineVariables(IloCplex cplex) throws IloException {
		overlappingVariablesSet.setCplex(cplex);
		overlappingVariablesSet.defineVariables();
	}

	

}
