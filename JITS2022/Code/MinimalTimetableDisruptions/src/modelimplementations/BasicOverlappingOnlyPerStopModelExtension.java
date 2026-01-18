package modelimplementations;

import ilog.concert.IloException;
import ilog.concert.IloLinearNumExpr;
import ilog.cplex.IloCplex;
import modelinterface.OverlappingModelExtension;
import variables.BasicVariablesSet;
import variables.OverlappingOnlyPerStopVariablesSet;
import variables.OverlappingVariablesSet;

public class BasicOverlappingOnlyPerStopModelExtension extends OverlappingModelExtension {

	OverlappingOnlyPerStopVariablesSet overlappingVariablesSet;
	BasicVariablesSet routeVariablesSet;
	
	public BasicOverlappingOnlyPerStopModelExtension(OverlappingOnlyPerStopVariablesSet overlappingVariablesSet,
			BasicVariablesSet routeVariablesSet) {
		this.overlappingVariablesSet = overlappingVariablesSet;
		this.routeVariablesSet = routeVariablesSet;
	}

	@Override
	public void addConstraintsPerStation(int b, int d, int i, int j) throws IloException {
		
		
	}
	
	@Override
	public void addConstraintsPerStop(int b, int d, int k, int m, int i, int j) throws IloException {
		
		IloLinearNumExpr useSameStationConstraint1 = cplex.linearNumExpr();
		//System.out.printf("b:%s, d:%s, i:%s, j:%s\n", bu, d, k, m);
		useSameStationConstraint1.addTerm(1.0, overlappingVariablesSet.Z.get(b).get(d).get(k).get(m));
		useSameStationConstraint1.addTerm(-1.0, routeVariablesSet.xBStop[b][k]);
		cplex.addLe(useSameStationConstraint1, 0);
		IloLinearNumExpr useSameStationConstraint2 = cplex.linearNumExpr();
		useSameStationConstraint2.addTerm(1.0, overlappingVariablesSet.Z.get(b).get(d).get(k).get(m));
		useSameStationConstraint2.addTerm(-1.0, routeVariablesSet.xBStop[d][m]);
		cplex.addLe(useSameStationConstraint2, 0);
		IloLinearNumExpr useSameStationConstraint3 = cplex.linearNumExpr();
		useSameStationConstraint3.addTerm(-1.0, overlappingVariablesSet.Z.get(b).get(d).get(k).get(m));
		useSameStationConstraint3.addTerm(1.0, routeVariablesSet.xBStop[d][m]);
		useSameStationConstraint3.addTerm(1.0, routeVariablesSet.xBStop[b][k]);
		cplex.addLe(useSameStationConstraint3, 1);	
		
		
		
		IloLinearNumExpr sameStationSameTimeConstraint3 = cplex.linearNumExpr();
		sameStationSameTimeConstraint3.addTerm(1.0, overlappingVariablesSet.zbd.get(b).get(d).get(k).get(m));
		sameStationSameTimeConstraint3.addTerm(1.0, overlappingVariablesSet.zdb.get(b).get(d).get(k).get(m));
		sameStationSameTimeConstraint3.addTerm(1.0, overlappingVariablesSet.Z.get(b).get(d).get(k).get(m));
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
