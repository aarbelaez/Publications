package modelimplementations;
import ilog.concert.IloException;
import ilog.concert.IloLinearNumExpr;
import modelinterface.RouteModelExtension;
import variables.BasicVariablesSet;

public class LaurasModelExtension extends RouteModelExtension {

	BasicVariablesSet variablesSet;
	
	public LaurasModelExtension(BasicVariablesSet variablesSet) {
		this.variablesSet = variablesSet;
	}

	@Override
	public void addConstraintsPerStop(int b, int k, int m, int i, int j) throws IloException {
		IloLinearNumExpr limitedChargeConstraint = cplex.linearNumExpr();
		limitedChargeConstraint.addTerm(1.0, variablesSet.c[b][k]);
		limitedChargeConstraint.addTerm(1.0, variablesSet.e[b][k]);
		cplex.addLe(limitedChargeConstraint, instance.Cmax); // Constraint (15)

	}

	@Override
	public void addConstraintsPerStation(int b, int i) throws IloException {
		// TODO Auto-generated method stub

	}
	
	public void setWarmStarts() throws IloException {
		if (variablesSet.initialSolution != null) {
			//variablesSet.setWarmStart();
		}
	};

}
