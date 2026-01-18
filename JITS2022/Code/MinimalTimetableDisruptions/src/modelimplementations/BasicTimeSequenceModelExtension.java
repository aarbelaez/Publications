package modelimplementations;
import ilog.concert.IloException;
import ilog.concert.IloLinearNumExpr;
import modelinterface.RouteModelExtension;
import variables.BasicVariablesSet;

public class BasicTimeSequenceModelExtension extends RouteModelExtension {
	
	BasicVariablesSet variablesSet;
	
	@Override
	public void addConstraintsPerStop(int b, int k, int m, int i, int j) throws IloException {
		IloLinearNumExpr timeContraint = cplex.linearNumExpr();
		timeContraint.addTerm(1.0, variablesSet.arrivalTime[b][k]);
		timeContraint.addTerm(-1.0, variablesSet.arrivalTime[b][m]);
		timeContraint.addTerm(-1.0, variablesSet.ct[b][m]);
		cplex.addGe(timeContraint, instance.T[b][m]); // CONSTRAINT (3)

	}

	@Override
	public void addConstraintsPerStation(int b, int i) throws IloException {
		// TODO Auto-generated method stub

	}

}
