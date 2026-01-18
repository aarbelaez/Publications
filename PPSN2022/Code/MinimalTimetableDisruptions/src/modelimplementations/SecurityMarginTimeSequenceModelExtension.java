package modelimplementations;
import ilog.concert.IloException;
import ilog.concert.IloLinearNumExpr;
import modelinterface.RouteModelExtension;
import variables.BasicVariablesSet;

public class SecurityMarginTimeSequenceModelExtension extends RouteModelExtension {
	
	BasicVariablesSet variablesSet;
	
	public SecurityMarginTimeSequenceModelExtension(BasicVariablesSet variablesSet) {
		this.variablesSet = variablesSet;
	}

	@Override
	public void addConstraintsPerStop(int b, int k, int m, int i, int j) throws IloException {
		IloLinearNumExpr timeContraint = cplex.linearNumExpr();
		timeContraint.addTerm(1.0, variablesSet.arrivalTime[b][k]);
		timeContraint.addTerm(-1.0, variablesSet.arrivalTime[b][m]);
		timeContraint.addTerm(-1.0, variablesSet.ct[b][m]);
		cplex.addGe(timeContraint, instance.T[b][m]); // CONSTRAINT (3)		
		
		if (i == j) {
			IloLinearNumExpr securityMarginContraint = cplex.linearNumExpr();
			securityMarginContraint.addTerm(1.0, variablesSet.arrivalTime[b][k]);
			securityMarginContraint.addTerm(-1.0, variablesSet.arrivalTime[b][m]);
			securityMarginContraint.addTerm(-1.0, variablesSet.ct[b][m]);
			securityMarginContraint.addTerm(-1*instance.SM, variablesSet.xBStop[b][m]);
			cplex.addGe(securityMarginContraint, 0); // CONSTRAINT (17)	
		}
	}

	@Override
	public void addConstraintsPerStation(int b, int i) throws IloException {
		// TODO Auto-generated method stub

	}

}
