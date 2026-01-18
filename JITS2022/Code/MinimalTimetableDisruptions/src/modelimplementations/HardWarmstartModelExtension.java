package modelimplementations;

import java.util.ArrayList;

import ilog.concert.IloException;
import ilog.concert.IloLinearNumExpr;
import modelinterface.ModelExtension;
import variables.StationVariablesSet;

/**
 * To start with a fixed set of open stations
 * @author cdlq1
 *
 */
public class HardWarmstartModelExtension extends ModelExtension {

	StationVariablesSet stationVariablesSet;
	int[] openStations= {114, 84, 237, 175, 239, 180, 553, 468};
	
	public HardWarmstartModelExtension(StationVariablesSet stationVars) {
		this.stationVariablesSet = stationVars;
	}
	
	@Override
	public void addConstraintsGlobally() throws IloException {
		for (int station: openStations) {
			IloLinearNumExpr hardWarmContraint = cplex.linearNumExpr();
			hardWarmContraint.addTerm(1, stationVariablesSet.x[station]);
			cplex.addEq(hardWarmContraint, 1);
		}
	}
}
