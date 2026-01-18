package modelimplementations;
import java.util.HashMap;
import java.util.LinkedHashMap;

import core.ModelMTD;
import ilog.concert.IloException;
import ilog.concert.IloLinearNumExpr;
import ilog.concert.IloNumVar;
import ilog.cplex.IloCplex;
import ilog.cplex.IloCplex.UnknownObjectException;
import modelinterface.RouteModelExtension;
import utils.ToolsMTD;
import variables.BasicVariablesSet;
import variables.StationVariablesSet;

public class BasicRouteModelExtension extends RouteModelExtension {

	

	BasicVariablesSet variablesSet;
	StationVariablesSet stationVariablesSet;
	int objectiveType;
	
	boolean setObjectiveStation = true;
	

	String prefixVarname = "";
	

	public BasicRouteModelExtension(BasicVariablesSet variablesSet, StationVariablesSet stationVars, int objectiveType) {
		this.variablesSet = variablesSet;
		this.stationVariablesSet = stationVars;
		this.objectiveType = objectiveType;
	}

	@Override
	public void defineVariables(IloCplex cplex) throws IloException {
		variablesSet.setCplex(cplex);
		variablesSet.defineVariables();
		stationVariablesSet.setCplex(cplex);
		stationVariablesSet.defineVariables();
	}
	

	/*
	@Override
	
	public HashMap<String, Double> getPrintableVarVals(int b, int k) {
		// TODO Auto-generated method stub
		
	}
	*/

	@Override
	public void addObjective(IloLinearNumExpr objective) throws IloException {
		
		if (objectiveType == ModelMTD.CHARGERS_OBJ && setObjectiveStation) {
			System.out.println("Optimizing number of chargers");
			for (int i = 0; i < instance.n; i++) {
				objective.addTerm(1.0, stationVariablesSet.x[i]);
			}
		} else if (objectiveType == ModelMTD.TIME_OBJ && setObjectiveStation) {
			System.out.println("Optimizing time deviations");
			for (int i = 0; i < instance.b; i++) {
				for (int j = 0; j < instance.paths[i].length; j++) {
					objective.addTerm(1.0, variablesSet.deviationTime[i][j]);
				}
			}
		} else if (objectiveType == ModelMTD.CHARGES_OBJ && setObjectiveStation) {
			System.out.println("Optimizing number of charges");
			for (int i = 0; i < instance.b; i++) {
				for (int j = 0; j < instance.paths[i].length; j++) {
					objective.addTerm(1.0, variablesSet.xBStop[i][j]);
				}
			}	
		/*} else if (objectiveType == ModelMTD.CHARGERS_PER_STATION_OBJ && setObjectiveStation) {
			System.out.println("Optimizing number of chargers per bus");
			for (int i = 0; i < instance.b; i++) {
				for (int j = 0; j < instance.n; j++) {
					objective.addTerm(1.0, variablesSet.xBStation[i][j]);
				}
			}	*/
		} else if (objectiveType == ModelMTD.BOTH_CHARGERS_CHARGES_OBJ) {
			System.out.println("Optimizing both number of chargers and charges per bus");
			int alfa = 100000; 
			if (setObjectiveStation) {
				for (int i = 0; i < instance.n; i++) {
					objective.addTerm(alfa, stationVariablesSet.x[i]);
				}
			}
			for (int i = 0; i < instance.b; i++) {
				for (int j = 0; j < instance.paths[i].length; j++) {
					objective.addTerm(1.0, variablesSet.xBStop[i][j]);
				}
			}		
		} else if (objectiveType == ModelMTD.MINIMAX_TIME_OBJ && setObjectiveStation) {
			System.out.println("Optimizing max time deviations");
			IloNumVar[] z = cplex.numVarArray(instance.b, 0, instance.DTmax);
			for (int bu = 0; bu < instance.b; bu++) {
				for (int k = 1; k < instance.paths[bu].length; k++) {
					IloLinearNumExpr maxConstraint = cplex.linearNumExpr();
					maxConstraint.addTerm(1.0, z[bu]);
					maxConstraint.addTerm(-1.0, variablesSet.deviationTime[bu][k]);
					cplex.addGe(maxConstraint, 0);	
				}
				objective.addTerm(1.0, z[bu]);
			}
		}
	}

	@Override
	public void addConstraintsPerStop(int b, int k, int m, int i, int j) throws IloException {
		IloLinearNumExpr capacityContraint = cplex.linearNumExpr();
		capacityContraint.addTerm(1.0, variablesSet.c[b][k]);
		capacityContraint.addTerm(-1.0, variablesSet.c[b][m]);
		capacityContraint.addTerm(-1.0, variablesSet.e[b][m]);
		cplex.addLe(capacityContraint, -1*instance.D[i][j]); // CONSTRAINT (2)
		
		IloLinearNumExpr deltaTimeConstraint1 = cplex.linearNumExpr();
		deltaTimeConstraint1.addTerm(-1.0, variablesSet.arrivalTime[b][k]);
		deltaTimeConstraint1.addTerm(1.0, variablesSet.deviationTime[b][k]);
		cplex.addGe(deltaTimeConstraint1, -1*instance.originalTimetable[b][k]); // CONSTRAINT (4)
		IloLinearNumExpr deltaTimeConstraint2 = cplex.linearNumExpr();
		deltaTimeConstraint2.addTerm(1.0, variablesSet.arrivalTime[b][k]);
		deltaTimeConstraint2.addTerm(1.0, variablesSet.deviationTime[b][k]);
		cplex.addGe(deltaTimeConstraint2, instance.originalTimetable[b][k]); // CONSTRAINT (5)		
		
	    IloLinearNumExpr maxChargingTimeConstraint = cplex.linearNumExpr();
	    maxChargingTimeConstraint.addTerm(instance.maxChargingTime, variablesSet.xBStop[b][k]);
	    maxChargingTimeConstraint.addTerm(-1.0, variablesSet.ct[b][k]);
	    cplex.addGe(maxChargingTimeConstraint, 0); // CONSTRAINT (6)
		 					
		// ADDED by me
	    
	    
		IloLinearNumExpr relationTimeCapacityConstraint = cplex.linearNumExpr();
		relationTimeCapacityConstraint.addTerm(instance.chargingRate, variablesSet.ct[b][k]);
		relationTimeCapacityConstraint.addTerm(-1.0, variablesSet.e[b][k]);
		cplex.addGe(relationTimeCapacityConstraint, 0);	// CONSTRAINT (14)
		
	}



	@Override
	public void addConstraintsPerStation(int b, int i) throws IloException {
		
		
	}
	
	@Override
	public void addConstraintsPerBus(int b) throws IloException {
		// Buses full at the beginning
		IloLinearNumExpr busesFullBeggining = cplex.linearNumExpr();
		busesFullBeggining.addTerm(1.0, variablesSet.c[b][0]);
		cplex.addEq(busesFullBeggining, instance.Cmax);
		// e is 0 at the beginning
		IloLinearNumExpr busesEBeggining = cplex.linearNumExpr();
		busesEBeggining = cplex.linearNumExpr();
		busesEBeggining.addTerm(1.0, variablesSet.e[b][0]);
		cplex.addEq(busesEBeggining, 0);
		// xBStop is 0 at the beginning
		IloLinearNumExpr busesXBStopBeggining = cplex.linearNumExpr();
		busesXBStopBeggining = cplex.linearNumExpr();
		busesXBStopBeggining.addTerm(1.0, variablesSet.xBStop[b][0]);
		cplex.addEq(busesXBStopBeggining, 0);
		
	}
	
	
	
	@Override
	public HashMap<String, Double> getPrintableVarVals(int b, int k, int i) throws UnknownObjectException, IloException {
		HashMap<String, Double> results = new LinkedHashMap<String, Double>();
		results.put("arrivalTime", cplex.getValue(variablesSet.arrivalTime[b][k]));
		results.put("c", cplex.getValue(variablesSet.c[b][k]));
		results.put("e", cplex.getValue(variablesSet.e[b][k]));
		results.put("ct", cplex.getValue(variablesSet.ct[b][k]));
		results.put("xBStop", cplex.getValue(variablesSet.xBStop[b][k]));
		try {
			results.put("x", cplex.getValue(stationVariablesSet.x[i]));
			//results.put("dt", cplex.getValue(variablesSet.deviationTime[b][k]));
		} catch (ilog.cplex.IloCplex.UnknownObjectException e) {
			System.out.printf("b:%s, stop:%s, station:%s\n", b, k, i);
			results.put("x", 0.0);
			e.printStackTrace();
		}
		
		//results.put("xs", cplex.getValue(variablesSet.xBStation[b][i]));
		return results;
	}

	public void setObjectiveType(int objectiveType) {
		this.objectiveType = objectiveType;
	}
	
	public void setSetObjectiveStation(boolean setObjectiveStation) {
		this.setObjectiveStation = setObjectiveStation;
	}


}
