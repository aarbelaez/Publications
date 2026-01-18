package variables;
import ilog.concert.IloException;

public abstract class RouteVariablesSet extends VariablesSet {
	
	
	@Override
	public void defineVariables() throws IloException {
		if (!areVariablesDefined) {
			defineBasicVariablesArrays();
			for (int b = 0; b < instance.b; b++) {		
				defineBasicVariablesPerBus(b);
				for (int i = 0; i < instance.paths[b].length; i++) {
					defineBasicVariablesPerBusPerStop(b, i);
				}
			}
			areVariablesDefined = true;
		}
		
	}
	
	public abstract void defineBasicVariablesArrays() throws IloException;
	
	public abstract void defineBasicVariablesPerBus(int b) throws IloException;
	
	public abstract void defineBasicVariablesPerBusPerStop(int b, int k) throws IloException;


}
