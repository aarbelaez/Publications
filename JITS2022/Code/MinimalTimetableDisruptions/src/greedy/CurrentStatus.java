package greedy;

/**
 * The current status in the greedy execution. Current time and battery of a bus
 * @author cedaloaiza
 *
 */
public class CurrentStatus {
	public double currentTime;
	public double currentEnergy;
	
	public void reset() {
		currentTime = 0;
		currentEnergy = 0;
	}
}
