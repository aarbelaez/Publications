package heuristics;

public class Stop {
	public int bus;
	public int stop;
	public Double arrivalTime;
	
	public Stop(int bus, int stop, double arrivalTime) {
		this.bus = bus;
		this.stop = stop;
		this.arrivalTime = arrivalTime;
	}
	
	public void print() {
		System.out.printf("[%s][%s]\n", bus, stop);
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj)
			return true;
		if (obj == null)
			return false;
		if (getClass() != obj.getClass())
			return false;
		Stop other = (Stop) obj;
		if (bus != other.bus)
			return false;
		if (stop != other.stop)
			return false;
		return true;
	}
	
	
}
