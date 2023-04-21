import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public abstract class Subject {

	List<Observer> observers = Collections.synchronizedList(new ArrayList<Observer>());

	public void attach(Observer observer) {
		observers.add(observer);
	}

	public void notifyAllObservers(Message message) {
		for (Observer observer : observers) {
			observer.update(message);
		}
	}
}