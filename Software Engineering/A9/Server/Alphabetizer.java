import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

public class Alphabetizer extends Subject implements Observer {
	private TreeSet<String> lineSorter;

	Alphabetizer() {
		lineSorter = new TreeSet<>(String.CASE_INSENSITIVE_ORDER);
	}

	@Override
	public void update(Message message) {
		if (!message.isFinished()) {
			lineSorter.add(message.getString());
		} else {
			sendAlphabetizedLines();
			notifyAllObservers(message);
		}
	}

	public void sendAlphabetizedLines() {
		List<String> alphabetizedLines = new ArrayList<>(this.lineSorter);
		for (String line : alphabetizedLines) {
			notifyAllObservers(new Message(line));
		}
	}
}