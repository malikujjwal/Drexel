import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

public class Alphabetizer implements Observer {
	private TreeSet<String> lineSorter;

	Alphabetizer() {
		lineSorter = new TreeSet<>(String.CASE_INSENSITIVE_ORDER);
	}

	@Override
	public void update(String line) {
		this.lineSorter.add(line);
	}

	public List<String> getAlphabetizedLines() {
		List<String> alphabetizedLines = new ArrayList<>(this.lineSorter);
		return alphabetizedLines;
	}
}