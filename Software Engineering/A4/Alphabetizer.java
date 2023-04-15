import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

public class Alphabetizer {
	private TreeSet<String> lineSorter;

	Alphabetizer() {
		lineSorter = new TreeSet<>(String.CASE_INSENSITIVE_ORDER);
	}

	public void addToAlphabetizedSet(String line) {
		lineSorter.add(line);
	}

	public List<String> getAlphabetizedLines() {
		List<String> alphabetizedLines = new ArrayList<>(this.lineSorter);
		return alphabetizedLines;
	}
}