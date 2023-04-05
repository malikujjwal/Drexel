import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

public class Alphabetizer {
	public List<String> sort(List<String> lines) {
		List<String> alphabetizedLines = sortLines(lines);
		return alphabetizedLines;
	}

	private List<String> sortLines(List<String> lines) {
		TreeSet<String> lineSorter = new TreeSet<String>(String.CASE_INSENSITIVE_ORDER);
		lineSorter.addAll(lines);
		List<String> sortedLines = new ArrayList<>(lineSorter);
		return sortedLines;
	}
}