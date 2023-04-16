import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

public class Alphabetizer extends Filter {
	private TreeSet<String> lineSorter;

	Alphabetizer(Pipe inPipe, Pipe outPipe) {
		super(inPipe, outPipe);
		lineSorter = new TreeSet<>(String.CASE_INSENSITIVE_ORDER);
	}

	@Override
	void filter() {
		if (inPipe.isNotEmptyOrIsNotClosed()) {
			if (inPipe.hasNext()) {
				addToAlphabetizedSet(inPipe.read());
			}
		} else {
			sendAlphabetizedLines();
			outPipe.close();
			stop();
		}
	}

	private void addToAlphabetizedSet(String line) {
		lineSorter.add(line);
	}

	private void sendAlphabetizedLines() {
		List<String> alphabetizedLines = new ArrayList<>(lineSorter);
		for (String line : alphabetizedLines) {
			outPipe.write(line);
		}
	}
}