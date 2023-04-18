import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class CircularShifter extends Filter {
	CircularShifter(Pipe inPipe, Pipe outPipe) {
		super(inPipe, outPipe);
	}

	@Override
	void filter() {
		if (inPipe.isNotEmptyOrIsNotClosed()) {
			if (inPipe.hasNext()) {
				circularShiftLine(inPipe.read());
			}
		} else {
			outPipe.close();
			stop();
		}
	}

	private void circularShiftLine(String currentLine) {
		List<String> words = getWordsInLine(currentLine);
		for (int i = 0; i < words.size(); i++) {
			String rotatedLine = shiftWordsByIndex(words, i);
			outPipe.write(rotatedLine);
		}
	}

	private String shiftWordsByIndex(List<String> words, int rotationIndex) {
		List<String> wordsToRotate = new ArrayList<>(words);
		Collections.rotate(wordsToRotate, rotationIndex);
		return String.join(" ", wordsToRotate);
	}

	private List<String> getWordsInLine(String splitLine) {
		return Arrays.asList(splitLine.split(" "));
	}
}