import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class CircularShifter extends Subject implements Observer {
	@Override
	public void update(String line) {
		circularShiftLine(line);
	}

	private void circularShiftLine(String currentLine) {
		List<String> words = getWordsInLine(currentLine);
		for (int i = 0; i < words.size(); i++) {
			String rotatedLine = shiftWordsByIndex(words, i);
			notifyAllObservers(rotatedLine);
		}
	}

	private String shiftWordsByIndex(List<String> words, int rotationIndex) {
		List<String> wordsToRotate = new ArrayList<>(words);
		Collections.rotate(wordsToRotate, rotationIndex);
		String shiftedLine = String.join(" ", wordsToRotate);
		return shiftedLine;
	}

	private List<String> getWordsInLine(String splitLine) {
		List<String> words = Arrays.asList(splitLine.split(" "));
		return words;
	}
}