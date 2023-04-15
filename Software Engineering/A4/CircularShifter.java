import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class CircularShifter extends Colleague {
	CircularShifter(Mediator mediator) {
		super(mediator);
	}

	public void shiftLine(String currentLine) {
		List<String> words = getWordsInLine(currentLine);
		for (int i = 0; i < words.size(); i++) {
			String rotatedLine = shiftWordsByIndex(words, i);
			mediator.shiftedLine(rotatedLine);
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