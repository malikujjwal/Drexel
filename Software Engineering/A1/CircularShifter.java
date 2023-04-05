import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class CircularShifter {

	public List<String> shiftLines(List<String> lines) {
		return getCircularShiftLines(lines);
	}

	private List<String> getCircularShiftLines(List<String> lines) {
		List<String> circularShifts = new ArrayList<String>(lines);
		for (String currentLine : lines) {
			List<String> circularShiftResult = shiftSingleLine(currentLine);
			circularShifts.addAll(circularShiftResult);
		}
		return circularShifts;
	}

	private List<String> shiftSingleLine(String currentLine) {
		List<String> rotatedLines = new ArrayList<String>();
		List<String> words = getWordsInLine(currentLine);
		for (int i = 1; i < words.size(); i++) {
			String rotatedLine = shiftWordsByIndex(words, i);
			rotatedLines.add(rotatedLine);
		}
		return rotatedLines;
	}

	private String shiftWordsByIndex(List<String> words, int rotationIndex) {
		List<String> wordsToRotate = new ArrayList<>(words);
		Collections.rotate(wordsToRotate, rotationIndex);
		String ShiftedLine = String.join(" ", wordsToRotate);
		return ShiftedLine;
	}

	private List<String> getWordsInLine(String splitLine) {
		List<String> words = Arrays.asList(splitLine.split(" "));
		return words;
	}
}