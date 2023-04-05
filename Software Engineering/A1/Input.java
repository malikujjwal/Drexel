import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class Input extends FileHandler {
	Input() {
		setFileName("kwic.txt");
	}

	public List<String> read() {
		List<String> linesInFile = new ArrayList<String>();
		try {
			linesInFile = readFileFromName();
		} catch (IOException e) {
			logError(e);
		}
		return linesInFile;
	}

	private List<String> readFileFromName() throws IOException {
		String fileName = getFileName();
		List<String> lines = Files.readAllLines(Paths.get(fileName), StandardCharsets.UTF_8);
		return lines;
	}
}