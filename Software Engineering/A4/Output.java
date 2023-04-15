import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

public class Output {
	private FileHandler fileHandler;

	Output() {
		fileHandler = new FileHandler();
		fileHandler.setFileName("kwic_output.txt");
	}

	public void write(List<String> lines) {
		try {
			writeFile(lines);
		} catch (IOException e) {
			fileHandler.logError(e);
		}
	}

	private void writeFile(List<String> lines) throws IOException {
		String fileName = fileHandler.getFileName();
		Files.write(Path.of(fileName), lines, Charset.defaultCharset());
	}
}