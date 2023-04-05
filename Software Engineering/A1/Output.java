import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

public class Output extends FileHandler {
	Output() {
		setFileName("kwic_output.txt");
	}

	public void write(List<String> lines) {
		try {
			writeFile(lines);
		} catch (IOException e) {
			logError(e);
		}
	}

	private void writeFile(List<String> lines) throws IOException {
		String fileName = getFileName();
		Files.write(Path.of(fileName), lines, Charset.defaultCharset());
	}
}