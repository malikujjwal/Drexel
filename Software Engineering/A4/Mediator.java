public interface Mediator {
	void inputProvidesNextLine(String line);

	void inputIsFinished();

	void shiftedLine(String line);
}
