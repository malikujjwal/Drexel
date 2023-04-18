public abstract class Pipe {
	protected boolean closed;

	public Pipe() {
		this.closed = false;
	}

	abstract void write(String string);

	abstract String read();

	abstract boolean hasNext();

	abstract boolean isNotEmptyOrIsNotClosed();

	void close() {
		closed = true;
	}
}
