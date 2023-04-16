import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

public class Pipe {

	private Queue<String> queue;
	private boolean closed;

	public Pipe() {
		queue = new ConcurrentLinkedQueue<>();
		closed = false;
	}

	void write(String string) {
		queue.offer(string);
	}

	String read() {
		return queue.poll();
	}

	boolean isNotEmptyOrIsNotClosed() {
		return !queue.isEmpty() || !closed;
	}

	boolean hasNext() {
		return !queue.isEmpty();
	}

	void close() {
		closed = true;
	}
}