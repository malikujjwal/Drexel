import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;

public class PipeQueue extends Pipe {
	private Queue<String> queue;

	public PipeQueue() {
		queue = new ConcurrentLinkedQueue<>();
	}

	@Override
	void write(String string) {
		queue.offer(string);
	}

	@Override
	String read() {
		return queue.poll();
	}

	@Override
	boolean hasNext() {
		return !queue.isEmpty();
	}

	@Override
	boolean isNotEmptyOrIsNotClosed() {
		return !queue.isEmpty() || !closed;
	}
}