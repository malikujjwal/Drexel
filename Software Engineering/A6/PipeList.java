import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class PipeList extends Pipe {
	private List<String> list;

	public PipeList() {
		list = Collections.synchronizedList(new ArrayList<>());
	}

	@Override
	void write(String string) {
		list.add(string);
	}

	@Override
	String read() {
		return list.remove(0);
	}

	@Override
	boolean isNotEmptyOrIsNotClosed() {
		return !list.isEmpty() || !closed;
	}

	@Override
	boolean hasNext() {
		return !list.isEmpty();
	}
}
