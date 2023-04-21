import java.util.ArrayList;

public class Worker extends Subject implements Runnable, Observer {
	private final ArrayList<String> outputLines = new ArrayList<>();
	private final IOHandler ioHandler;
	private boolean isInputLinesFinished;
	private CircularShifter circularShifter;
	private Alphabetizer alphabetizer;

	Worker(IOHandler ioHandler) {
		this.ioHandler = ioHandler;
		circularShifter = new CircularShifter();
		alphabetizer = new Alphabetizer();
	}

	private void processMessagesAndNotify(Message message) {
		if (message.isFinished()) {
			// All lines received from alpha
			notifyAllObservers(new Message(true));
			for (String s : outputLines) {
				ioHandler.sendMessage(new Message(s));
			}
			isInputLinesFinished = true;
		} else {
			notifyAllObservers(new Message(message.getString()));
		}
	}

	private void attachSubjects() {
		this.attach(circularShifter);
		circularShifter.attach(alphabetizer);
		alphabetizer.attach(this);
	}

	@Override
	public void run() {
		this.attachSubjects();
		while (!this.isInputLinesFinished) {
			this.processMessagesAndNotify(ioHandler.receiveMessage());
		}
		ioHandler.sendMessage(new Message(true));
	}

	@Override
	public void update(Message message) {
		if (!message.isFinished()) {
			outputLines.add(message.getString());
		}
	}
}
