public class SocketConfigParser extends SocketConfig {
	private static final String DEFAULT_HOST = "127.0.0.1";
	private static final int DEFAULT_PORT = 6666;

	SocketConfigParser() {
		super(DEFAULT_HOST, DEFAULT_PORT);
	}

	SocketConfigParser(String hostName, int port) {
		super(hostName, port);
	}
}
