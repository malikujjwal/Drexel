public class ServerController extends Thread {
	@Override
	public void run() {
		Server server = new ServerHandler();
		server.acceptConnections();
	}
}
