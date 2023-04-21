import java.util.ArrayList;
import java.util.Arrays;

public class MasterControllerSingle {
	static ArrayList<String> test = new ArrayList<String>(Arrays.asList("Descent of Man", "The Ascent of Man"));

	public static void main(String[] args) {
		new SocketConfigParser("localhost", 8080);

		ServerController serverController = new ServerController();
		serverController.start();

		ClientController clientController = new ClientController("Client-1");
		clientController.addLines(test);
		clientController.startClient();
	}
}
