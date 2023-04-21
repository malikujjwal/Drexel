import java.util.ArrayList;
import java.util.Arrays;

public class MasterControllerMulti {
	static ArrayList<String> test = new ArrayList<String>(
			Arrays.asList("Descent of Man", "The Ascent of Man", "The Old Man and The Sea"));

	public static void main(String[] args) {
		ServerController serverController = new ServerController();
		serverController.start();

		ClientController clientController1 = new ClientController("Client-1");
		ClientController clientController2 = new ClientController("Client-2");
		ClientController clientController3 = new ClientController("Client-3");
		ClientController clientController4 = new ClientController("Client-4");

		clientController1.addLines(test);
		clientController2.addLines(test);
		clientController3.addLines(test);
		clientController4.addLines(test);
		clientController1.startClient();
		clientController2.startClient();
		clientController3.startClient();
		clientController4.startClient();
	}
}
