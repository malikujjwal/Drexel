public class Addition extends OperatorComponent {
	private final static String OPERATOR_NAME = "addition";

	public Addition(Component left, Component right) {
		super(OPERATOR_NAME, left, right);
	}
}