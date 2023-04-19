public class Number extends Component {
	private final static String NUMBER_NAME = "number";
	private int value;

	public Number(int value) {
		super(NUMBER_NAME);
		this.value = value;
	}

	public int getValue() {
		return this.value;
	}
}