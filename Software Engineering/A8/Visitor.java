public interface Visitor {
	int visit(Addition addition);

	int visit(Subtraction subtraction);

	int visit(Number number);
}
